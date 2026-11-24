"""CARD-010. Freeze paired interventions, score them, then verify the three RVs before publication."""
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
from itertools import groupby
import json
import math
import os
from pathlib import Path
import sys
from threading import Timer
from time import perf_counter

import numpy as np
import torch

from mind_din import DIN, pool_scores
from mind_evaluation import Impression, METRICS, evaluate, ranks_from_scores
from mind_two_tower import MeanTowers, build_catalog, exact_top
from profile_mind_small import fields, require
from run_mind_baselines import ClickWindow, TitleIndex, Retrieval, split_prefix
from run_mind_din import batch_logits, sources
from run_mind_two_tower import histories, read_sources, registered_ref, train_epochs, weight_identity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.common import git, head, payload_sha256, sha256_file, snapshot, status_paths
from tools.rv import write_result

CONFIG = "configs/mind-evaluation-audit-v0.1.json"
GROUPS = ("uniform5", "uniform20", "weighted5", "weighted20")
MODELS = ("tower", "din")


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    return sha256_file(Path(path))


def save(path, value):
    with Path(path).open("x", encoding="utf8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def line(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def sequence(ids):
    return hashlib.sha256("".join(n + "\n" for n in ids).encode()).hexdigest()


def difference(a, b):
    return {k: b[k] - a[k] for k in a}


def audit_sources(cfg):
    snap = snapshot()
    for path, identity in cfg["configs"].items():
        require(digest(ROOT/path) == identity, f"configuration changed: {path}")
    din_cfg = read(ROOT/cfg["din_config"])
    base, encoder, ref, folder = sources(din_cfg)
    require(base["data_version"] == snap["DATA"]["version"] and base["lock"] == snap["LOCK"]["version"],
            "DATA or LOCK mismatch")
    refs = {}
    for obj, identity in cfg["references"].items():
        require(snap[obj]["identity"] == identity, f"{obj}: registry changed")
        refs[obj] = registered_ref(obj, snap)
    for seed in cfg["seeds"]:
        tower = refs[f"RV-two-tower-s{seed}"]
        din = refs[f"RV-din-s{seed}"]
        for path, identity in (tower["artifacts"] | din["weights"]).items():
            require(digest(ROOT/path) == identity, f"changed: {path}")
        p = din["pool"]
        require(din["reference"] == cfg["references"]["RV-faiss-s0"]
                and digest(ROOT/cfg["pool_manifest"]) == din["pool_manifest_sha256"]
                and p == read(ROOT/cfg["pool_manifest"]) and p["inputs"] == ref["inputs_sha256"]
                and p["news_ids"] == ref["inputs"]["news_id_sha256"]
                and p["impression_ids"] == ref["inputs"]["impression_id_sha256"]
                and digest(ROOT/p["path"]) == p["sha256"], "fixed pool identity changed")
    return base, din_cfg, refs, folder


def split_ids(original, chronological, count, seed):
    perm = np.random.default_rng(seed).permutation(len(original))
    train = [original[i] for i in perm[:count]]
    valid = [original[i] for i in perm[count:]]
    random_test = set(valid)
    common = [n for n in chronological[count:] if n in random_test]
    require(len(set(original)) == len(original) and set(original) == set(chronological)
            and len(original) == len(chronological), "duplicate or missing exposure ID")
    require(not (set(train) & random_test) and not (set(common) & set(chronological[:count])),
            "training exposure in holdout")
    return dict(time_train=chronological[:count], time_valid=chronological[count:],
                random_train=train, random_valid=valid, common=common)


@dataclass
class Dataset:
    rows: list
    texts: dict
    users: dict
    by_id: dict
    splits: dict
    corpus: list
    pool: np.ndarray


def dataset(cfg, base, refs, folder):
    rows, texts = read_sources(base)
    with (ROOT/base["behaviors"]).open("rb") as f:
        users = {r[0]: r[1] for _, r in fields(f, "behaviors.tsv", 5)}
    time_train, _ = split_prefix(rows, refs["RV-data-profile"]["split"], base)
    ordered = [r.impression_id for r in rows]
    splits = split_ids(list(users), ordered, len(time_train), cfg["split_seed"])
    require([len(splits[k]) for k in ("time_train","time_valid","random_train","random_valid","common")]
            == [cfg["train"],cfg["valid"],cfg["train"],cfg["valid"],cfg["common"]], "split denominator mismatch")
    valid = rows[len(time_train):]
    require(all(r.impression_id == q["id"] and list(r.clicks()) == q["clicks"]
                for r,q in zip(valid, read(folder/"queries.json"), strict=True)), "CARD-008 query order differs")
    corpus = read(folder/"news-ids.json")
    pool = np.load(ROOT/refs["RV-din-s0"]["pool"]["path"], allow_pickle=False)
    require(corpus == sorted(set(corpus)) and pool.shape == (len(valid),100), "invalid pool shape/order")
    return Dataset(rows,texts,users,dict(zip(ordered,rows)),splits,corpus,pool)


def rng_stream(seed, stream):
    return np.random.default_rng(np.random.SeedSequence([seed, stream]))


def pair_donors(rows, users, valid, limit, rng):
    """Visit source-hour buckets in order. Only previous seconds may donate."""
    matches = {}
    for _, hour in groupby(rows, lambda r: r.timestamp.replace(minute=0,second=0,microsecond=0)):
        bins = defaultdict(list)
        for _, same_time in groupby(hour, lambda r: r.timestamp):
            group = list(same_time)
            for row in group:
                iid = row.impression_id
                length = min(len(row.history),limit)
                if not length or iid not in valid:
                    continue
                options = [d for d in bins[length] if users[d.impression_id] != users[iid]]
                donor = options[int(rng.integers(len(options)))] if options else None
                matches[iid] = donor.impression_id if donor else None
                if donor:
                    require(donor.timestamp < row.timestamp and donor.timestamp.date() == row.timestamp.date()
                            and donor.timestamp.hour == row.timestamp.hour and users[donor.impression_id] != users[iid]
                            and min(len(donor.history), limit) == length, f"ineligible donor {iid}")
            for r in group:
                if r.history:
                    bins[min(len(r.history),limit)].append(r)
    return matches


def negatives(row, visible):
    pos = tuple(row.clicks())
    require(len(row.candidates) == len(set(row.candidates)), f"duplicate source candidate {row.impression_id}")
    if not pos or not set(pos) <= visible:
        return None
    neg = [n for n,l in zip(row.candidates,row.labels) if not l and n in visible]
    return (pos,neg) if len(neg) >= 20 else None


def sample_lists(row, pos, neg, counts, streams):
    result = {}
    weights = np.array([counts[n]+1 for n in neg], dtype=np.float64)
    for key, rng in zip(GROUPS, streams, strict=True):
        size = 5 if key.endswith("5") else 20
        probs = weights/weights.sum() if key.startswith("weighted") else None
        ix = rng.choice(len(neg), size=size, replace=False, p=probs)
        keep = set(pos).union(neg[i] for i in ix)
        ids = [n for n in row.candidates if n in keep]
        require(len(ids) == size+len(pos) and set(row.clicks()) <= set(ids), f"invalid sample {row.impression_id}/{key}")
        result[key] = ids
    return result


def sample_replay(rows, valid, streams):
    window = ClickWindow(24)
    visible = set()
    for timestamp, batch in groupby(rows, lambda r:r.timestamp):
        group = list(batch)
        window.expire(timestamp)
        for r in group:
            if r.impression_id in valid:
                eligible = negatives(r, visible)
                if eligible:
                    pos, neg = eligible
                    yield r, sample_lists(r,pos,neg,window.counts,streams)
        for r in group:
            visible.update(r.candidates)
        window.update(group)


def plans(data, cfg, seed):
    valid = set(data.splits["time_valid"])
    matches = pair_donors(data.rows,data.users,valid,cfg["history_limit"],rng_stream(seed,cfg["streams"]["donor"]))
    streams = [rng_stream(seed,cfg["streams"][group]) for group in GROUPS]
    samples = {r.impression_id:g for r,g in sample_replay(data.rows,valid,streams)}
    require(len(samples) == cfg["sample_count"] and sequence(samples) == cfg["sample_query_sha256"],
            f"candidate denominator or query identity mismatch: {len(samples)}")
    return matches,samples


def donor_entry(iid, did, data, limit):
    def identity(n):
        r=data.by_id[n]
        h=r.history[-limit:]
        return dict(id=n,user=data.users[n],time=r.timestamp.isoformat(),length=len(h),history_sha256=sequence(h))
    return dict(recipient=identity(iid),donor=identity(did) if did else None)


def freeze(cfg, data, out):
    out.mkdir(parents=True,exist_ok=False)
    save(out/"splits.json", data.splits)
    paths = [out/"splits.json"]
    for seed in cfg["seeds"]:
        matches, samples = plans(data,cfg,seed)
        sp, dp = out/f"samples-{seed}.json", out/f"donors-{seed}.jsonl"
        save(sp,samples)
        with dp.open("x",encoding="utf8",newline="\n") as f:
            for iid,did in matches.items():
                line(f,donor_entry(iid,did,data,cfg["history_limit"]))
        paths.extend([sp,dp])
        print("freeze",seed,len(samples),len(matches),sum(d is not None for d in matches.values()),flush=True)
    save(out/"freeze.json",{p.relative_to(ROOT).as_posix():digest(p) for p in paths})


def load_plans(cfg,data,out,seed,expected):
    require(digest(out/"freeze.json") == expected, "freeze manifest hash mismatch")
    manifest = read(out/"freeze.json")
    for path,h in manifest.items():
        require(digest(ROOT/path) == h, f"freeze artifact changed: {path}")
    require(read(out/"splits.json") == data.splits, "split replay differs")
    samples = read(out/f"samples-{seed}.json")
    matches = {}
    with (out/f"donors-{seed}.jsonl").open(encoding="utf8") as f:
        for raw in f:
            record=json.loads(raw)
            iid=record["recipient"]["id"]
            did=record["donor"]["id"] if record["donor"] else None
            require(record == donor_entry(iid,did,data,cfg["history_limit"]),f"donor identity changed: {iid}")
            matches[iid]=did
    rebuilt, lists=plans(data,cfg,seed)
    require(matches == rebuilt and samples == lists,"donor or candidate replay mismatch")
    return matches,samples

def news_seen(rows, history=False):
    news = {n for r in rows for n in r.candidates}
    if history:
        news.update(n for r in rows for n in r.history)
    return news


def catalog(data, train, base):
    return build_catalog(data.texts,news_seen(train,True),news_seen(data.rows,True),base)


def rows_for(data, name):
    return [data.by_id[n] for n in data.splits[name]]


def overlap(train, valid, common, users):
    def stats(a,b):
        return dict(train=len(a),validation=len(b),shared=len(a & b))
    ut,uv=({users[r.impression_id] for r in side} for side in (train,valid))
    nt,nv=(news_seen(side) for side in (train,valid))
    cold=[r for r in common if users[r.impression_id] not in ut]
    unseen=[n for r in common for n in r.candidates if n not in nt]
    return dict(users=stats(ut,uv),candidate_news=stats(nt,nv),common_exposures=len(common),
                common_users=len({users[r.impression_id] for r in common}),cold_user_exposures=len(cold),
                distinct_cold_users=len({users[r.impression_id] for r in cold}),
                candidate_occurrences=sum(len(r.candidates) for r in common),unseen_occurrences=len(unseen),
                distinct_common_candidates=len(news_seen(common)), distinct_unseen_news=len(set(unseen)))


def official(labels, ranks):
    report=evaluate([Impression(labels,ranks)])
    return report["metrics"]


def mean_dict(rows):
    require(bool(rows),"empty paired denominator")
    return {k:math.fsum(r[k] for r in rows)/len(rows) for k in rows[0]}


@torch.inference_mode()
def score_arm(model,table,valid,index,common,corpus,base):
    model.eval()
    size=base["evaluation_batch"]
    bank=torch.cat([model.news(c) for c in table.split(size)])
    h=histories(valid,index,base["history_limit"])
    ids=[index[n] for n in corpus]
    pool=set(corpus)
    visible=np.ones(len(ids),dtype=bool)
    paired={}
    items=[]
    retrieved=Retrieval()
    for start in range(0,len(valid),size):
        batch=valid[start:start+size]
        hist=torch.as_tensor(h[start:start+size],device=table.device)
        dots=(model.aggregate_history(bank[hist],hist) @ bank.T).cpu().numpy()
        for row,scores in zip(batch,dots,strict=True):
            ranks=ranks_from_scores(row.candidates,scores[[index[n] for n in row.candidates]])
            items.append(Impression(row.labels,ranks))
            if row.impression_id in common:
                top={corpus[i] for i in exact_top(scores[ids],visible,100)}
                clicked=list(row.clicks())
                hits=sum(n in top for n in clicked)
                covered=sum(n in pool for n in clicked)
                retrieved.add(clicked,pool,top)
                paired[row.impression_id]=dict(ndcg10=official(row.labels,ranks)["nDCG@10"],
                                               clicks=len(clicked),covered=covered,hits=hits)
    report=evaluate(items)
    require(report["scoring_denominator"]==len(valid) and len(paired)==len(common),"arm denominator mismatch")
    return dict(full=report,common_ndcg10=math.fsum(r["ndcg10"] for r in paired.values())/len(paired),
                common_denominator=len(paired),recall100=retrieved.report()),paired


def split_audit(data, base, cfg, refs, seed, folder, device):
    common=rows_for(data,"common")
    ids=set(data.splits["common"])
    result={}
    evidence={}
    for arm in ("time","random"):
        train=rows_for(data,arm+"_train")
        valid=rows_for(data,arm+"_valid")
        names,vocab,table=catalog(data,train,base)
        index={n:i for i,n in enumerate(names,1)}
        torch.manual_seed(seed)
        model=MeanTowers(len(vocab),base["embedding_dim"]).to(device)
        table=table.to(device)
        if arm=="time":
            prior=ROOT/f"checkpoints/card-004/s{seed}"
            require(vocab==read(prior/"vocabulary.json"),"frozen vocabulary differs")
            model.load_state_dict(torch.load(prior/"model.pt",map_location=device,weights_only=True))
            require(weight_identity(model)==refs[f"RV-two-tower-s{seed}"]["trained_weight_sha256"],"tower weights differ")
            artifacts=refs[f"RV-two-tower-s{seed}"]["artifacts"]
            training=None
        else:
            vfile,weights=folder/"vocabulary.json",folder/"model.pt"
            save(vfile,vocab)
            training,_=train_epochs(model,table,train,index,base,device,seed)
            torch.save(model.state_dict(),weights)
            artifacts={p.relative_to(ROOT).as_posix():digest(p) for p in (vfile,weights)}
        metrics,evidence[arm]=score_arm(model,table,valid,index,ids,data.corpus,base)
        if arm=="time":
            recorded=refs[f"RV-two-tower-s{seed}"]["metrics"]["full"]["official"]
            require(abs(recorded["metrics"]["nDCG@10"]-metrics["full"]["metrics"]["nDCG@10"])<=1e-8,
                    "CARD-004 time-arm replay differs")
        result[arm]=dict(**metrics,overlap=overlap(train,valid,common,data.users),artifacts=artifacts,
                         training=training,validation_ids=sequence(r.impression_id for r in valid))
    rows=[]
    for iid in data.splits["common"]:
        a,b=(evidence[arm][iid] for arm in ("time","random"))
        require((a["clicks"],a["covered"])==(b["clicks"],b["covered"]),f"common pool mismatch: {iid}")
        rows.append(dict(id=iid,time=a,random=b,ndcg_delta=b["ndcg10"]-a["ndcg10"],hit_delta=b["hits"]-a["hits"]))
    save(folder/"split-pairs.json",rows)
    result["random_minus_time"]=dict(ndcg10=math.fsum(r["ndcg_delta"] for r in rows)/len(rows),
        recall100=sum(r["hit_delta"] for r in rows)/sum(r["time"]["clicks"] for r in rows))
    result["common_query_sha256"]=sequence(data.splits["common"])
    result["pool_news_sha256"]=payload_sha256(data.corpus)
    result["train_common_intersections"]={arm:len(ids.intersection(data.splits[arm+"_train"])) for arm in ("time","random")}
    return result


@lru_cache(maxsize=None)
def expected_random(n,p):
    require(0<p<n,"random expectation requires mixed labels")
    def discount(k):
        return math.fsum(1/math.log2(i+1) for i in range(1,k+1))
    return {"AUC":0.5,"MRR":math.fsum(1/i for i in range(1,n+1))/n,
            **{f"nDCG@{k}":(p/n)*discount(min(k,n))/discount(min(k,p)) for k in (5,10)}}


def permuted(row,rng):
    labels=bytes(rng.permutation(np.frombuffer(row.labels,dtype=np.uint8)))
    require(len(labels)==len(row.labels) and sum(labels)==sum(row.labels),f"label counts changed: {row.impression_id}")
    return replace(row,labels=labels)


class Scorer:
    """One production scoring entry for observed and label-replaced events."""
    def __init__(self,tower,model,bank,index,limit):
        self.tower=tower
        self.model=model
        self.bank=bank
        self.index=index
        self.limit=limit

    @torch.inference_mode()
    def score(self,rows,*,fixed=False,probe=False,frozen_order=False):
        hist=histories(rows,self.index,self.limit)
        lengths=np.array([len(r.candidates) for r in rows])
        flat=np.array([self.index[n] for r in rows for n in r.candidates],dtype=np.int64)
        device=self.bank.device
        # Compute the frozen mean on CPU, as in CARD-008/009; dot products use the scoring device.
        users=self.tower.aggregate_history(self.cpu_bank[torch.from_numpy(hist)],torch.from_numpy(hist)).to(device)
        repeat=torch.as_tensor(np.repeat(np.arange(len(rows)), lengths),device=device)
        c=self.bank[torch.as_tensor(flat,device=device)]
        tower=(c*users[repeat]).sum(-1).cpu().numpy()
        din=batch_logits(self.model,self.bank,hist,lengths,flat).cpu().numpy()
        branches={"tower":tower,"din":din}
        if probe:
            zeros=torch.zeros_like(c)
            branches["candidate_only"]=self.model.rank(torch.cat((c,zeros,zeros),dim=-1)).squeeze(-1).cpu().numpy()
        offsets=np.cumsum(lengths)[:-1]
        sliced={k:np.split(v,offsets) for k,v in branches.items()}
        results=[]
        for j,r in enumerate(rows):
            scores={k:v[j] for k,v in sliced.items()}
            if fixed and frozen_order:
                scores["tower"]=-np.arange(len(r.candidates),dtype=np.float32)
            if not r.history:
                scores["din"]=(-np.arange(len(r.candidates),dtype=np.float32)
                               if fixed else scores["tower"])
            ranks={k:ranks_from_scores(r.candidates,v) for k,v in scores.items()}
            results.append((r.candidates,scores,ranks))
        return results


def load_scorer(data,base,din_cfg,seed,folder,device):
    names,vocab,tokens=catalog(data,rows_for(data,"time_train"),base)
    index={n:i for i,n in enumerate(names,1)}
    tower=MeanTowers(len(vocab),base["embedding_dim"]).eval()
    tower.load_state_dict(torch.load(ROOT/"checkpoints/card-004/s0/model.pt",map_location="cpu",weights_only=True))
    with torch.inference_mode():
        bank=torch.cat([tower.news(t) for t in tokens.split(256)])
        require(np.array_equal(bank[[index[n] for n in data.corpus]].numpy(),np.load(folder/"news.npy")),
                "CARD-008 corpus encodings differ")
        hist=torch.from_numpy(histories(rows_for(data,"time_valid"),index,din_cfg["history"]))
        q=torch.cat([tower.aggregate_history(bank[h],h) for h in hist.split(256)])
        require(np.array_equal(q.numpy(),np.load(folder/"queries.npy")),"CARD-008 queries differ")
    model=DIN(din_cfg["dim"],din_cfg["attention_widths"],din_cfg["head_width"]).to(device).eval()
    model.load_state_dict(torch.load(ROOT/f"checkpoints/card-009/s{seed}.pt",map_location=device,weights_only=True))
    scorer=Scorer(tower,model,bank.to(device),index,din_cfg["history"])
    scorer.cpu_bank=bank
    return scorer


def invariant(iid,before,after):
    ids,sc,ranks=before
    ids2,sc2,ranks2=after
    require(ids==ids2 and ranks==ranks2 and all(np.array_equal(sc[k],sc2[k]) for k in sc),
            f"label replacement changed prediction: {iid}")


def sorted_names(names,ranks):
    return [n for _,n in sorted(zip(ranks,names))]


def sense(delta):
    return "DIN" if delta>0 else ("tower" if delta<0 else "tie")

def fixed_row(row, ids):
    clicked=set(row.clicks())
    return replace(row,candidates=tuple(ids),labels=bytes(n in clicked for n in ids))


def baseline_scores(data,matched):
    """Index fitted to training exposure titles; replay every click, including unmatched queries."""
    train=rows_for(data,"time_train")
    params=read(ROOT/"configs/mind-baselines-smoke-v0.1.json")
    titles=TitleIndex({n:txt[0] for n,txt in data.texts.items()},news_seen(train),params["tfidf"])
    transpose=titles.matrix.T.tocsr()
    @lru_cache(maxsize=32)
    def similarity(last):
        if last not in titles.index:
            return None
        row=titles.matrix[titles.index[last]]
        if row.nnz==0:
            return None
        # Same normalized title inner product as TitleIndex.query, using CSR posting lists.
        return np.append((row @ transpose).toarray().ravel(),0.)
    result={}
    window=ClickWindow(params["window_hours"])
    window.update(train)
    valid=rows_for(data,"time_valid")
    pos=0
    for stamp,group in groupby(valid,lambda r:r.timestamp):
        rows=list(group)
        window.expire(stamp)
        for row in rows:
            if row.impression_id in matched:
                fixed=[data.corpus[i] for i in data.pool[pos]]
                sims=similarity(row.history[-1])
                values={}
                for key,names in (("official",row.candidates),("fixed",fixed)):
                    pop=[window.counts[n] for n in names]
                    recent=pop if sims is None else titles.scores(sims,names)
                    values[key]={k:ranks_from_scores(names,s) for k,s in (("popularity24h",pop),("recent_click",recent))}
                result[row.impression_id]=values
            pos+=1
        window.update(rows)
    require(set(result)==matched,"baseline sample mismatch")
    return result


class HistoryReport:
    def __init__(self):
        self.official=defaultdict(list)
        self.pool=defaultdict(list)
        self.end=defaultdict(list)

    def add(self,row,fixed,logged_ranks,pool_ranks):
        require(logged_ranks.keys()==pool_ranks.keys(),"mismatched history branches")
        record={}
        for k in logged_ranks:
            m=official(row.labels,logged_ranks[k])
            within,end=pool_scores(sorted_names(fixed,pool_ranks[k]),list(row.clicks()))
            require(m is not None,"invalid paired official query")
            self.official[k].append(m)
            self.end[k].append(end)
            if within is not None:
                self.pool[k].append(within)
            record[k]=[m["nDCG@10"],within,end]
        return record

    def report(self):
        result={}
        for k,items in self.official.items():
            result[k]=dict(official=mean_dict(items),official_n=len(items),
                           fixed_conditional=math.fsum(self.pool[k])/len(self.pool[k]) if self.pool[k] else None,
                           fixed_n=len(self.pool[k]),end_to_end=math.fsum(self.end[k])/len(items),end_n=len(items))
        require(len({(v["official_n"],v["fixed_n"],v["end_n"]) for v in result.values()})==1,
                "history intervention denominators differ")
        return result


def history_batch(scorer,rows,fixed,donors,baseline,acc,f):
    if not rows:
        return
    branches=[]
    for p,is_fixed in ((rows,False),(fixed,True)):
        original=scorer.score(p,fixed=is_fixed,frozen_order=is_fixed,probe=True)
        changed=scorer.score([replace(r,history=d.history) for r,d in zip(p,donors,strict=True)],fixed=is_fixed)
        empty=scorer.score([replace(r,history=()) for r in p],fixed=is_fixed)
        branch=[]
        for i,r in enumerate(p):
            require(original[i][0]==changed[i][0]==empty[i][0]==r.candidates,
                    f"history candidate mismatch: {r.impression_id}")
            ranks={"real_"+k:original[i][2][k] for k in MODELS}
            ranks.update({"donor_"+k:changed[i][2][k] for k in MODELS})
            ranks.update(din_empty_fallback=empty[i][2]["din"],candidate_only=original[i][2]["candidate_only"])
            ranks.update(baseline[r.impression_id]["fixed" if is_fixed else "official"])
            branch.append(ranks)
        branches.append(branch)
    for row,fix,logged,pool in zip(rows,fixed,*branches,strict=True):
        detail=acc.add(row,fix.candidates,logged,pool)
        line(f,dict(id=row.impression_id,inputs=payload_sha256([row.candidates,list(row.labels),fix.candidates,list(fix.labels)]),
                    scores=detail))


def history_audit(cfg,data,scorer,matches,folder):
    matched={iid for iid,did in matches.items() if did}
    baseline=baseline_scores(data,matched)
    valid=rows_for(data,"time_valid")
    pairs=[(i,r) for i,r in enumerate(valid) if r.impression_id in matched]
    acc=HistoryReport()
    with (folder/"history.jsonl").open("x",encoding="utf8",newline="\n") as f:
        for start in range(0,len(pairs),32):
            batch=pairs[start:start+32]
            rows=[r for _,r in batch]
            fixed=[fixed_row(r,[data.corpus[j] for j in data.pool[i]]) for i,r in batch]
            donors=[data.by_id[matches[r.impression_id]] for r in rows]
            history_batch(scorer,rows,fixed,donors,baseline,acc,f)
    stats=acc.report()
    deltas={}
    for k in MODELS:
        orig,alt=stats["real_"+k],stats["donor_"+k]
        deltas[k]=dict(official=difference(orig["official"],alt["official"]),
                       fixed_conditional=alt["fixed_conditional"]-orig["fixed_conditional"],
                       end_to_end=alt["end_to_end"]-orig["end_to_end"])
    return dict(total=len(valid),empty_history=len(valid)-len(matches),with_history=len(matches),
                no_donor=len(matches)-len(matched),matched=len(matched),
                paired_query_sha256=sequence(r.impression_id for _,r in pairs),branches=stats,
                donor_minus_real=deltas)


def random_record(row,shuffled,pred,random_ranks):
    names,scores,ranks=pred
    n,p=len(names),sum(row.labels)
    actual={m:official(row.labels,ranks[m]) for m in MODELS}
    measures={m:official(shuffled.labels,ranks[m]) for m in MODELS}
    measures["independent_random"]=official(row.labels,random_ranks)
    require(all(v is not None for v in measures.values()),f"random control invalid denominator: {row.impression_id}")
    evidence=dict(id=row.impression_id,candidate_sha256=sequence(names),positive=p,n=n,
                  scores_sha256={k:hashlib.sha256(np.asarray(v,dtype="<f4").tobytes()).hexdigest() for k,v in scores.items()},
                  permuted_labels=list(shuffled.labels),ranks=ranks,random_ranks=random_ranks)
    return actual,measures,expected_random(n,p),evidence


def sampling_row(row,scored,groups):
    names,scores,full_ranks=scored
    lookup={n:i for i,n in enumerate(names)}
    result={}
    for key,ids in {"original":names,**groups}.items():
        ix=[lookup[n] for n in ids]
        labels=bytes(row.labels[i] for i in ix)
        require(sum(labels)==sum(row.labels) and len(ix)==len(set(ix)),f"sample labels or duplicates: {row.impression_id}/{key}")
        values={}
        for m in MODELS:
            ranks=full_ranks[m] if key=="original" else ranks_from_scores(ids,scores[m][ix])
            values[m]=official(labels,ranks)["nDCG@10"]
        result[key]=dict(candidates=len(ids),tower=values["tower"],din=values["din"],delta=values["din"]-values["tower"])
    return result


def random_sampling_audit(data,cfg,scorer,samples,seed,folder):
    label_rng=rng_stream(seed,cfg["streams"]["labels"])
    rank_rng=rng_stream(seed,cfg["streams"]["random_ranks"])
    valid=rows_for(data,"time_valid")
    actual=defaultdict(list)
    controls=defaultdict(list)
    expectations=[]
    biggest={}
    draws=defaultdict(list)
    with (folder/"random.jsonl").open("x",encoding="utf8",newline="\n") as f, (folder/"sampling.jsonl").open("x",encoding="utf8",newline="\n") as g:
        for offset in range(0,len(valid),32):
            rows=valid[offset:offset+32]
            shuffled=[permuted(r,label_rng) for r in rows]
            before=scorer.score(rows)
            after=scorer.score(shuffled)
            for row,alt,a,b in zip(rows,shuffled,before,after,strict=True):
                iid=row.impression_id
                invariant(iid,a,b)
                ranks=(rank_rng.permutation(len(row.candidates))+1).tolist()
                orig,measure,expected,evidence=random_record(row,alt,a,ranks)
                for m,v in orig.items():
                    actual[m].append(v)
                for m,v in measure.items():
                    controls[m].append(v)
                    for metric,d in difference(expected,v).items():
                        key=m+"/"+metric
                        if key not in biggest or abs(d)>abs(biggest[key]["delta"]):
                            biggest[key]=dict(id=iid,delta=d,n=len(row.candidates),positive=sum(row.labels))
                expectations.append(expected)
                line(f,evidence)
                if iid in samples:
                    res=sampling_row(row,a,samples[iid])
                    for k,v in res.items():
                        draws[k].append(v)
                    line(g,dict(id=iid,groups=res))
    expected=mean_dict(expectations)
    ctrl=dict(denominator=len(valid),expectation=expected,original={m:mean_dict(v) for m,v in actual.items()},
              prediction_invariance={k:len(valid) for k in MODELS},max_individual_deviations=biggest,controls={})
    for m,values in controls.items():
        avg=mean_dict(values)
        ctrl["controls"][m]=dict(denominator=len(values),metrics=avg,observed_minus_expected=difference(expected,avg))
    result={}
    for k,values in draws.items():
        require(len(values)==cfg["sample_count"],f"sampling denominator differs: {k}")
        avg=mean_dict(values)
        result[k]=dict(denominator=len(values),candidate_total=sum(v["candidates"] for v in values),
                       min_candidates=min(v["candidates"] for v in values),max_candidates=max(v["candidates"] for v in values),
                       mean_candidates=avg["candidates"],tower=avg["tower"],din=avg["din"],
                       din_minus_tower=avg["delta"],direction=sense(avg["delta"]))
    for v in result.values():
        v["direction_changed"]=v["direction"]!=result["original"]["direction"]
    return ctrl,dict(query_sha256=sequence(samples),groups=result)

def bounded_seed(cfg,data,base,din_cfg,refs,inputs,seed,out,manifest):
    started=perf_counter()
    def timeout():
        print(f"BLOCK seed {seed}: exceeded {cfg['seconds']} seconds",flush=True)
        os._exit(124)
    timer=Timer(cfg["seconds"],timeout)
    timer.daemon=True
    timer.start()
    try:
        folder=out/f"s{seed}"
        folder.mkdir(exist_ok=False)
        matches,samples=load_plans(cfg,data,out,seed,manifest)
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("A split",seed,device,flush=True)
        a=split_audit(data,base,cfg,refs,seed,folder,device)
        scorer=load_scorer(data,base,din_cfg,seed,inputs,device)
        print("B history",seed,flush=True)
        b=history_audit(cfg,data,scorer,matches,folder)
        print("C labels / D samples",seed,flush=True)
        c,d=random_sampling_audit(data,cfg,scorer,samples,seed,folder)
        datafiles={p.relative_to(ROOT).as_posix():digest(p) for p in sorted(folder.iterdir())}
        frozen=read(out/"freeze.json")
        rv=dict(seed=seed,references=cfg["references"],upstream_weights={
                    k:refs[k]["artifacts" if k.startswith("RV-two-tower") else "weights"]
                    for k in ("RV-two-tower-s0",f"RV-two-tower-s{seed}",f"RV-din-s{seed}")},
                pool=refs["RV-din-s0"]["pool"],freeze_sha256=manifest,frozen_files=frozen,
                evidence=datafiles,split=a,history=b,random=c,sampling=d)
        dest=Path(cfg["scratch"])/Path(cfg["outputs"][seed]).name
        require(not (ROOT/dest).exists() and not status_paths(),"seed output exists or working tree changed")
        write_result(dest.as_posix(),rv,"RV",dest.stem,CONFIG,seed,
                     dict(elapsed_seconds=perf_counter()-started,device=str(device),code_tag="card-010"))
    finally:
        timer.cancel()


def records(path):
    with path.open(encoding="utf8") as f:
        for raw in f:
            yield json.loads(raw)


def near(x,y):
    require(math.isclose(x,y,rel_tol=0,abs_tol=1e-10),f"recomputed metric differs: {x} vs {y}")


def verify_seed(cfg,data,refs,out,seed,doc,manifest):
    meta,p=doc["_meta"],doc["payload"]
    require(meta["commit"]==head() and meta["dirty"] is False and meta["code_tag"]=="card-010"
            and meta["seed"]==p["seed"]==seed and meta["config"]==CONFIG and meta["config_sha256"]==digest(ROOT/CONFIG)
            and meta["kind"]=="RV" and meta["version"]==Path(cfg["outputs"][seed]).stem
            and meta["data_version"]==snapshot()["DATA"]["version"] and meta["lock"]==snapshot()["LOCK"]["version"]
            and 0<meta["elapsed_seconds"]<cfg["seconds"] and meta["payload_sha256"]==payload_sha256(p),
            f"RV metadata or payload mismatch: s{seed}")
    require(p["references"]==cfg["references"] and p["freeze_sha256"]==manifest and p["frozen_files"]==read(out/"freeze.json")
            and p["pool"]==refs["RV-din-s0"]["pool"],"RV intervention provenance differs")
    for path,h in (p["evidence"] | p["frozen_files"] | p["split"]["random"]["artifacts"]).items():
        require(digest(ROOT/path)==h,f"evidence file changed: {path}")
    for files in p["upstream_weights"].values():
        require(all(digest(ROOT/path)==h for path,h in files.items()),"upstream weight identity differs")
    matches,samples=load_plans(cfg,data,out,seed,manifest)
    folder=out/f"s{seed}"
    a=p["split"]
    pairs=read(folder/"split-pairs.json")
    require([r["id"] for r in pairs]==data.splits["common"] and len(pairs)==cfg["common"]
            and all(n==0 for n in a["train_common_intersections"].values()),"common query mismatch")
    for arm in ("time","random"):
        require(a[arm]["full"]["scoring_denominator"]==cfg["valid"],"full denominator differs")
        vals=[r[arm] for r in pairs]
        q=mean_dict(vals)
        near(q["ndcg10"],a[arm]["common_ndcg10"])
        ret=a[arm]["recall100"]
        for k in ("clicks","covered","hits"):
            require(sum(r[k] for r in vals)==ret[k],f"common {k} count differs")
        near(ret["hits"]/ret["clicks"],ret["recall"])
    near(a["random"]["common_ndcg10"]-a["time"]["common_ndcg10"],a["random_minus_time"]["ndcg10"])
    near(a["random"]["recall100"]["recall"]-a["time"]["recall100"]["recall"],a["random_minus_time"]["recall100"])
    b=p["history"]
    paired=[i for i in data.splits["time_valid"] if matches.get(i)]
    require(len(paired)==b["matched"] and len(matches)-len(paired)==b["no_donor"] and
            b["empty_history"]+len(matches)==cfg["valid"] and sequence(paired)==b["paired_query_sha256"],
            "history denominator differs")
    pool_indices={iid:i for i,iid in enumerate(data.splits["time_valid"])}
    totals=defaultdict(lambda:np.zeros(4))
    for iid,r in zip(paired,records(folder/"history.jsonl"),strict=True):
        row=data.by_id[iid]
        fixed=fixed_row(row,[data.corpus[j] for j in data.pool[pool_indices[iid]]])
        require(r["id"]==iid and r["inputs"]==payload_sha256([row.candidates,list(row.labels),fixed.candidates,list(fixed.labels)]),
                f"history labels/candidates changed: {iid}")
        for k,(logged,conditional,end) in r["scores"].items():
            require((conditional is not None)==bool(sum(fixed.labels)),f"conditional denominator mismatch: {iid}")
            totals[k]+=[logged,conditional or 0.,end,int(conditional is not None)]
    for k,(logged,within,end,n) in totals.items():
        v=b["branches"][k]
        require(v["fixed_n"]==n and v["end_n"]==v["official_n"]==len(paired),"history branch denominator differs")
        near(logged/len(paired),v["official"]["nDCG@10"])
        near(within/n,v["fixed_conditional"])
        near(end/len(paired),v["end_to_end"])
    ctrl=p["random"]
    rand=defaultdict(list)
    expected=[]
    labels_rng=rng_stream(seed,cfg["streams"]["labels"])
    ranks_rng=rng_stream(seed,cfg["streams"]["random_ranks"])
    for iid,r in zip(data.splits["time_valid"],records(folder/"random.jsonl"),strict=True):
        row=data.by_id[iid]
        n,pcount=len(row.candidates),sum(row.labels)
        require(r["id"]==iid and sequence(row.candidates)==r["candidate_sha256"] and r["n"]==n
                and sum(r["permuted_labels"])==r["positive"]==pcount,"random evidence differs")
        require(r["permuted_labels"]==list(permuted(row,labels_rng).labels) and
                r["random_ranks"]==(ranks_rng.permutation(n)+1).tolist(),f"random stream replay differs: {iid}")
        expected.append(expected_random(n,pcount))
        for m in MODELS:
            rand[m].append(official(r["permuted_labels"],r["ranks"][m]))
        rand["independent_random"].append(official(row.labels,r["random_ranks"]))
    for k,v in mean_dict(expected).items():
        near(v,ctrl["expectation"][k])
    require(ctrl["denominator"]==len(expected)==cfg["valid"],"random denominator differs")
    for m,rows in rand.items():
        measured=ctrl["controls"][m]
        require(measured["denominator"]==len(rows),"control denominator differs")
        for k,v in mean_dict(rows).items():
            near(v,measured["metrics"][k])
            near(v-ctrl["expectation"][k],measured["observed_minus_expected"][k])
    ds=defaultdict(list)
    for iid,r in zip(samples,records(folder/"sampling.jsonl"),strict=True):
        require(r["id"]==iid and set(r["groups"])=={"original",*GROUPS},"sample group identity differs")
        row=data.by_id[iid]
        for k,v in r["groups"].items():
            ids=row.candidates if k=="original" else samples[iid][k]
            require(v["candidates"]==len(ids) and set(row.clicks())<=set(ids),"sample candidate count differs")
            near(v["din"]-v["tower"],v["delta"])
            ds[k].append(v)
    g=p["sampling"]["groups"]
    for k,rows in ds.items():
        v=g[k]
        x=mean_dict(rows)
        require(len(rows)==v["denominator"]==cfg["sample_count"] and
                v["candidate_total"]==sum(r["candidates"] for r in rows),"sample denominator differs")
        for name in MODELS:
            near(x[name],v[name])
        near(x["delta"],v["din_minus_tower"])
        require(sense(x["delta"])==v["direction"] and v["direction_changed"]==(v["direction"]!=g["original"]["direction"]),
                "sample direction differs")
    print("verified",seed,meta["payload_sha256"],"common",len(pairs),"paired",len(paired),"sampled",len(samples),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sources",action="store_true")
    mode.add_argument("--freeze",action="store_true")
    mode.add_argument("--seed",type=int,choices=(0,1,2))
    mode.add_argument("--publish",action="store_true")
    parser.add_argument("--freeze-sha256")
    args=parser.parse_args()
    cfg=read(ROOT/CONFIG)
    base,din_cfg,refs,inputs=audit_sources(cfg)
    if args.sources:
        print("verified upstream RV and artifact references",len(refs))
        return
    require(not status_paths() and head()==git("rev-parse","card-010",check=True).stdout.strip(),"requires clean card-010")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    data=dataset(cfg,base,refs,inputs)
    out=ROOT/cfg["folder"]
    if args.freeze:
        freeze(cfg,data,out)
        print("freeze SHA256",digest(out/"freeze.json"))
    elif args.seed is not None:
        bounded_seed(cfg,data,base,din_cfg,refs,inputs,args.seed,out,args.freeze_sha256)
    else:
        manifest=read(ROOT/cfg["scratch"]/"manifest.json")
        require(manifest["outputs"]==cfg["outputs"],"output manifest differs")
        copies=[]
        for seed,dst in enumerate(cfg["outputs"]):
            src=ROOT/cfg["scratch"]/Path(dst).name
            raw=src.read_bytes()
            verify_seed(cfg,data,refs,out,seed,json.loads(raw),args.freeze_sha256)
            require(not (ROOT/dst).exists(),f"published destination exists: {dst}")
            copies.append((ROOT/dst,raw))
        require(not status_paths(),"worktree changed during verification")
        for dst,raw in copies:
            with dst.open("xb") as f:
                f.write(raw)
            require(dst.read_bytes()==raw,f"published bytes differ: {dst}")
            print("published",dst.name,digest(dst))


if __name__=="__main__":
    main()
