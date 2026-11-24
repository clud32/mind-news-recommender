"""Reconstruct the registered Flat pool, train one DIN seed, and evaluate three protocols."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
from threading import Timer
from time import perf_counter

import faiss
import numpy as np
import torch

from mind_din import DIN, fixed_order, impression_loss, pool_scores
from mind_evaluation import Impression, evaluate, ranks_from_scores
from mind_two_tower import MeanTowers, build_catalog
from profile_mind_small import require
from run_mind_faiss import encoder_source, exact, read, save
from run_mind_baselines import split_prefix
from run_mind_two_tower import histories, read_sources, registered_ref, weight_identity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.common import git, head, payload_sha256, sha256_file, snapshot, status_paths
from tools.rv import write_result

CONFIG = "configs/mind-din-v0.1.json"
FLAT = "checkpoints/card-008/s0/Flat.faiss"
QUALITY = "checkpoints/card-008/s0/query-quality.json"


def sources(cfg):
    snap = snapshot()
    require(sha256_file(ROOT/cfg["faiss_config"]) == cfg["faiss_config_sha256"], "FAISS config changed")
    fc = read(ROOT/cfg["faiss_config"])
    base, encoder = encoder_source(fc, snap)
    ref = registered_ref(cfg["faiss_object"], snap)
    require(snap[cfg["faiss_object"]]["identity"] == cfg["faiss_payload"] and
            ref["index_seed"] == ref["encoder_seed"] == 0, "Flat control changed")
    inp = ROOT/ref["inputs_path"]
    require(sha256_file(inp) == ref["inputs_sha256"] and ref["inputs"] == read(inp), "inputs changed")
    for name, digest in (ref["inputs"]["artifacts"] | {key:ref["artifacts"][key] for key in (FLAT, QUALITY)}).items():
        require(sha256_file(ROOT/name) == digest, f"frozen file changed: {name}")
    return base, encoder, ref, inp.parent


def rebuild(cfg):
    _, _, ref, folder = sources(cfg)
    index = faiss.read_index(str(ROOT/FLAT))
    ids = read(folder/"news-ids.json")
    queries = np.load(folder/"queries.npy", allow_pickle=False)
    meta = read(folder/"queries.json")
    quality = read(ROOT/QUALITY)
    require(index.ntotal == len(ids) and index.metric_type == faiss.METRIC_INNER_PRODUCT, "Flat mapping mismatch")
    all_top = np.empty((len(meta),100),dtype=np.int32)
    totals = dict(impressions=len(meta), outside=0, missed=0, hits=0, pool_exposures=0)
    for i,(q,row,record) in enumerate(zip(queries,meta,quality,strict=True)):
        require(row["id"] == record["id"], "impression order differs")
        top = exact(index,q[None],100)[0] if q.any() else np.arange(100)
        require(len(set(top))==100 and min(top)>=0 and max(top)<len(ids), "invalid Flat IDs")
        all_top[i] = top
        names = {ids[j] for j in top}
        hits = sum(n in names for n in row["clicks"])
        require(record["Flat"]["hits"] == hits and len(row["clicks"]) == record["clicks"],
                f"Flat hits differ for {row['id']}")
        totals["hits"] += hits
        totals["pool_exposures"] += int(hits>0)
        totals["outside"] += record["clicks"]-record["covered"]
        totals["missed"] += record["covered"]-hits
    actual = dict(impressions=totals["impressions"], outside=totals["outside"], missed=totals["missed"],
                  hits=totals["hits"], pool_exposures=totals["pool_exposures"])
    expected = cfg["expected"]
    require(actual == dict(impressions=expected["impressions"], outside=expected["outside"],
                           missed=expected["missed"], hits=expected["recalled"], pool_exposures=expected["pool_denominator"]),
            f"fixed pool counts {actual}")
    out = ROOT/"checkpoints/card-009"
    out.mkdir(exist_ok=False)
    np.save(out/"top100.npy",all_top,allow_pickle=False)
    manifest = dict(commit=head(), flat=ref["artifacts"][FLAT], inputs=ref["inputs_sha256"],
                    impression_ids=ref["inputs"]["impression_id_sha256"],news_ids=ref["inputs"]["news_id_sha256"],
                    path="checkpoints/card-009/top100.npy",sha256=sha256_file(out/"top100.npy"),counts=totals)
    save(out/"pool.json",manifest)
    print("pool",sha256_file(out/"pool.json"),totals,flush=True)


def batches_data(rows, index, limit):
    lengths = np.array([len(row.candidates) for row in rows],dtype=np.int64)
    offsets = np.concatenate(([0],np.cumsum(lengths)))
    ids = np.fromiter((index[n] for row in rows for n in row.candidates),np.int64,count=int(offsets[-1]))
    require(all(len(r.candidates)==len(r.labels)>0 for r in rows),"missing training or validation labels")
    labels = np.frombuffer(b"".join(r.labels for r in rows),dtype=np.uint8)
    return histories(rows,index,limit), lengths,offsets,ids,labels


def unpack(data, selected):
    history,lengths,offsets,ids,labels = data
    lens = lengths[selected]
    starts = offsets[selected] - (np.cumsum(lens)-lens)
    flat = np.arange(lens.sum()) + np.repeat(starts,lens)
    return history[selected], lens, ids[flat], labels[flat]


def batch_logits(model, bank, hist, lengths, ids):
    device = bank.device
    repeat = np.repeat(np.arange(len(lengths)),lengths)
    h = torch.as_tensor(hist,device=device)
    h = h[torch.as_tensor(repeat,device=device)]
    c = bank[torch.as_tensor(ids,device=device)]
    return model(c,bank[h],h.ne(0))


def train(model, bank, data, cfg, seed):
    history,lengths,*_ = data
    eligible = np.flatnonzero(history.any(axis=1))
    skipped = len(history)-len(eligible)
    opt = torch.optim.Adam(model.parameters(),lr=cfg["lr"])
    rng = np.random.default_rng(seed)
    records=[]
    for epoch in range(cfg["epochs"]):
        model.train()
        start=perf_counter()
        order=rng.permutation(eligible)
        total_loss=0.
        candidates=0
        for offset in range(0,len(order),cfg["batch_size"]):
            batch=order[offset:offset+cfg["batch_size"]]
            h,l,ids,labels=unpack(data,batch)
            opt.zero_grad(set_to_none=True)
            logits=batch_logits(model,bank,h,l,ids)
            loss=impression_loss(logits,torch.as_tensor(labels,device=bank.device,dtype=torch.float32),
                                 torch.as_tensor(l,device=bank.device))
            value=loss.item()
            require(math.isfinite(value),"nonfinite training loss")
            loss.backward()
            opt.step()
            total_loss+=value*len(batch)
            candidates+=len(ids)
        require(candidates==lengths[eligible].sum(),"training candidate accounting mismatch")
        record=dict(epoch=epoch+1,valid=len(order),empty_fallback=skipped,candidates=candidates,
                    fallback_candidates=int(lengths.sum())-candidates,loss=total_loss/len(order),
                    seconds=perf_counter()-start)
        print(record,flush=True)
        records.append(record)
    require(all(p.isfinite().all().item() for p in model.parameters()),"nonfinite weights")
    return records


@torch.inference_mode()
def compare(model, bank, data, valid, index, pool, corpus, queries):
    model.eval()
    groups=("tower","din")
    official={k:[] for k in groups}
    sums={k:[0.,0.] for k in groups}
    pool_n=end_n=empty=hits=0
    corpus_index=np.array([index[n] for n in corpus],dtype=np.int64)
    for start in range(0,len(valid),32):
        sel=np.arange(start,min(start+32,len(valid)))
        h,l,orig,_=unpack(data,sel)
        flat_pool=pool[sel]
        candidate=corpus_index[flat_pool].ravel()
        first=batch_logits(model,bank,h,np.full(len(sel),100),candidate).reshape(-1,100).cpu().numpy()
        second=batch_logits(model,bank,h,l,orig).cpu().numpy()
        require(np.isfinite(first).all() and np.isfinite(second).all(),"nonfinite DIN evaluation")
        q=torch.as_tensor(queries[sel],device=bank.device)
        dot=(bank[torch.as_tensor(orig,device=bank.device)]*q[torch.as_tensor(np.repeat(np.arange(len(l)),l),device=bank.device)]).sum(-1).cpu().numpy()
        require(np.isfinite(dot).all(),"nonfinite control")
        pos=0
        for j,i in enumerate(sel):
            row=valid[i]
            clicked=list(row.clicks())
            names=[corpus[n] for n in flat_pool[j]]
            fallback=not row.history
            empty+=int(fallback)
            reranked=fixed_order(names,first[j],fallback)
            require(len(reranked)==100 and set(reranked)==set(names),"candidates changed")
            orig_tower=ranks_from_scores(row.candidates,dot[pos:pos+l[j]])
            orig_din=orig_tower if fallback else ranks_from_scores(row.candidates,second[pos:pos+l[j]])
            pos+=l[j]
            valid_imp=0<sum(row.labels)<len(row.labels)
            end_n+=int(valid_imp)
            set_names=set(names)
            count=sum(n in set_names for n in clicked)
            hits+=count
            pool_n+=int(count>0)
            for k,ordering,ranks in (("tower",names,orig_tower),("din",reranked,orig_din)):
                within,end=pool_scores(ordering,clicked)
                if within is not None:sums[k][0]+=within
                if valid_imp:sums[k][1]+=end
                official[k].append(Impression(row.labels,ranks))
    result={}
    for k in groups:
        report=evaluate(official[k])
        require(report["scoring_denominator"]==end_n,"official denominator differs")
        result[k]=dict(pool_ndcg10=sums[k][0]/pool_n,end_ndcg10=sums[k][1]/end_n,official=report)
    return result,dict(pool_denominator=pool_n,end_denominator=end_n,hits=hits,empty_fallback=empty)


def run(cfg,seed,pool_digest):
    start=perf_counter()
    base,encoder,ref,folder=sources(cfg)
    out=ROOT/"checkpoints/card-009"
    require(sha256_file(out/"pool.json")==pool_digest,"fixed manifest changed")
    manifest=read(out/"pool.json")
    require(manifest["commit"]==head() and manifest["inputs"]==ref["inputs_sha256"] and
            manifest["flat"]==ref["artifacts"][FLAT] and sha256_file(ROOT/manifest["path"])==manifest["sha256"],"pool identity differs")
    pool=np.load(ROOT/manifest["path"],allow_pickle=False)
    corpus=read(folder/"news-ids.json")
    queries=np.load(folder/"queries.npy",allow_pickle=False)
    rows,texts=read_sources(base)
    train_rows,_=split_prefix(rows,registered_ref("RV-data-profile",snapshot())["split"],base)
    valid=rows[len(train_rows):]
    meta=read(folder/"queries.json")
    require(all(r.impression_id==m["id"] and list(r.clicks())==m["clicks"] for r,m in zip(valid,meta,strict=True)),
            "validation identity differs")
    fit_ids={n for row in train_rows for source in (row.history,row.candidates) for n in source}
    all_ids=fit_ids.union(n for row in valid for source in (row.history,row.candidates) for n in source)
    names,vocab,table=build_catalog(texts,fit_ids,all_ids,base)
    index={n:i for i,n in enumerate(names,1)}
    frozen=ROOT/"checkpoints/card-004/s0"
    require(read(frozen/"vocabulary.json")==vocab,"vocabulary differs")
    tower=MeanTowers(len(vocab),cfg["dim"])
    tower.load_state_dict(torch.load(frozen/"model.pt",map_location="cpu",weights_only=True))
    require(weight_identity(tower)==encoder["trained_weight_sha256"],"encoder identity differs")
    with torch.no_grad():
        bank=torch.cat([tower.news(chunk) for chunk in table.split(256)])
        require(np.array_equal(bank[[index[n] for n in corpus]].numpy(),np.load(folder/"news.npy")),
                "corpus vectors differ")
        vh=torch.from_numpy(histories(valid,index,cfg["history"]))
        check=torch.cat([tower.aggregate_history(bank[h],h) for h in vh.split(256)])
        require(np.array_equal(check.numpy(),queries),"frozen queries differ")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bank=bank.to(device)
    torch.manual_seed(seed)
    model=DIN(cfg["dim"],cfg["attention_widths"],cfg["head_width"]).to(device)
    records=train(model,bank,batches_data(train_rows,index,cfg["history"]),cfg,seed)
    weights=out/f"s{seed}.pt"
    require(not weights.exists(),"existing checkpoint")
    torch.save(model.state_dict(),weights)
    metrics,counts=compare(model,bank,batches_data(valid,index,cfg["history"]),valid,index,pool,corpus,queries)
    require((counts["hits"],counts["end_denominator"],counts["pool_denominator"])==
            (manifest["counts"]["hits"],len(valid),manifest["counts"]["pool_exposures"]),"metric denominators differ")
    payload=dict(seed=seed,config_sha256=sha256_file(ROOT/CONFIG),reference=cfg["faiss_payload"],
                 pool_manifest_sha256=pool_digest,pool=manifest,epochs=records,metrics=metrics,counts=counts,
                 weights={weights.relative_to(ROOT).as_posix():sha256_file(weights)})
    dest=f"scratch/card-009/RV-din-001-s{seed}.json"
    require(not (ROOT/dest).exists(),"existing RV")
    write_result(dest,payload,"RV",Path(dest).stem,CONFIG,seed,
                 dict(elapsed_seconds=perf_counter()-start,device=str(device)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pool",action="store_true")
    mode.add_argument("--seed",type=int,choices=(0,1,2))
    parser.add_argument("--pool-sha256")
    args=parser.parse_args()
    cfg=read(ROOT/CONFIG)
    require(not status_paths() and head()==git("rev-parse","card-009",check=True).stdout.strip(),"requires clean card-009")
    faiss.omp_set_num_threads(1)
    torch.set_num_threads(1)
    if args.pool:
        rebuild(cfg)
    else:
        timer=Timer(cfg["seconds"],lambda: os._exit(124))
        timer.daemon=True
        timer.start()
        try:
            run(cfg,args.seed,args.pool_sha256)
        finally:
            timer.cancel()


if __name__=="__main__":
    main()
