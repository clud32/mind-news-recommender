# MIND 新闻推荐 · G3 离线实验快照

这是 G3 已验收实验的脱敏发布快照，面向离线汇总验证。包含评测代码、配置、汇总结果、报告与数据身份清单。服务、完整重建验收和 v1.0 发布按后续任务卡实施。

## 在干净检出验证结果

在仓库根目录使用 Python 3.11 运行。此入口仅使用标准库，直接验证发布文件校验值、结果 payload、逐项重算的指标和结论，以及重新生成的宏和数字追溯表。

```powershell
python -B tools/verify_g3_release.py
```

[发布清单](release/MANIFEST.json)给出发布版身份值。`results` 内原 RV 的运行提交、原始输入身份和证据路径描述内部实验来源。改动后的 payload 身份和发布版引用以发布清单为准。发布版的结果校验不依赖内部 Git 标签、数据文件或训练产物。

## 数据和实验产物

MIND-small train/dev 使用 DEC-002 冻结的 Recommenders/MIND 发行版，来源、版本和 SHA-256 见[数据清单](data/MANIFEST.sha256)。取得数据需遵守 MIND 数据条款，并在本地按清单核验后解包。与微软主页所链接发行包的 TSV 等价性待验证。

原始数据、模型权重、索引和逐曝光证据需在本地按各自流程获取或生成。实验依赖见 `requirements.lock`。均值双塔、FAISS、DIN、审计与最终评测的入口在 `src/run_mind_*.py`，对应配置在 `configs/`。原实验的 `tools/check.py` 依赖内部 `authority/REGISTRY.md`、`card-011` 标签、`.venv` 环境锁和 DATA 清单中的本地 ZIP/TSV。旧报告检查依赖该登记簿。本快照使用上面的发布验证入口。评测入口的实际运行还需要本地微软评分器和检查点。

## 实测结论

small-dev 的 73,152 条曝光中，双塔三种子平均官方曝光排序 nDCG@10 为 0.3651，热度基线为 0.3518，最近点击基线为 0.3062。全时序候选池中，双塔平均 Recall@100 为 2.86%，热度基线为 62.94%。固定候选池上的 DIN 端到端配对区间跨零，尚无稳定收益证据。微软评分器用于核对同一份预测的计分，项目自建基线用于模型比较。

[结果表](report/g3-results.tex) · [结论与证据](results/conclusions.json) · [数字追溯](results/number_trace.csv)
