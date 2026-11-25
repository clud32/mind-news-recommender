# MIND 新闻推荐 · 召回与排序离线评测

基于 MIND-small，用用户点击历史和新闻文本评估新闻推荐。项目按曝光时点回放，仅用此前可见的新闻和点击，分别衡量 Top-100 召回和候选排序。当前公开交付是可校验的 G3 脱敏快照，包含评测代码、配置、汇总结果和报告。

## G3 实测结果

small-dev 共 **73,152 条曝光**。双塔为三个种子的均值，基线和双塔使用相同的评测口径。

| 方法 | 曝光内 nDCG@10 | 时序候选池 Recall@100 |
| --- | ---: | ---: |
| 24 小时点击热度 | 0.3518 | **62.94%** |
| 最近点击 TF-IDF | 0.3062 | 3.48% |
| 均值双塔 | **0.3651** | 2.86% |

双塔在曝光内排序优于两条项目基线，在时序召回落后于热度。DIN 在固定候选池上的端到端配对区间跨零，尚无稳定收益证据。微软官方评分器用于复核同一份预测的计分，模型表现按表中项目基线比较。 [完整结果表](report/g3-results.tex) · [结论与证据](results/conclusions.json) · [数字追溯](results/number_trace.csv)

### 后续优化

围绕全池召回的热度差距，项目在内部历史回放中评估了双塔兴趣分与近 24 小时点击热度的融合，并通过替换用户点击历史检验个性化信号。融合效果待独立时序数据复核。下文的校验入口覆盖已公开的 G3 结果。

## 如何验证已发布结果

新克隆的使用者只需 **Git 和 Python 3.11**。验证脚本使用 Python 标准库，从随仓库发布的汇总结果重算指标和结论，核对 SHA-256，并重新生成报告宏和数字追溯表。

```powershell
git clone https://github.com/clud32/mind-news-recommender.git
cd mind-news-recommender
python -B tools/verify_g3_release.py
```

预期输出包含以下两行。

```text
OK: 79 file SHA-256 values and 29 published payloads
OK: 402 recomputed indicators, 4 conclusions, 406 report citations
```

发布文件与结果的校验值见[发布清单](release/MANIFEST.json)。上述命令在干净检出即可运行，无需安装模型依赖或下载数据。

## 推荐和评测流程

| 环节 | 实现 | 要回答的问题 |
| --- | --- | --- |
| 基线 | 过去 24 小时点击热度、基于最近点击新闻的 TF-IDF 相似度 | 简单方法在同一数据和候选口径下能做到什么水平 |
| 候选召回 | PyTorch 均值双塔将点击历史和新闻文本编码成向量。另用 FAISS Flat、HNSWFlat、IVFPQ 比较精确与近似检索 | 在曝光时可见的新闻池中，Top-100 找回了多少真实点击 |
| 候选排序 | 比较双塔打分和 DIN 风格精排。精排使用相同候选，仅改变顺序 | 给定候选时，点击新闻能否排得更靠前，以及精排后整体表现是否改善 |
| 离线评测审计 | 对照时间与随机切分、替换用户历史、置换标签、改变负例采样 | 模型结论对数据切分、历史信号和候选构造有多敏感 |

曝光内排序在每条曝光原有的候选上计算 AUC、MRR 和 nDCG。时序召回按时间回放，候选池只纳入此前已出现的新闻，使用 Recall@100 统计真实点击的命中比例。端到端指标计入未被召回的点击，用于衡量完整链路。 [最终评测配置](configs/mind-final-v0.1.json)

## 技术栈与复现环境

完整实验的**实测环境**是 Windows x86-64、Python 3.11.9 和 NVIDIA GeForce RTX 4060 Laptop GPU。PyTorch 训练及最终评测使用 CUDA 12.8，FAISS 检索对照使用 CPU。

| 用途 | 依赖 |
| --- | --- |
| 双塔训练与 DIN 精排 | PyTorch 2.10.0 + CUDA 12.8 |
| 向量检索 | FAISS CPU 1.16.0 |
| TF-IDF 与数值计算 | scikit-learn 1.9.1、NumPy 2.4.6 |
| 发布结果校验 | Python 标准库、SHA-256、JSON、CSV |

完整实验的依赖版本见 [requirements.lock](requirements.lock)。在 Windows PowerShell 中可按记录的环境安装依赖。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock --find-links https://download.pytorch.org/whl/cu128/torch/
```

完整重跑还需要自行取得 MIND-small train/dev，按[数据清单](data/MANIFEST.sha256)核对 Recommenders/MIND 发行版的 ZIP 和 TSV，再生成模型权重、FAISS 索引及逐曝光证据。取得数据需遵守 MIND 数据条款。该发行版与微软主页链接版本的 TSV 等价性待验证。

仓库当前交付的是 **G3 脱敏离线快照**。原实验入口位于 [src](src/) 和 [configs](configs/)。原内部核验依赖登记簿和 Git 标签。重跑最终评测还需要本地微软评分器和模型检查点。完整重建验收、在线服务和 v1.0 发布属于后续里程碑。
