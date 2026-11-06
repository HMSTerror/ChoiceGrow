# Toys 新版邻居＋新版 b＋全目录 Root

仅执行 seed 100、101、102 三次新训练。数值代码和 protocol.py 原样复制自
`E:/Grow/comparisons/all_positions_three_datasets_20261104`。
同种子对照为该实验的 `runs/main/Toys/seed_{seed}`。

Root 直接复用同种子 prepared splits、base、neighbor_ids、neighbor_weights。
准备时独立重算 prefix b 和 all_positions 邻居并要求张量逐项相等。
只有 blocks=[1,11924] 全零、level_rates=[原 total_budget]、关闭动态门控、移除
initial_shares，以及实验/分区描述发生变化。

实际生成器公式沿用原代码：
`Q_u=(1-reset_rate)*Lambda*(P_root-I)+reset_rate*(G-I)`。
Lambda=2.302585092994046，Root 有效速率=2.221994614739254，reset=0.035。
G 的每行是 prefix b。Root 无可学习层间份额，测试份额为 null。

## 后台执行

`E:/anaco/python.exe -B -u E:/Grow/comparisons/toys_root_only_20261104/run_suite.py`

协调器每 15 秒核验实际 Python 进程、创建时间、命令行和 GPU 显存。
现有随机三层协调器不共享准入锁，因此先等待其完整队列结束，再启动 Root；
每次启动前重新核验，全局最多两路活跃 GPU 工作进程。
旧 b 实验的保存内存状态的暂停进程仅检查，不恢复或终止。

## 进度与结果

- `runs/status.json`：真实 queued/running/complete/failed、等待原因、资源快照及进程身份。
- `runs/verification.json`：三个同种子的数据、结构、预算公式和长尾核验。
- `runs/root/seed_*/`：配置、结构、job、worker 日志、完整 choice_log、checkpoint、训练/测试结果。
- `runs/root/seed_*/comparison.json`：每个种子的原生两层减 Root，以及相对变化。
- `runs/summary.json`、`runs/summary.md`：整体和长尾 HR/NDCG@5/10、长尾曝光@10，
  三训练种子的均值与样本标准差（ddof=1），以及同种子差值和配对均值的相对变化。

训练用验证 NDCG@10 选最佳 checkpoint，patience=150，每 10 epoch 验证。
每次进度更新遍历完整 choice_log.jsonl 求最佳验证值和 epoch。
训练完成后测试 3 个 sampling streams，不视作 3 次训练。
长尾完全调用 protocol.tail_metrics，并强制核验 strict frequency < 12、9367 个长尾物品、
1035 条长尾测试目标及 1942 条测试样本。

checkpoint 不含 optimizer，不支持训练断点恢复。
完成训练后评估失败，确认同任务无活 worker 时最多重试两次评估；
训练未完成而留下权重或日志时直接标记失败，不重新训练或假称恢复。
