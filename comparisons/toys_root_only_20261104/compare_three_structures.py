"""Compare completed Toys native-two, random-three and Root results without GPU evaluation."""
from copy import deepcopy
from pathlib import Path
import json

import run_suite as q

p = q.p
ARMS = ("root", "random_three", "native_two")
LABELS = {"root": "Root 单层", "random_three": "随机三层", "native_two": "原生社区两层"}
METRICS = ("hr@5", "hr@10", "ndcg@5", "ndcg@10", "tail_hr@5", "tail_hr@10",
           "tail_ndcg@5", "tail_ndcg@10", "tail_exposure@10")
METRIC_LABELS = {key: key.replace("tail_", "长尾 ").replace("exposure", "曝光") for key in METRICS}
PAIRS = (("root", "native_two"), ("root", "random_three"), ("random_three", "native_two"))


def shared_settings(config):
    result = deepcopy(config)
    for key in ("experiment", "partition"):
        result.pop(key, None)
    for key in ("level_rates", "initial_shares", "dynamic_rate_gate", "fine_size", "coarse_size"):
        result["diffusion"].pop(key, None)
    return result


def main():
    previous_root = p.read_json(q.OUTPUT / "summary.json")
    previous_random = p.read_json(q.RANDOM / "runs/summary.json")
    assert previous_root["state"] == "complete"
    assert p.read_json(q.RANDOM / "runs/status.json")["state"] == "complete"
    rows, checks = [], []
    for seed in q.SEEDS:
        native = q.SOURCE / f"seed_{seed}"
        native_config = p.read_json(native / "resolved_config.json")
        native_artifact = p.torch.load(native / "structure.pt", map_location="cpu", weights_only=True)
        for arm in ARMS:
            folder = (q.OUTPUT / f"root/seed_{seed}" if arm == "root" else
                      q.RANDOM / f"runs/random_three/seed_{seed}" if arm == "random_three" else native)
            row = q.verify_completed(folder, root=arm == "root")
            config = p.read_json(folder / "resolved_config.json")
            assert config["seed"] == seed
            assert shared_settings(config) == shared_settings(native_config)
            assert p.prepared_root(folder).resolve() == p.prepared_root(native).resolve()
            artifact = p.torch.load(folder / "structure.pt", map_location="cpu", weights_only=True)
            assert all(p.torch.equal(artifact[key], native_artifact[key])
                       for key in ("base", "neighbor_ids", "neighbor_weights"))
            p.verify_reuse(folder, folder, config, artifact)
            tail = p.tail_metrics(folder)
            q.verify_tail(tail)
            assert tail == p.read_json(folder / "long_tail.json")
            report = p.read_json(folder / "report.json")
            assert report["evaluation"] == "full_catalog" and report["selection_split"] == "val"
            assert report["dynamic_rate_gate"] == (arm != "root")
            if arm == "root":
                assert report["test_rate_shares"] is None
                assert artifact["blocks"].shape == (1, 11924) and not artifact["blocks"].any()
            elif arm == "random_three":
                assert config["diffusion"]["initial_shares"] == [.1920355, .0244645, .7835]
                assert (config["diffusion"]["fine_size"], config["diffusion"]["coarse_size"]) == (64, 1024)
                rank = p.torch.empty(11924, dtype=p.torch.long)
                order = p.torch.randperm(11924, generator=p.torch.Generator().manual_seed(100))
                rank[order] = p.torch.arange(11924)
                expected = p.torch.stack((rank // 64, rank // 1024, p.torch.zeros_like(rank)))
                assert p.torch.equal(artifact["blocks"], expected)
            source_summary = previous_random if arm == "random_three" else previous_root
            saved_row = next(r for r in source_summary["rows"] if r["arm"] == arm and r["seed"] == seed)
            assert all(row["metrics"][key] == saved_row["metrics"][key] for key in METRICS)
            rows.append({"arm": arm, **row, "test_rate_shares": report["test_rate_shares"]})
            checks.append({"arm": arm, "seed": seed, "same_prepared_splits": True,
                           "same_base_and_neighbors": True, "same_shared_settings": True,
                           "checkpoint_and_full_log_match": True, "long_tail_recomputed": True})

    by_arm = {arm: {row["seed"]: row for row in rows if row["arm"] == arm} for arm in ARMS}
    groups = {arm: {"seeds": list(q.SEEDS), "metrics": {
        key: p.moments([by_arm[arm][seed]["metrics"][key] for seed in q.SEEDS])
        for key in METRICS}} for arm in ARMS}
    pairwise = {}
    for candidate, reference in PAIRS:
        values = {}
        for key in METRICS:
            cand = [by_arm[candidate][seed]["metrics"][key] for seed in q.SEEDS]
            ref = [by_arm[reference][seed]["metrics"][key] for seed in q.SEEDS]
            values[key] = {
                "paired_difference": p.moments([a - b for a, b in zip(cand, ref)]),
                "relative_change_percent": (groups[candidate]["metrics"][key]["mean"] /
                                             groups[reference]["metrics"][key]["mean"] - 1) * 100,
                "paired_relative_change_percent": p.moments([(a / b - 1) * 100 for a, b in zip(cand, ref)]),
                "paired_wins": sum(a > b for a, b in zip(cand, ref)),
            }
        pairwise[f"{candidate}_vs_{reference}"] = {
            "candidate": candidate, "reference": reference, "paired_seeds": list(q.SEEDS), "metrics": values}
    result = {"state": "complete", "updated_at": p.now(), "dataset": "Toys",
              "aggregation": "3 streams per fitted model, then 3 training-seed means and sample std (ddof=1)",
              "tail_protocol": {"counts": "training history plus target", "strict_frequency_less_than": 12,
                                "tail_catalog_items": 9367, "tail_test_examples": 1035, "test_examples": 1942},
              "comparison_scope": "Complete structural settings: Root has no gate; native two and random three learn shares. Random three also changes partition, depth and initial shares.",
              "rows": rows, "groups": groups, "pairwise": pairwise, "verification": checks}
    p.write_json(q.OUTPUT / "three_way_summary.json", result)
    def fmt(arm, key):
        value = groups[arm]["metrics"][key]
        return f"{value['mean']:.6f} ± {value['std']:.6f}"
    lines = ["# Toys：Root 单层、随机三层、原生社区两层", "",
             "共用 all_positions 新邻居和 prefix 新 b；训练种子 100、101、102。",
             "按验证 NDCG@10 选 checkpoint，完整目录测试，每个训练模型平均 3 个 sampling streams；",
             "以下为 3 个训练种子的均值 ± 样本标准差（ddof=1）。", "",
             "长尾：仅训练历史＋目标频次，strict frequency < 12；9367 个物品，1035/1942 条测试目标。", "",
             "| 指标 | Root 单层 | 随机三层 | 原生社区两层 |",
             "| --- | ---: | ---: | ---: |"]
    for key in METRICS:
        lines.append(f"| {METRIC_LABELS[key]} | " + " | ".join(fmt(arm, key) for arm in ARMS) + " |")
    lines += ["", "相对变化使用三种子均值之比减一；正值表示前者更高。", "",
              "| 指标 | Root / 两层 | Root / 随机三层 | 随机三层 / 两层 |",
              "| --- | ---: | ---: | ---: |"]
    for key in METRICS:
        values = [pairwise[f"{candidate}_vs_{reference}"]["metrics"][key]["relative_change_percent"]
                  for candidate, reference in PAIRS]
        lines.append(f"| {METRIC_LABELS[key]} | " + " | ".join(f"{v:+.2f}%" for v in values) + " |")
    lines += ["", "| Seed | Root 最佳 epoch | 随机三层最佳 epoch | 两层最佳 epoch | Root 测试 NDCG@10 | 随机三层测试 NDCG@10 | 两层测试 NDCG@10 |",
              "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for seed in q.SEEDS:
        epochs = [str(by_arm[arm][seed]["checkpoint_epoch"]) for arm in ARMS]
        metrics = [f"{by_arm[arm][seed]['metrics']['ndcg@10']:.6f}" for arm in ARMS]
        lines.append(f"| {seed} | " + " | ".join(epochs + metrics) + " |")
    lines += ["", "九项指标均值均为 Root > 随机三层 > 原生社区两层。",
              "按同种子整体 NDCG@10，Root 对随机三层为 2 胜 1 负，随机三层对原生两层为 1 胜 2 负。",
              "这是三套完整结构设置的比较：Root 关闭门控；原生两层与随机三层学习份额，随机三层还改变分区、层数和初始份额。",
              "因此这些结果不能单独归因为层数或社区划分质量。Root 没有可学习社区份额。", "",
              "JSON 保留逐种子结果、配对差值的均值和样本标准差、逐种子相对变化以及原始结果路径。"]
    path = q.OUTPUT / "three_way_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)
    print(json.dumps({"groups": groups, "pairwise": pairwise}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
