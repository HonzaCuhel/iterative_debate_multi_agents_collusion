import os
import json
from typing import Any, Dict, List, Tuple

import matplotlib


# Use a non-interactive backend so the script can run headless and save PNGs
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
try:
    import seaborn as sns  # type: ignore  # noqa: F401
except Exception:
    sns = None


def get_repo_root() -> str:
    """Return the absolute path to the project root directory.

    This file lives in `<repo>/src/plot_results.py`, so the repo root is one level up.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def list_json_files(directory_path: str) -> List[str]:
    """Return a list of absolute paths to .json files in a directory.

    The order is sorted for reproducibility.
    """
    if not os.path.isdir(directory_path):
        return []
    files = [
        os.path.join(directory_path, filename)
        for filename in os.listdir(directory_path)
        if filename.endswith(".json")
    ]
    return sorted(files)


def extract_model_triple(meta_models: Dict[str, Any]) -> Tuple[str, str, str]:
    """Return a canonical (bob, alice, judge) model triple from the meta.models block."""
    bob_model = str(meta_models.get("bob", "unknown"))
    alice_model = str(meta_models.get("alice", "unknown"))
    judge_model = str(meta_models.get("judge", "unknown"))
    return bob_model, alice_model, judge_model


def compute_per_run_metrics(summary: Dict[str, Any]) -> Dict[str, float]:
    """Compute normalized metrics from the run summary block.

    Expected keys inside `summary` based on observed schema:
    - truthful_win_percentage, lying_win_percentage, draw_percentage
    - judge_correct_count, judge_incorrect_count, total_questions
    If percentages are present, prefer them; otherwise derive from counts.
    """
    # Prefer provided percentages if present
    truthful_win_rate = summary.get("truthful_win_percentage")
    lying_win_rate = summary.get("lying_win_percentage")
    draw_rate = summary.get("draw_percentage")

    total_questions = summary.get("total_questions")
    judge_correct_count = summary.get("judge_correct_count")
    judge_incorrect_count = summary.get("judge_incorrect_count")

    # Fallbacks: derive rates if percentages missing but counts present
    if truthful_win_rate is None and isinstance(total_questions, (int, float)) and total_questions:
        truthful_win_count = summary.get("truthful_agent_won_count")
        truthful_win_rate = (truthful_win_count or 0) / float(total_questions)

    if lying_win_rate is None and isinstance(total_questions, (int, float)) and total_questions:
        lying_win_count = summary.get("lying_agent_won_count")
        lying_win_rate = (lying_win_count or 0) / float(total_questions)

    if draw_rate is None and isinstance(total_questions, (int, float)) and total_questions:
        draw_count = summary.get("draw_count")
        draw_rate = (draw_count or 0) / float(total_questions)

    # Judge accuracy: prefer computing from counts for robustness
    judge_total = None
    if isinstance(judge_correct_count, (int, float)) and isinstance(judge_incorrect_count, (int, float)):
        judge_total = judge_correct_count + judge_incorrect_count
    if not judge_total and isinstance(total_questions, (int, float)):
        judge_total = total_questions

    judge_accuracy = None
    if judge_total:
        judge_accuracy = float(judge_correct_count or 0) / float(judge_total)

    return {
        "truthful_win_rate": float(truthful_win_rate) if truthful_win_rate is not None else 0.0,
        "lying_win_rate": float(lying_win_rate) if lying_win_rate is not None else 0.0,
        "draw_rate": float(draw_rate) if draw_rate is not None else 0.0,
        "judge_accuracy": float(judge_accuracy) if judge_accuracy is not None else 0.0,
    }


COUNT_KEYS: List[str] = [
    "alice_truthful_count",
    "bob_truthful_count",
    "alice_won_count",
    "bob_won_count",
    "truthful_agent_won_count",
    "lying_agent_won_count",
    "draw_count",
]


def compute_per_run_counts(summary: Dict[str, Any]) -> Dict[str, int]:
    """Extract raw count metrics from the run summary block.

    Includes counts requested by the user, plus judge correctness counts.
    """
    counts: Dict[str, int] = {}
    for key in COUNT_KEYS:
        counts[key] = int(summary.get(key, 0) or 0)
    counts["judge_correct_count"] = int(summary.get("judge_correct_count", 0) or 0)
    counts["judge_incorrect_count"] = int(summary.get("judge_incorrect_count", 0) or 0)
    counts["total_questions"] = int(summary.get("total_questions", 0) or 0)
    return counts


def load_runs(results_dir: str) -> List[Dict[str, Any]]:
    """Load all runs from a results directory and return normalized records.

    Each record contains: `bob`, `alice`, `judge`, `metrics` and `path`.
    """
    records: List[Dict[str, Any]] = []
    for json_path in list_json_files(results_dir):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            # Skip malformed files but continue processing others
            continue

        meta = data.get("meta", {})
        models = meta.get("models", {})
        bob_model, alice_model, judge_model = extract_model_triple(models)

        summary = data.get("summary", {})
        metrics = compute_per_run_metrics(summary)
        counts = compute_per_run_counts(summary)

        records.append(
            {
                "bob": bob_model,
                "alice": alice_model,
                "judge": judge_model,
                "metrics": metrics,
                "counts": counts,
                "path": json_path,
            }
        )
    return records


def group_by_model_triple(records: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str], List[Dict[str, float]]]:
    """Group run metrics by (bob, alice, judge) model triple."""
    grouped: Dict[Tuple[str, str, str], List[Dict[str, float]]] = {}
    for rec in records:
        key = (rec["bob"], rec["alice"], rec["judge"])
        grouped.setdefault(key, []).append(rec["metrics"]) 
    return grouped


def aggregate_group(metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Compute mean and standard deviation for each metric across runs.

    Returns a mapping of metric_name -> {"mean": float, "std": float, "n": int}.
    """
    if not metrics_list:
        return {}

    metric_names = list(metrics_list[0].keys())
    result: Dict[str, Dict[str, float]] = {}
    for name in metric_names:
        values = [float(m.get(name, 0.0)) for m in metrics_list]
        n = len(values)
        mean_val = sum(values) / float(n) if n else 0.0
        # Population std if n==1 -> 0.0 to avoid matplotlib warnings
        if n <= 1:
            std_val = 0.0
        else:
            mean_diff_sq_sum = sum((v - mean_val) ** 2 for v in values)
            std_val = (mean_diff_sq_sum / float(n)) ** 0.5
        result[name] = {"mean": mean_val, "std": std_val, "n": float(n)}
    return result


def aggregate_all_groups(grouped: Dict[Tuple[str, str, str], List[Dict[str, float]]]) -> Dict[Tuple[str, str, str], Dict[str, Dict[str, float]]]:
    """Aggregate metrics for every model triple group."""
    return {key: aggregate_group(metrics) for key, metrics in grouped.items()}


def slugify(text: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in text).strip("_")


def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def make_xtick_label(bob: str, alice: str, judge: str) -> str:
    return f"bob={bob}\nalice={alice}\njudge={judge}"


def make_legend_label(bob: str, alice: str, judge: str) -> str:
    return f"bob={bob}, alice={alice}, judge={judge}"


def get_palette(num_colors: int) -> List[Any]:
    if sns is not None:
        try:
            return list(sns.color_palette("deep", n_colors=num_colors))
        except Exception:
            pass
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(num_colors)]


def plot_joint(
    dataset_name: str,
    aggregates: Dict[Tuple[str, str, str], Dict[str, Dict[str, float]]],
    counts_grouped: Dict[Tuple[str, str, str], Dict[str, int]],
    judge_totals: Tuple[int, int],
    output_path: str,
) -> None:
    """Create a single joint plot per dataset with seaborn styling.

    The figure contains two vertically stacked subplots:
    - Top: grouped bars for truthful/lying/draw rates with error bars
    - Bottom-left: judge accuracy pie chart (correct vs incorrect)
    - Bottom-right: grouped counts per model triple for requested metrics
    """
    if not aggregates:
        return

    triples = list(aggregates.keys())
    triples.sort(key=lambda t: (t[0], t[1], t[2]))

    labels = [make_xtick_label(*t) for t in triples]
    truthful_means = [aggregates[t]["truthful_win_rate"]["mean"] for t in triples]
    truthful_stds = [aggregates[t]["truthful_win_rate"]["std"] for t in triples]
    lying_means = [aggregates[t]["lying_win_rate"]["mean"] for t in triples]
    lying_stds = [aggregates[t]["lying_win_rate"]["std"] for t in triples]
    draw_means = [aggregates[t]["draw_rate"]["mean"] for t in triples]
    draw_stds = [aggregates[t]["draw_rate"]["std"] for t in triples]

    # Counts per triple for requested categories
    categories = COUNT_KEYS
    counts_by_triple = [
        [int(counts_grouped.get(t, {}).get(cat, 0)) for cat in categories]
        for t in triples
    ]

    judge_correct_total, judge_incorrect_total = judge_totals

    x = list(range(len(triples)))
    width = 0.25

    # Seaborn theme if available
    if sns is not None:
        sns.set_theme(style="whitegrid")

    # Grid: top spans both columns; bottom has two columns
    fig = plt.figure(figsize=(max(10, 3.0 * len(triples)), 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 3])
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    # Top subplot: win/draw rates
    ax1.bar([i - width for i in x], truthful_means, width=width, yerr=truthful_stds, capsize=4, label="Truthful win", color="#2e7d32")
    ax1.bar(x, lying_means, width=width, yerr=lying_stds, capsize=4, label="Lying win", color="#c62828")
    ax1.bar([i + width for i in x], draw_means, width=width, yerr=draw_stds, capsize=4, label="Draw", color="#757575")
    ax1.set_title(f"{dataset_name} — Outcomes by Model Triple")
    ax1.set_ylabel("Rate (0–1)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0.0, 1.0)
    ax1.legend(frameon=True)

    # Bottom-left: judge accuracy pie (overall)
    total = judge_correct_total + judge_incorrect_total
    ax2.set_title("Judge Decisions (Overall)")
    if total > 0:
        sizes = [judge_correct_total, judge_incorrect_total]
        explode = (0.03, 0.03)
        colors = ["#2e7d32", "#c62828"]
        ax2.pie(
            sizes,
            labels=["Correct", "Incorrect"],
            autopct=lambda p: f"{p:.1f}%\n({int(round(p * total / 100.0))})",
            startangle=90,
            explode=explode,
            colors=colors,
            wedgeprops={"linewidth": 1, "edgecolor": "white"},
            textprops={"fontsize": 9},
        )
        ax2.axis("equal")
    else:
        ax2.text(0, 0, "No judge data", ha="center", va="center")
        ax2.axis("off")

    # Bottom-right: grouped counts per category across triples
    ax3.set_title("Counts by Category and Model Triple")
    num_triples = len(triples)
    num_cats = len(categories)
    cat_x = list(range(num_cats))
    bar_width = 0.8 / max(1, num_triples)
    palette = get_palette(num_triples)

    for t_idx, t in enumerate(triples):
        offsets = [cx - 0.4 + t_idx * bar_width + bar_width / 2.0 for cx in cat_x]
        values = counts_by_triple[t_idx]
        ax3.bar(offsets, values, width=bar_width, color=palette[t_idx], label=make_legend_label(*t))

    ax3.set_xticks(cat_x)
    ax3.set_xticklabels([
        "alice_truthful",
        "bob_truthful",
        "alice_won",
        "bob_won",
        "truthful_wins",
        "lying_wins",
        "draws",
    ], rotation=20)
    ax3.set_ylabel("Count")
    ax3.grid(axis="y", linestyle=":", alpha=0.4)
    ax3.legend(loc="upper right", fontsize=8, frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def print_aggregate_table(dataset_name: str, aggregates: Dict[Tuple[str, str, str], Dict[str, Dict[str, float]]]) -> None:
    """Print a concise text summary to stdout."""
    if not aggregates:
        print(f"[WARN] No aggregates for {dataset_name}")
        return

    print(f"\n===== {dataset_name} (grouped by model triple) =====")
    header = (
        "bob | alice | judge | n | truthful_win_rate(mean±std) | lying_win_rate(mean±std) | draw_rate(mean±std) | judge_accuracy(mean±std)"
    )
    print(header)
    print("-" * len(header))
    for (bob, alice, judge), stats_map in sorted(aggregates.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        n = int(stats_map["truthful_win_rate"]["n"]) if "truthful_win_rate" in stats_map else 0
        def fmt(metric: str) -> str:
            m = stats_map.get(metric, {"mean": 0.0, "std": 0.0})
            return f"{m['mean']:.3f}±{m['std']:.3f}"
        print(
            f"{bob} | {alice} | {judge} | {n} | "
            f"{fmt('truthful_win_rate')} | {fmt('lying_win_rate')} | {fmt('draw_rate')} | {fmt('judge_accuracy')}"
        )


def main() -> None:
    repo_root = get_repo_root()

    datasets = [
        (
            "Explicit Deception",
            os.path.join(repo_root, "explicit_deception_results", "results"),
            os.path.join(repo_root, "explicit_deception_results", "reports"),
        ),
        (
            "Iterative Collusion",
            os.path.join(repo_root, "iterative_collusion_results", "results"),
            os.path.join(repo_root, "iterative_collusion_results", "reports"),
        ),
    ]

    for dataset_name, results_dir, reports_dir in datasets:
        ensure_dir(reports_dir)
        records = load_runs(results_dir)
        if not records:
            print(f"[WARN] No JSON runs found for {dataset_name} in: {results_dir}")
            continue

        grouped = group_by_model_triple(records)
        aggregates = aggregate_all_groups(grouped)

        # Aggregate counts per model triple and judge totals overall
        counts_grouped: Dict[Tuple[str, str, str], Dict[str, int]] = {}
        judge_correct_total = 0
        judge_incorrect_total = 0
        for rec in records:
            key = (rec["bob"], rec["alice"], rec["judge"]) 
            cg = counts_grouped.setdefault(key, {k: 0 for k in COUNT_KEYS})
            for k in COUNT_KEYS:
                cg[k] = cg.get(k, 0) + int(rec["counts"].get(k, 0))
            judge_correct_total += int(rec["counts"].get("judge_correct_count", 0))
            judge_incorrect_total += int(rec["counts"].get("judge_incorrect_count", 0))

        # Print a concise table to the console
        print_aggregate_table(dataset_name, aggregates)

        # Save a single joint plot to the dataset's reports folder
        base_slug = slugify(dataset_name)
        joint_path = os.path.join(reports_dir, f"{base_slug}_joint_metrics_by_model.png")
        plot_joint(
            dataset_name,
            aggregates,
            counts_grouped,
            (judge_correct_total, judge_incorrect_total),
            joint_path,
        )
        print(f"[OK] Saved joint plot: {joint_path}")


if __name__ == "__main__":
    main()


