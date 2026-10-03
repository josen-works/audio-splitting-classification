"""Write results as JSON, CSV and a bar chart."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def write_reports(results: dict, out_dir) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(results, indent=2))

    rows = []
    for stem, r in results["stems"].items():
        for rank, c in enumerate(r["top_classes"], 1):
            rows.append({"stem": stem, "rank": rank, "label": c["label"], "score": c["score"]})
    csv_path = out_dir / "classification.csv"
    pd.DataFrame(rows, columns=["stem", "rank", "label", "score"]).to_csv(csv_path, index=False)

    plot_path = out_dir / "classification.png"
    stems = results["stems"]
    fig, axes = plt.subplots(1, len(stems), figsize=(4 * len(stems), 3.5), squeeze=False)
    for ax, (stem, r) in zip(axes[0], stems.items()):
        ax.set_title(stem + (" (silent)" if r["silent"] else ""))
        if r["top_classes"]:
            labels = [c["label"] for c in r["top_classes"]][::-1]
            ax.barh(labels, [c["score"] for c in r["top_classes"]][::-1])
            ax.set_xlim(0, 1)
        else:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    return {"json": json_path, "csv": csv_path, "plot": plot_path}


def print_summary(results: dict):
    print(f"\n{results['input']}  ({results['duration_s']:.1f}s, demucs={results['demucs_model']})")
    for stem, r in results["stems"].items():
        if r["silent"]:
            db = r.get("level_db_vs_loudest")
            print(f"  {stem:8s} silent" + (f" ({db:+.0f} dB vs loudest stem)" if db is not None else ""))
            continue
        top = ", ".join(f"{c['label']} {c['score']:.2f}" for c in r["top_classes"][:3])
        print(f"  {stem:8s} {top}")
