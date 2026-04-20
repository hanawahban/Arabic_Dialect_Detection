import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

with open("results/all_results.json") as f:
    results = json.load(f)

dialects = ["EGY", "GLF", "LEV", "MGH", "Tunisien"]

for r in results:
    cm = np.array(r["Confusion Matrix"])
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=dialects, yticklabels=dialects, ax=ax)
    ax.set_title(f"Confusion Matrix: {r['Model']} ({r['Features']})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    filename = f"results/cm_{r['Model'].replace(' ', '_')}_{r['Features'].replace(' ', '_')}.png"
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"Saved: {filename}")