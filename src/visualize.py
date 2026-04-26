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

models = [f"{r['Model']} ({r['Features']})" for r in results]
accuracies = [r["Accuracy"] for r in results]

plt.figure(figsize=(11, 6))
plt.bar(models, accuracies)

plt.xlabel("Model and Feature Type")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison Across All Experiments")

plt.xticks(rotation=45, ha="right")
plt.ylim(0, 1)
plt.tight_layout()

plt.savefig("results/accuracy_comparison_all_experiments.png", dpi=150)
plt.show()

print("Saved: results/accuracy_comparison_all_experiments.png")
model_names = ["Naive Bayes", "Logistic Regression", "Linear SVM", "Random Forest", "KNN"]

word_acc = []
char_acc = []

for model in model_names:
    word_result = next(r for r in results if r["Model"] == model and r["Features"] == "Word TF-IDF")
    char_result = next(r for r in results if r["Model"] == model and r["Features"] == "Char TF-IDF")
    
    word_acc.append(word_result["Accuracy"])
    char_acc.append(char_result["Accuracy"])

x = np.arange(len(model_names))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, word_acc, width, label="Word TF-IDF")
plt.bar(x + width/2, char_acc, width, label="Char TF-IDF")

plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Word TF-IDF vs Char TF-IDF Accuracy Comparison")

plt.xticks(x, model_names, rotation=30, ha="right")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()

plt.savefig("results/word_vs_char_tfidf_comparison.png", dpi=150)
plt.show()

print("Saved: results/word_vs_char_tfidf_comparison.png")