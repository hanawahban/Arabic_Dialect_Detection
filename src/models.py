from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time
import json
import pandas as pd


def run_experiment(model, model_name, feature_name, X_train, X_test, y_train, y_test):
    start = time.time()
    model.fit(X_train, y_train)
    train_time = round(time.time() - start, 4)

    start = time.time()
    y_pred = model.predict(X_test)
    pred_time = round(time.time() - start, 4)

    report = classification_report(y_test, y_pred, output_dict=True)
    acc = round(accuracy_score(y_test, y_pred), 4)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"Model: {model_name} | Features: {feature_name}")
    print(f"Accuracy: {acc}")
    print(f"Training Time: {train_time}s | Prediction Time: {pred_time}s")
    print(classification_report(y_test, y_pred))

    return {
        "Model": model_name,
        "Features": feature_name,
        "Accuracy": acc,
        "Macro Precision": round(report["macro avg"]["precision"], 4),
        "Macro Recall": round(report["macro avg"]["recall"], 4),
        "Macro F1": round(report["macro avg"]["f1-score"], 4),
        "Train Time (s)": train_time,
        "Pred Time (s)": pred_time,
        "Confusion Matrix": cm.tolist()
    }


def run_all_experiments(X_train_word, X_test_word, X_train_char, X_test_char, y_train, y_test):
    experiments = [
        # Hamza's models
        (MultinomialNB(),                                        "Naive Bayes",         "Word TF-IDF"),
        (MultinomialNB(),                                        "Naive Bayes",         "Char TF-IDF"),
        (LogisticRegression(max_iter=1000, solver="lbfgs"),      "Logistic Regression", "Word TF-IDF"),
        (LogisticRegression(max_iter=1000, solver="lbfgs"),      "Logistic Regression", "Char TF-IDF"),
        (LinearSVC(max_iter=2000),                               "Linear SVM",          "Word TF-IDF"),
        (LinearSVC(max_iter=2000),                               "Linear SVM",          "Char TF-IDF"),
        # Hana's models
        (RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest",    "Word TF-IDF"),
        (RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest",    "Char TF-IDF"),
        (KNeighborsClassifier(n_neighbors=5),                       "KNN",              "Word TF-IDF"),
        (KNeighborsClassifier(n_neighbors=5),                       "KNN",              "Char TF-IDF"),
    ]

    results = []
    for model, model_name, feature_name in experiments:
        X_train = X_train_word if feature_name == "Word TF-IDF" else X_train_char
        X_test  = X_test_word  if feature_name == "Word TF-IDF" else X_test_char
        print(f"\nRunning {model_name} with {feature_name}...")
        results.append(run_experiment(model, model_name, feature_name, X_train, X_test, y_train, y_test))

    with open("results/all_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nAll results saved to results/all_results.json")

    summary = pd.DataFrame([{k: v for k, v in r.items() if k != "Confusion Matrix"} for r in results])
    print("\nResults Summary:")
    print(summary.to_string(index=False))

    return results