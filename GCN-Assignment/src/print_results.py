
from pathlib import Path
import csv
import re
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR / "results" / "citeseer_three_layer"
OUTPUT_DIR = BASE_DIR / "results2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_FILE = SOURCE_DIR / "predictions.csv"
METRICS_FILE = SOURCE_DIR / "metrics.txt"

# Check that existing model results are available
if not PREDICTIONS_FILE.exists():
    raise FileNotFoundError(
        f"Could not find predictions.csv at:\n{PREDICTIONS_FILE}"
    )

# Read the saved predictions
with open(PREDICTIONS_FILE, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

if not rows:
    raise ValueError("predictions.csv is empty.")

# Find columns without depending on capitalization
columns = {c.strip().lower(): c for c in rows[0].keys() if c}

actual_col = next(
    (columns[k] for k in columns if k in ("actual", "actual_class", "true_label", "true")),
    None
)
predicted_col = next(
    (columns[k] for k in columns if k in ("predicted", "predicted_class", "predicted_label", "prediction")),
    None
)

if actual_col is None or predicted_col is None:
    raise ValueError(
        "Could not identify actual and predicted columns. "
        f"Found columns: {list(rows[0].keys())}"
    )

actual = [row[actual_col].strip().strip('"') for row in rows]
predicted = [row[predicted_col].strip().strip('"') for row in rows]

# Use a consistent class order
classes = sorted(set(actual) | set(predicted))
n = len(actual)

# Calculate confusion matrix
matrix = [[0 for _ in classes] for _ in classes]
for a, p in zip(actual, predicted):
    matrix[classes.index(a)][classes.index(p)] += 1

# Calculate per-class precision, recall and F1
class_metrics = {}
for i, label in enumerate(classes):
    tp = matrix[i][i]
    fp = sum(matrix[r][i] for r in range(len(classes))) - tp
    fn = sum(matrix[i]) - tp

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )
    class_metrics[label] = (precision, recall, f1)

# Overall accuracy calculated from saved predictions
accuracy = sum(a == p for a, p in zip(actual, predicted)) / n

# Macro metrics calculated from the per-class values
macro_precision = sum(m[0] for m in class_metrics.values()) / len(classes)
macro_recall = sum(m[1] for m in class_metrics.values()) / len(classes)
macro_f1 = sum(m[2] for m in class_metrics.values()) / len(classes)

# Prefer the model's recorded evaluation metrics if available
if METRICS_FILE.exists():
    text = METRICS_FILE.read_text(encoding="utf-8", errors="replace")
    for name, variable in [
        ("Accuracy", "accuracy"),
        ("Macro Precision", "macro_precision"),
        ("Macro Recall", "macro_recall"),
        ("Macro F1-score", "macro_f1"),
    ]:
        match = re.search(
            rf"{re.escape(name)}\s*[:=]\s*([0-9]*\.?[0-9]+)",
            text,
            re.IGNORECASE
        )
        if match:
            value = float(match.group(1))
            if variable == "accuracy":
                accuracy = value
            elif variable == "macro_precision":
                macro_precision = value
            elif variable == "macro_recall":
                macro_recall = value
            elif variable == "macro_f1":
                macro_f1 = value

# 1. Overall evaluation metrics
metric_names = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1-score"]
metric_values = [
    accuracy, macro_precision, macro_recall, macro_f1
]

plt.figure(figsize=(9, 5))
bars = plt.bar(metric_names, metric_values)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Three-Layer GCN - Overall Evaluation Metrics")
plt.xticks(rotation=12)
plt.bar_label(bars, fmt="%.3f", padding=3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "evaluation_metrics.png", dpi=300)
plt.close()

# 2. Per-class evaluation metrics
plt.figure(figsize=(10, 6))
x = list(range(len(classes)))
width = 0.25

plt.bar(
    [i - width for i in x],
    [class_metrics[c][0] for c in classes],
    width, label="Precision"
)
plt.bar(
    x,
    [class_metrics[c][1] for c in classes],
    width, label="Recall"
)
plt.bar(
    [i + width for i in x],
    [class_metrics[c][2] for c in classes],
    width, label="F1-score"
)

plt.xticks(x, classes)
plt.ylim(0, 1)
plt.xlabel("CiteSeer Class")
plt.ylabel("Score")
plt.title("Three-Layer GCN - Per-Class Evaluation Metrics")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "per_class_metrics.png", dpi=300)
plt.close()

# 3. Confusion matrix
plt.figure(figsize=(8, 7))
plt.imshow(matrix, interpolation="nearest")
plt.title("Three-Layer GCN - Confusion Matrix")
plt.colorbar(label="Number of Papers")
plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
plt.yticks(range(len(classes)), classes)
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

threshold = max(max(row) for row in matrix) / 2 if matrix else 0
for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(
            j, i, str(matrix[i][j]),
            ha="center", va="center",
            color="white" if matrix[i][j] > threshold else "black"
        )

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=300)
plt.close()

print("\nThree-Layer GCN Evaluation PNGs created successfully!")
print("Saved in:", OUTPUT_DIR)
print("\nOverall Metrics:")
for name, value in zip(metric_names, metric_values):
    print(f"{name}: {value:.4f}")

print("\nGenerated files:")
print("1. evaluation_metrics.png")
print("2. per_class_metrics.png")
print("3. confusion_matrix.png")
