import pandas as pd
from typing import List, Any
from sklearn import metrics
import matplotlib.pyplot as plt

from adaboost import AdaBoost

adaboost = AdaBoost()

# Load dataset
train_dataset = pd.read_csv("dataset/car_train.csv")
test_dataset = pd.read_csv("dataset/car_test.csv")

# Set target attribute for classification
target_attribute = "condition"

# Function to calculat specificity metric
def specificity(y_true: List[Any], y_pred: List[Any]) -> float:
    tp, fn, fp, tn = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
    return tn / (fp + tn)


# Train adaboost classifier
adaboost.fit(dataset=train_dataset, target_attribute=target_attribute)

# Predict values
predictions = adaboost.predict(test_dataset.iloc[:, :-1])

# Reformat true values to compare with predicted values
true_values = test_dataset.iloc[:, -1].tolist()

# Print and plot metrics
accuracy = metrics.accuracy_score(y_true=true_values, y_pred=predictions)
print("Accuracy: \n", accuracy, "\n")

recall = metrics.recall_score(y_true=true_values, y_pred=predictions, pos_label="acc")
print("Recall: \n", recall, "\n")

specificity = specificity(y_true=true_values, y_pred=predictions)
print("Specificity: \n", specificity, "\n\n")

# Plot Confusion Matrix
confusion_matrix = metrics.ConfusionMatrixDisplay.from_predictions(y_true=true_values, y_pred=predictions)
confusion_matrix.figure_.savefig('assets/confusion_matrix.png',dpi=300)

fpr, tpr, thresholds = metrics.roc_curve(
    y_true=[1 if elem == "acc" else 0 for elem in true_values],
    y_score=[1 if elem == "acc" else 0 for elem in predictions],
    pos_label=1,
)
print("False Positive Rate \n", fpr, "\n")
print("True Positive Rate \n", tpr, "\n")
print("Thresholds \n", thresholds, "\n\n")

auc = metrics.auc(fpr, tpr)
print("Area Under the Curve (AUC): \n", auc, "\n")

fpr, tpr, thresholds = metrics.roc_curve(
    y_true=[1 if elem == "acc" else 0 for elem in true_values],
    y_score=[1 if elem == "acc" else 0 for elem in predictions],
)
roc_auc = metrics.auc(fpr, tpr)
roc_curve = metrics.RocCurveDisplay(
    fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="example estimator"
)
# Plot receiver operating characteristic (ROC) curve
roc_curve.plot()
roc_curve.figure_.savefig('assets/roc_curve.png',dpi=300)
plt.show()

f1_score = metrics.f1_score(
    y_true=true_values,
    y_pred=predictions,
    average="binary",
    pos_label="acc",
)
print("F1 score: \n", f1_score, "\n")

report = metrics.classification_report(y_true=true_values, y_pred=predictions)
print("Report: \n", report, "\n\n")
