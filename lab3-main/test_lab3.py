import json

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from linear_models import (
    train_linear_regression,
    train_logistic_classifier,
    compare_classical_models,
)

from grader_contracts.linear_models import (
    RegressionInput,
    ClassificationInput,
)


def classification_metrics(
    y_true,
    probabilities,
    predictions,
):
    return {
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
    }


print("=== LINEAR REGRESSION ===")

student_df = pd.read_csv(
    "Student_Performance.csv"
)

student_df[
    "Extracurricular Activities"
] = student_df[
    "Extracurricular Activities"
].map(
    {
        "Yes": 1,
        "No": 0,
    }
)

X_regression = student_df.drop(
    columns=["Performance Index"]
)

y_regression = student_df[
    "Performance Index"
]

(
    X_train_reg,
    X_test_reg,
    y_train_reg,
    y_test_reg,
) = train_test_split(
    X_regression,
    y_regression,
    test_size=0.2,
    random_state=42,
)

regression_data = RegressionInput(
    train_features=X_train_reg,
    train_target=y_train_reg,
    test_features=X_test_reg,
    test_target=y_test_reg,
)

regression_result = train_linear_regression(
    regression_data
)

test_mse = mean_squared_error(
    y_test_reg,
    regression_result.predictions,
)

print("First MSE:")
print(regression_result.loss_history[0])

print("Final training MSE:")
print(regression_result.loss_history[-1])

print("Test MSE:")
print(test_mse)

print("Weights:")
print(regression_result.weights)

print("Bias:")
print(regression_result.bias)


print("\n=== OWN LOGISTIC REGRESSION ===")

german_df = pd.read_csv(
    "german.csv",
    sep=";",
)

X_classification = german_df.drop(
    columns=["Creditability"]
)

y_classification = german_df[
    "Creditability"
]

(
    X_train_cls,
    X_test_cls,
    y_train_cls,
    y_test_cls,
) = train_test_split(
    X_classification,
    y_classification,
    test_size=0.2,
    random_state=42,
    stratify=y_classification,
)

classification_data = ClassificationInput(
    train_features=X_train_cls,
    train_target=y_train_cls,
    test_features=X_test_cls,
    test_target=y_test_cls,
)

classification_result = (
    train_logistic_classifier(
        classification_data
    )
)

own_logistic_metrics = classification_metrics(
    y_test_cls,
    classification_result.probabilities,
    classification_result.predictions,
)

print("First BCE:")
print(
    classification_result.loss_history[0]
)

print("Final BCE:")
print(
    classification_result.loss_history[-1]
)

for name, value in own_logistic_metrics.items():
    print(f"{name}: {value}")


print("\n=== CLASSICAL MODELS ===")

comparison = compare_classical_models(
    classification_data
)

logistic_metrics = {
    "roc_auc": comparison.logistic_regression.roc_auc,
    "accuracy": comparison.logistic_regression.accuracy,
    "precision": comparison.logistic_regression.precision,
    "recall": comparison.logistic_regression.recall,
    "f1": comparison.logistic_regression.f1,
}

tree_metrics = {
    "roc_auc": comparison.decision_tree.roc_auc,
    "accuracy": comparison.decision_tree.accuracy,
    "precision": comparison.decision_tree.precision,
    "recall": comparison.decision_tree.recall,
    "f1": comparison.decision_tree.f1,
}

knn_metrics = {
    "roc_auc": comparison.knn.roc_auc,
    "accuracy": comparison.knn.accuracy,
    "precision": comparison.knn.precision,
    "recall": comparison.knn.recall,
    "f1": comparison.knn.f1,
}


print("\nLogistic Regression:")
for name, value in logistic_metrics.items():
    print(f"{name}: {value}")

print("\nDecision Tree:")
for name, value in tree_metrics.items():
    print(f"{name}: {value}")

print("\nKNN:")
for name, value in knn_metrics.items():
    print(f"{name}: {value}")


experiment_results = {
    "linear_regression": {
        "test_mse": float(test_mse),
    },
    "classification": {
        "own_logistic": own_logistic_metrics,
        "logistic_regression": logistic_metrics,
        "decision_tree": tree_metrics,
        "knn": knn_metrics,
    },
}


with open(
    "experiment_results.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        experiment_results,
        file,
        indent=4,
    )


print(
    "\nexperiment_results.json created successfully."
)