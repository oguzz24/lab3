import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)


def train_linear_regression(data: RegressionInput) -> RegressionResult:

    X_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=float).reshape(-1)
    X_test = np.asarray(data.test_features, dtype=float)

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)

    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std == 0, 1.0, std)

    X_train_scaled = (X_train - mean) / std

    weights = np.zeros(X_train_scaled.shape[1], dtype=float)
    bias = 0.0

    loss_history = []
    n_samples = X_train_scaled.shape[0]

    for _ in range(data.epochs):
        predictions = X_train_scaled @ weights + bias
        errors = predictions - y_train

        mse = np.mean(errors ** 2)
        loss_history.append(float(mse))

        weight_gradient = (2.0 / n_samples) * (
            X_train_scaled.T @ errors
        )
        bias_gradient = (2.0 / n_samples) * np.sum(errors)

        weights -= data.learning_rate * weight_gradient
        bias -= data.learning_rate * bias_gradient

    original_weights = weights / std
    original_bias = bias - np.sum((mean / std) * weights)

    test_predictions = X_test @ original_weights + original_bias

    return RegressionResult(
        weights=original_weights,
        bias=float(original_bias),
        loss_history=loss_history,
        predictions=test_predictions,
    )


def train_logistic_classifier(
    data: ClassificationInput,
) -> ClassificationResult:

    X_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=float).reshape(-1)
    X_test = np.asarray(data.test_features, dtype=float)

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)

    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std == 0, 1.0, std)

    X_train_scaled = (X_train - mean) / std

    weights = np.zeros(X_train_scaled.shape[1], dtype=float)
    bias = 0.0

    loss_history = []
    n_samples = X_train_scaled.shape[0]

    for _ in range(data.epochs):
        logits = X_train_scaled @ weights + bias
        logits = np.clip(logits, -500, 500)

        probabilities = 1.0 / (1.0 + np.exp(-logits))

        epsilon = 1e-12

        loss = -np.mean(
            y_train * np.log(probabilities + epsilon)
            + (1.0 - y_train)
            * np.log(1.0 - probabilities + epsilon)
        )

        loss_history.append(float(loss))

        errors = probabilities - y_train

        weight_gradient = (
            X_train_scaled.T @ errors
        ) / n_samples

        bias_gradient = np.mean(errors)

        weights -= data.learning_rate * weight_gradient
        bias -= data.learning_rate * bias_gradient

    original_weights = weights / std
    original_bias = bias - np.sum((mean / std) * weights)

    test_logits = X_test @ original_weights + original_bias
    test_logits = np.clip(test_logits, -500, 500)

    test_probabilities = 1.0 / (
        1.0 + np.exp(-test_logits)
    )

    test_predictions = (
        test_probabilities >= 0.5
    ).astype(int)

    return ClassificationResult(
        weights=original_weights,
        bias=float(original_bias),
        loss_history=loss_history,
        probabilities=test_probabilities,
        predictions=test_predictions,
    )


def _calculate_metrics(
    y_true,
    probabilities,
    predictions,
) -> ClassificationMetrics:
    return ClassificationMetrics(
        roc_auc=float(
            roc_auc_score(y_true, probabilities)
        ),
        accuracy=float(
            accuracy_score(y_true, predictions)
        ),
        precision=float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        recall=float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        f1=float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
    )


def compare_classical_models(
    data: ClassificationInput,
) -> BaselineComparisonResult:

    X_train = np.asarray(
        data.train_features,
        dtype=float,
    )
    y_train = np.asarray(
        data.train_target,
        dtype=int,
    ).reshape(-1)

    X_test = np.asarray(
        data.test_features,
        dtype=float,
    )
    y_test = np.asarray(
        data.test_target,
        dtype=int,
    ).reshape(-1)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logistic_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    logistic_model.fit(
        X_train_scaled,
        y_train,
    )

    logistic_predictions = logistic_model.predict(
        X_test_scaled
    )

    logistic_probabilities = (
        logistic_model.predict_proba(
            X_test_scaled
        )[:, 1]
    )

    logistic_metrics = _calculate_metrics(
        y_test,
        logistic_probabilities,
        logistic_predictions,
    )

    tree_model = DecisionTreeClassifier(
        random_state=42
    )

    tree_model.fit(
        X_train_scaled,
        y_train,
    )

    tree_predictions = tree_model.predict(
        X_test_scaled
    )

    tree_probabilities = tree_model.predict_proba(
        X_test_scaled
    )[:, 1]

    tree_metrics = _calculate_metrics(
        y_test,
        tree_probabilities,
        tree_predictions,
    )

    knn_model = KNeighborsClassifier()

    knn_model.fit(
        X_train_scaled,
        y_train,
    )

    knn_predictions = knn_model.predict(
        X_test_scaled
    )

    knn_probabilities = knn_model.predict_proba(
        X_test_scaled
    )[:, 1]

    knn_metrics = _calculate_metrics(
        y_test,
        knn_probabilities,
        knn_predictions,
    )

    return BaselineComparisonResult(
        logistic_regression=logistic_metrics,
        decision_tree=tree_metrics,
        knn=knn_metrics,
    )