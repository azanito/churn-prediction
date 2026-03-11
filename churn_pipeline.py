import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, plot_importance


warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


# =========================
# Configuration
# =========================
DATA_PATH = Path("Churn_Modelling.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42
TARGET_COLUMN = "Exited"


def save_and_show_plot(filename: str) -> None:
    """Save the current matplotlib figure and display it."""
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


# =========================
# Data Loading
# =========================
def load_data(file_path: Path) -> pd.DataFrame:
    """Load the churn dataset into a pandas DataFrame."""
    df = pd.read_csv(file_path)
    return df


def display_basic_info(df: pd.DataFrame) -> None:
    """Print high-level information about the dataset."""
    print("\n" + "=" * 80)
    print("DATASET OVERVIEW")
    print("=" * 80)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nData Types:")
    print(df.dtypes)


# =========================
# EDA
# =========================
def plot_target_distribution(df: pd.DataFrame) -> None:
    """Plot and print the target class distribution."""
    class_counts = df[TARGET_COLUMN].value_counts().sort_index()
    class_percentages = df[TARGET_COLUMN].value_counts(normalize=True).sort_index() * 100

    print("\n" + "=" * 80)
    print("TARGET VARIABLE DISTRIBUTION")
    print("=" * 80)
    print("Class Counts:")
    print(class_counts)
    print("\nClass Percentages:")
    print(class_percentages.round(2).astype(str) + "%")

    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=df, x=TARGET_COLUMN, palette="Blues")
    plt.title("Target Variable Distribution (Exited)")
    plt.xlabel("Exited")
    plt.ylabel("Count")
    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            f"{height}",
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    save_and_show_plot("target_distribution.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Plot the correlation matrix for numerical features."""
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Matrix Heatmap")
    save_and_show_plot("correlation_heatmap.png")


# =========================
# Preprocessing
# =========================
def preprocess_data(df: pd.DataFrame):
    """
    Encode categorical features with LabelEncoder and split the dataset.

    Identifier-like columns are removed before modeling because they do not carry
    stable predictive signal for churn and can add noise.
    """
    model_df = df.copy()

    columns_to_drop = ["RowNumber", "CustomerId"]
    model_df = model_df.drop(columns=columns_to_drop)

    categorical_columns = model_df.select_dtypes(include=["object"]).columns.tolist()

    print("\n" + "=" * 80)
    print("PREPROCESSING")
    print("=" * 80)
    print(f"Categorical Columns: {categorical_columns}")
    print(f"Dropped Identifier Columns: {columns_to_drop}")

    label_encoders = {}
    for column in categorical_columns:
        encoder = LabelEncoder()
        model_df[column] = encoder.fit_transform(model_df[column])
        label_encoders[column] = encoder

    X = model_df.drop(columns=[TARGET_COLUMN])
    y = model_df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Feature Matrix Shape: {X.shape}")
    print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")

    return X, y, X_train, X_test, y_train, y_test, label_encoders


# =========================
# Evaluation
# =========================
def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> None:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("\n" + "=" * 70)
    print(f"{model_name.upper()} PERFORMANCE")
    print("=" * 70)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    print("\nExample Prediction:")

    sample_index = 0
    prob = y_proba[sample_index]
    pred_class = y_pred[sample_index]

    print(f"Predicted churn probability: {prob:.4f}")

    if pred_class == 1:
        print("Predicted class: Customer WILL churn")
    else:
        print("Predicted class: Customer will NOT churn")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


# =========================
# Modeling
# =========================
def train_baseline_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Train a simple baseline Random Forest model."""
    baseline_model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=1,
    )
    baseline_model.fit(X_train, y_train)
    return baseline_model


def get_scale_pos_weight(y: pd.Series) -> float:
    """Calculate class imbalance ratio for XGBoost."""
    negative_count = (y == 0).sum()
    positive_count = (y == 1).sum()
    return negative_count / positive_count


def train_tuned_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    """Train an XGBoost model with GridSearchCV hyperparameter tuning."""
    class_percentages = y_train.value_counts(normalize=True) * 100
    minority_percentage = class_percentages.min()
    imbalance_exists = minority_percentage < 40
    scale_pos_weight = get_scale_pos_weight(y_train) if imbalance_exists else 1.0

    print("\n" + "=" * 80)
    print("CLASS IMBALANCE ANALYSIS")
    print("=" * 80)
    print("Training Set Class Percentages:")
    print(class_percentages.sort_index().round(2).astype(str) + "%")
    print(f"Imbalance Detected: {imbalance_exists}")
    print(f"scale_pos_weight Used: {scale_pos_weight:.2f}")

    xgb_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring="f1",
        cv=3,
        n_jobs=1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    print("\nBest XGBoost Parameters:")
    print(grid_search.best_params_)
    print(f"Best Cross-Validation F1-score: {grid_search.best_score_:.4f}")

    return grid_search


# =========================
# Feature Importance
# =========================
def plot_xgboost_feature_importance(model: XGBClassifier, top_n: int = 10) -> None:
    """Plot the top feature importances from the XGBoost model."""
    feature_importance = pd.Series(
        model.feature_importances_,
        index=model.feature_names_in_,
    ).sort_values(ascending=False)

    print("\n" + "=" * 80)
    print("TOP FEATURES AFFECTING CHURN")
    print("=" * 80)
    print(feature_importance.head(top_n))

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=feature_importance.head(top_n).values,
        y=feature_importance.head(top_n).index,
        palette="viridis",
    )
    plt.title("Top Feature Importances from XGBoost")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    save_and_show_plot("feature_importance_barplot.png")

    plt.figure(figsize=(10, 6))
    plot_importance(model, max_num_features=top_n, importance_type="gain")
    plt.title("XGBoost Feature Importance (Gain)")
    save_and_show_plot("feature_importance_xgboost.png")


# =========================
# Explainability
# =========================
def explain_with_shap(model: XGBClassifier, X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
    """Generate SHAP visualizations and local explanations."""
    print("\n" + "=" * 80)
    print("SHAP EXPLAINABILITY")
    print("=" * 80)

    background = shap.sample(X_train, 200, random_state=RANDOM_STATE)
    explain_data = shap.sample(X_test, 200, random_state=RANDOM_STATE)
    explainer = shap.Explainer(model.predict_proba, background)
    shap_values = explainer(explain_data)

    plt.figure()
    shap.summary_plot(shap_values[:, :, 1], explain_data, show=False)
    plt.title("SHAP Summary Plot")
    save_and_show_plot("shap_summary_plot.png")

    plt.figure()
    shap.summary_plot(shap_values[:, :, 1], explain_data, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance Plot")
    save_and_show_plot("shap_feature_importance_plot.png")

    for idx in [0, 1]:
        print(f"\nGenerating SHAP explanation for test instance index: {idx}")
        shap.plots.waterfall(shap_values[idx, :, 1], show=False)
        plt.title(f"SHAP Waterfall Plot for Prediction {idx + 1}")
        save_and_show_plot(f"shap_waterfall_prediction_{idx + 1}.png")


def main() -> None:
    """Run the full customer churn machine learning pipeline."""
    # Data loading and overview
    df = load_data(DATA_PATH)
    display_basic_info(df)

    # Exploratory data analysis
    plot_target_distribution(df)
    plot_correlation_heatmap(df)

    # Preprocessing and train-test split
    X, y, X_train, X_test, y_train, y_test, _ = preprocess_data(df)

    # Baseline model training and evaluation
    baseline_model = train_baseline_model(X_train, y_train)
    evaluate_model(baseline_model, X_test, y_test, "Baseline Random Forest")

    # Advanced model training with hyperparameter tuning
    grid_search = train_tuned_xgboost(X_train, y_train)
    best_xgb_model = grid_search.best_estimator_
    evaluate_model(best_xgb_model, X_test, y_test, "Tuned XGBoost")

    # Feature importance analysis
    plot_xgboost_feature_importance(best_xgb_model)

    # Model explainability with SHAP
    explain_with_shap(best_xgb_model, X_train, X_test)

    print("\nAll plots have been saved to:", OUTPUT_DIR.resolve())



if __name__ == "__main__":
    main()



