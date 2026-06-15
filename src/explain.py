"""
src/explain.py
==============
Model explainability using SHAP (SHapley Additive exPlanations).

Important disclaimer:
---------------------
SHAP values describe how each feature contributed to a SPECIFIC MODEL's
prediction. They do NOT imply causal relationships between variables and
% Silica Concentrate in the real process. Operational decisions should not be
made based solely on these explanations without domain expertise validation.
"""

from pathlib import Path
from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence

try:
    from .config import CFG
except ImportError:
    from config import CFG

# SHAP import — required
try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
    print("[explain] shap not installed — explainability functions unavailable.")


# ---------------------------------------------------------------------------
# 1. Build SHAP explainer
# ---------------------------------------------------------------------------

def build_explainer(model, X_background: pd.DataFrame, model_name: str = "") -> "shap.Explainer":
    """Create an appropriate SHAP explainer for the given model.

    Tree-based models (RF, XGBoost, LightGBM) use TreeExplainer for speed.
    Linear models use LinearExplainer.
    All others fall back to the general KernelExplainer (slower).

    Parameters
    ----------
    model : fitted estimator
    X_background : background dataset for the explainer (training set or a sample)
    model_name : optional name for logging

    Returns
    -------
    shap.Explainer instance (not yet called — call with X_explain)
    """
    if not _SHAP_AVAILABLE:
        raise ImportError("Install shap: pip install shap")

    model_class = type(model).__name__
    print(f"[explain] Building SHAP explainer for: {model_class}")

    tree_models = {"RandomForestRegressor", "XGBRegressor", "LGBMRegressor",
                   "GradientBoostingRegressor", "ExtraTreesRegressor"}
    linear_models = {"Ridge", "Lasso", "LinearRegression", "ElasticNet"}

    if model_class in tree_models:
        explainer = shap.TreeExplainer(model)
    elif model_class in linear_models:
        explainer = shap.LinearExplainer(model, X_background)
    else:
        # KernelExplainer is model-agnostic but slow — sample background
        background = shap.sample(X_background, min(100, len(X_background)))
        explainer = shap.KernelExplainer(model.predict, background)

    return explainer


# ---------------------------------------------------------------------------
# 2. Compute SHAP values
# ---------------------------------------------------------------------------

def compute_shap_values(
    explainer,
    X_explain: pd.DataFrame,
    sample_size: int | None = None,
    cfg: dict | None = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Compute SHAP values for a subset of observations.

    Parameters
    ----------
    explainer : fitted SHAP explainer
    X_explain : feature DataFrame to explain
    sample_size : number of rows to sample (None = use cfg default)
    cfg : project config

    Returns
    -------
    shap_values : np.ndarray, shape (n_samples, n_features)
    X_sample    : pd.DataFrame, the sampled rows (same order as shap_values)
    """
    cfg = cfg or CFG
    if sample_size is None:
        sample_size = cfg["explainability"]["shap_sample_size"]

    if len(X_explain) > sample_size:
        X_sample = X_explain.sample(n=sample_size, random_state=cfg["models"]["random_state"])
    else:
        X_sample = X_explain.copy()

    sv = explainer(X_sample)
    # shap.Explanation object → numpy array
    if hasattr(sv, "values"):
        shap_array = sv.values
    else:
        shap_array = np.array(sv)

    return shap_array, X_sample


def align_feature_frame(
    X: pd.DataFrame,
    expected_columns: list[str],
    dataset_name: str = "features",
) -> pd.DataFrame:
    """Ensure a feature matrix matches the exact training feature schema."""
    missing = [col for col in expected_columns if col not in X.columns]
    extra = [col for col in X.columns if col not in expected_columns]

    if missing or extra:
        raise ValueError(
            f"{dataset_name} does not match the training feature schema. "
            f"Missing columns: {missing[:10]}{'...' if len(missing) > 10 else ''}. "
            f"Unexpected columns: {extra[:10]}{'...' if len(extra) > 10 else ''}."
        )

    return X.loc[:, expected_columns].copy()


# ---------------------------------------------------------------------------
# 3. Global feature importance
# ---------------------------------------------------------------------------

def mean_absolute_shap(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Compute mean |SHAP| per feature as a global importance measure."""
    importance = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": importance})
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def get_model_feature_importance(model: Any, feature_names: list[str]) -> pd.DataFrame:
    """Return a comparable feature-importance table when the model exposes one."""
    if hasattr(model, "feature_importances_"):
        scores = np.asarray(model.feature_importances_)
        metric_name = "importance"
    elif hasattr(model, "coef_"):
        scores = np.abs(np.ravel(model.coef_))
        metric_name = "abs_coefficient"
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    df = pd.DataFrame({"feature": feature_names, metric_name: scores})
    return df.sort_values(metric_name, ascending=False).reset_index(drop=True)


def infer_shap_direction(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    feature_name: str,
) -> str:
    """Summarise whether higher feature values are associated with higher or lower predictions."""
    feature_idx = list(X_sample.columns).index(feature_name)
    feature_values = X_sample.iloc[:, feature_idx].to_numpy()
    shap_feature = shap_values[:, feature_idx]

    if np.allclose(np.std(feature_values), 0) or np.allclose(np.std(shap_feature), 0):
        return "No strong directional pattern observed"

    corr = np.corrcoef(feature_values, shap_feature)[0, 1]
    if np.isnan(corr):
        return "No strong directional pattern observed"
    if corr >= 0.15:
        return "Higher values are linked to higher predicted silica"
    if corr <= -0.15:
        return "Higher values are linked to lower predicted silica"
    return "Mixed or non-linear direction"


# ---------------------------------------------------------------------------
# 4. Summary plots
# ---------------------------------------------------------------------------

def plot_shap_summary(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    cfg: dict | None = None,
    save: bool = True,
    title: str = "SHAP Summary Plot",
) -> plt.Figure:
    """Generate and optionally save a SHAP beeswarm summary plot.

    The beeswarm plot shows the distribution of SHAP values per feature,
    coloured by feature value — useful for understanding direction of effect.
    """
    cfg = cfg or CFG
    top_n = cfg["explainability"]["top_features"]
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_sample,
        max_display=top_n,
        show=False,
        plot_size=None,
    )
    plt.title(title, fontsize=13)
    plt.tight_layout()

    if save:
        fig_path = Path(cfg["paths"]["reports_figures"]) / "shap_summary.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"[explain] Saved: {fig_path}")

    return plt.gcf()


def plot_shap_bar(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    cfg: dict | None = None,
    save: bool = True,
    title: str = "SHAP Global Feature Importance",
) -> plt.Figure:
    """Generate and optionally save a SHAP mean |SHAP| bar chart."""
    cfg = cfg or CFG
    top_n = cfg["explainability"]["top_features"]
    imp = mean_absolute_shap(shap_values, list(X_sample.columns))
    imp_top = imp.head(top_n)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(imp_top["feature"][::-1], imp_top["mean_abs_shap"][::-1], color="steelblue")
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title(title, fontsize=13)
    plt.tight_layout()

    if save:
        fig_path = Path(cfg["paths"]["reports_figures"]) / "shap_bar.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"[explain] Saved: {fig_path}")

    return fig


# ---------------------------------------------------------------------------
# 5. Local (single-observation) SHAP waterfall
# ---------------------------------------------------------------------------

def plot_shap_waterfall(
    explainer,
    X_single: pd.DataFrame,
    obs_idx: int = 0,
    cfg: dict | None = None,
    save: bool = True,
    title_suffix: str = "",
) -> None:
    """Plot a SHAP waterfall for a single observation.

    The waterfall shows which features push the prediction above or below
    the model's expected value, in order of their contribution.

    Parameters
    ----------
    explainer : fitted SHAP explainer
    X_single  : DataFrame with exactly the row(s) to explain
    obs_idx   : which row within X_single to plot (default 0)
    """
    cfg = cfg or CFG
    sv = explainer(X_single)
    shap.plots.waterfall(sv[obs_idx], show=False)
    plt.title(f"SHAP Waterfall{' — ' + title_suffix if title_suffix else ''}", fontsize=12)
    plt.tight_layout()

    if save:
        fname = f"shap_waterfall_{obs_idx}{('_' + title_suffix.replace(' ', '_')) if title_suffix else ''}.png"
        fig_path = Path(cfg["paths"]["reports_figures"]) / fname
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"[explain] Saved: {fig_path}")

    plt.show()


def select_explainability_cases(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """Pick low-error, high-error, and representative cases from a prediction set."""
    df = pd.DataFrame({
        "y_true": np.asarray(y_true),
        "y_pred": np.asarray(y_pred),
    }, index=y_true.index)
    df["error"] = df["y_true"] - df["y_pred"]
    df["abs_error"] = df["error"].abs()
    df["distance_to_median_target"] = (df["y_true"] - df["y_true"].median()).abs()

    return {
        "low_error": df["abs_error"].idxmin(),
        "high_error": df["abs_error"].idxmax(),
        "representative": df["distance_to_median_target"].idxmin(),
    }


def build_local_explanations_table(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    case_indices: dict[str, Any],
    top_n: int = 5,
) -> pd.DataFrame:
    """Build a local-explanation summary table for selected cases."""
    rows = []
    index_lookup = {idx: pos for pos, idx in enumerate(X_sample.index)}

    for case_label, case_index in case_indices.items():
        pos = index_lookup[case_index]
        contrib = pd.Series(shap_values[pos], index=X_sample.columns).sort_values()
        top_negative = contrib.head(top_n)
        top_positive = contrib.tail(top_n).sort_values(ascending=False)

        rows.append({
            "case_label": case_label,
            "index": case_index,
            "y_true": float(y_true.loc[case_index]),
            "y_pred": float(y_pred[pos]),
            "error": float(y_true.loc[case_index] - y_pred[pos]),
            "abs_error": float(abs(y_true.loc[case_index] - y_pred[pos])),
            "top_positive_drivers": "; ".join(
                f"{feat} ({val:.4f})" for feat, val in top_positive.items()
            ),
            "top_negative_drivers": "; ".join(
                f"{feat} ({val:.4f})" for feat, val in top_negative.items()
            ),
        })

    return pd.DataFrame(rows)


def compute_partial_dependence_curve(
    model: Any,
    X: pd.DataFrame,
    feature_name: str,
    grid_resolution: int = 30,
) -> pd.DataFrame:
    """Compute a one-feature partial dependence curve."""
    pd_result = partial_dependence(
        model,
        X,
        features=[feature_name],
        grid_resolution=grid_resolution,
        kind="average",
    )
    return pd.DataFrame({
        "feature_value": pd_result["grid_values"][0],
        "predicted_response": pd_result["average"][0],
    })
