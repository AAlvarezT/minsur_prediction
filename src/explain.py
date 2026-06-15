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
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


# ---------------------------------------------------------------------------
# 3. Global feature importance
# ---------------------------------------------------------------------------

def mean_absolute_shap(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Compute mean |SHAP| per feature as a global importance measure."""
    importance = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": importance})
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


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
        fig_path = Path(cfg["paths"]["reports_figures"]) / "shap_bar_importance.png"
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
