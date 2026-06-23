"""
FIFA World Cup 2026 - Phase 4: Model Training
===============================================
Trains three models and picks the best one:
  1. Logistic Regression (baseline)
  2. Random Forest
  3. XGBoost (main model)

"""

import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("  [!] XGBoost not installed. Run: pip install xgboost")
    print("      Falling back to GradientBoosting.")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

warnings.filterwarnings("ignore")

PROCESSED  = Path("data/processed")
MODELS_DIR = Path("data/models")
EDA_DIR    = Path("data/eda")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
EDA_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP  = {-1: "Away win", 0: "Draw", 1: "Home win"}
META_COLS  = ["match_id", "date", "home_team", "away_team",
              "home_score", "away_score", "total_goals", "result",
              "year", "is_2026"]

# Max features to keep. Roughly: don't exceed (n_samples / 10) features as
# a conservative rule of thumb to limit overfitting risk. With ~104 combined
# rows, 8-10 is reasonable; revisit upward as more matches/years are added.
MAX_FEATURES = 8

# A feature must be non-null for at least this fraction of training rows to
# be eligible for selection. This is the key fix for the multi-year dataset
# - without it, 2026-only features (caps, height, rank) would dominate
# selection on correlation alone while only ever being measured on the
# 2026 subset.
MIN_COVERAGE = 0.90

KNOWN_DUPLICATE_GROUPS = [
    ["diff_fifa_rank", "diff_fifa_points"],
    ["diff_total_caps", "diff_avg_caps"],
    ["diff_total_national_goals", "diff_avg_national_goals"],
    ["diff_total_goals_scored", "diff_avg_goals_scored"],
    ["diff_top_league_count", "diff_top_league_ratio"],
]


def select_top_features(train: pd.DataFrame, max_features: int = MAX_FEATURES,
                          min_coverage: float = MIN_COVERAGE) -> list:
    """
    Pick the top-K diff_ features by |correlation| with result, restricted
    to features with at least `min_coverage` non-null rows, after
    collapsing known near-duplicate pairs down to one representative each.
    Sparse-but-strong features that get excluded are printed separately so
    they're visible rather than silently dropped.
    """
    diff_cols = [c for c in train.columns if c.startswith("diff_")]
    n_rows = len(train)

    coverage = train[diff_cols].notna().mean()
    eligible_cols = coverage[coverage >= min_coverage].index.tolist()
    sparse_cols = coverage[coverage < min_coverage].index.tolist()

    corr_all = train[diff_cols + ["result"]].corr()["result"].drop("result").dropna()
    ranked_eligible = corr_all[corr_all.index.isin(eligible_cols)].abs().sort_values(ascending=False)

    drop_set = set()
    for group in KNOWN_DUPLICATE_GROUPS:
        present = [c for c in group if c in ranked_eligible.index]
        for loser in present[1:]:
            drop_set.add(loser)
    ranked_eligible = ranked_eligible.drop(index=[c for c in drop_set if c in ranked_eligible.index])

    selected = ranked_eligible.head(max_features).index.tolist()

    print(f"\n  Feature selection (sample size = {n_rows}, min coverage = {min_coverage:.0%}):")
    print(f"  Selected {len(selected)} of {len(diff_cols)} diff_ features "
          f"(eligible: {len(eligible_cols)}, dropped {len(drop_set)} duplicate(s)):")
    for c in selected:
        cov = coverage[c]
        print(f"    {c:<35} |corr| = {ranked_eligible[c]:.3f}   coverage = {cov:.0%}")

    if sparse_cols:
        sparse_ranked = corr_all[corr_all.index.isin(sparse_cols)].abs().sort_values(ascending=False)
        print(f"\n  Excluded for low coverage (< {min_coverage:.0%} of rows) - "
              f"these may still be useful once more 2026 matches are played:")
        for c in sparse_ranked.head(5).index:
            print(f"    {c:<35} |corr| = {sparse_ranked[c]:.3f}   coverage = {coverage[c]:.0%}  (EXCLUDED)")

    return selected


def prepare_features(train: pd.DataFrame, feature_cols: list):
    """Select the given feature columns and encode result as 0/1/2 for sklearn."""
    X = train[feature_cols].fillna(train[feature_cols].median())

    y = train["result"].map({1: 1, 0: 0, -1: 2})
    unique_classes = sorted(y.unique())
    class_remap = {old: new for new, old in enumerate(unique_classes)}
    y = y.map(class_remap)

    print(f"\n  Features used: {len(feature_cols)}")
    print(f"  Samples:       {len(X)}")
    print(f"  Classes:       {dict(y.value_counts())}")
    if "year" in train.columns:
        print(f"  By year:       {train['year'].value_counts().to_dict()}")
    return X, y


def get_models():
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                max_iter=1000, C=0.5,
                class_weight="balanced", random_state=42
            )),
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=4, min_samples_leaf=3,
            class_weight="balanced", random_state=42
        ),

        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            subsample=0.8, eval_metric="mlogloss",
            random_state=42
            ) if HAS_XGB else
            GradientBoostingClassifier(
                n_estimators=150, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=42
            ),
    }
    return models


def evaluate_models(X, y, models: dict) -> dict:
    """K-fold cross-validation, K scaled down if the smallest class is tiny."""
    n_splits = min(5, y.value_counts().min())
    n_splits = max(n_splits, 2)
    print(f"\n  Cross-validation results ({n_splits}-fold):\n")
    print(f"  {'Model':<25} {'Accuracy':>14} {'Log-Loss':>14}")
    print("  " + "-" * 56)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {}

    for name, model in models.items():
        acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        ll  = cross_val_score(model, X, y, cv=cv, scoring="neg_log_loss")
        results[name] = {"acc_mean": acc.mean(), "acc_std": acc.std(),
                         "ll_mean": -ll.mean(),  "ll_std":  ll.std()}
        print(f"  {name:<25} {acc.mean():.3f} +/- {acc.std():.3f}   {-ll.mean():.3f} +/- {ll.std():.3f}")

    chance = 1 / y.nunique()
    print(f"\n  (Chance-level accuracy for {y.nunique()} classes: {chance:.3f})")

    best = max(results, key=lambda k: (round(results[k]["acc_mean"], 3), -results[k]["ll_mean"]))
    print(f"  Best model (by accuracy, log-loss as tiebreaker): {best}")

    if all(r["acc_mean"] <= chance for r in results.values()):
        print("\n  [!] WARNING: every model is at or below chance-level accuracy.")
        print("      Treat all outputs as provisional until more matches/years")
        print("      improve the signal.")

    return results, best


def evaluate_by_year(X, y, train: pd.DataFrame, model, model_name: str):
    """
    Audits whether the model performs suspiciously differently on 2026 rows
    vs historical rows. This is the direct follow-up to the missingness-
    leakage concern raised in Phase 2/3: if accuracy is wildly different by
    year in a way that doesn't track football reality (e.g. perfect on one
    year, terrible on another), the model may have partly learned to
    distinguish years by missingness pattern rather than genuine signal.
    """
    if "year" not in train.columns or train["year"].nunique() < 2:
        print("\n  Only one year present - skipping by-year audit.")
        return

    print("\n  By-year performance audit (missingness-leakage check):")
    model.fit(X, y)
    preds = model.predict(X)
    correct = (preds == y.values)

    for yr in sorted(train["year"].unique()):
        mask = (train["year"] == yr).values
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean()
        print(f"    {yr}: {mask.sum()} rows, in-sample accuracy = {acc:.3f}")
    print("    (In-sample, not cross-validated - this is a coarse diagnostic,")
    print("     not a performance claim. Look for years that stand out sharply.)")


def train_final_model(X, y, model, model_name: str):
    """
    Train on all data, optionally wrapped in a probability calibrator.

    IMPORTANT - calibration can do more harm than good at small sample
    sizes. Confirmed directly on this pipeline's real data (~105 training
    rows, smallest class ~27-29 samples): CalibratedClassifierCV with
    method="sigmoid" collapsed a well-behaved model (predictions tracking
    the real ~47/28/26 home/draw/away split) into a degenerate "almost
    always predict Draw" strategy (84/105 draw predictions), because each
    calibration fold only saw ~9-10 examples per class - far too few for
    sigmoid (Platt) scaling to learn reliable per-class probability curves.
    The calibrated model still scored ~51% CV accuracy, masking the
    problem, because defaulting to the most common-ish class is a
    legitimate way to rack up correct guesses without making real
    predictions.

    Below MIN_ROWS_FOR_CALIBRATION, this function skips calibration
    entirely and returns the raw fitted model. Above that threshold,
    calibration is applied as before. Revisit this threshold as the
    training set grows - recheck with the same uncalibrated-vs-calibrated
    prediction-distribution comparison before re-enabling calibration at
    a given sample size.
    """
    MIN_ROWS_FOR_CALIBRATION = 300  # conservative; recheck empirically before lowering

    if len(y) < MIN_ROWS_FOR_CALIBRATION:
        print(f"\n  [!] Skipping probability calibration - only {len(y)} training rows "
              f"(< {MIN_ROWS_FOR_CALIBRATION}).")
        print("      Calibration with this little data has been confirmed to distort")
        print("      predictions toward the majority class rather than improve them.")
        print("      Using the raw model's predict_proba() directly instead.")
        model.fit(X, y)
        final = model
    else:
        n_splits = min(3, y.value_counts().min())
        n_splits = max(n_splits, 2)
        final = CalibratedClassifierCV(model, cv=n_splits, method="sigmoid")
        final.fit(X, y)

    with open(MODELS_DIR / "final_model.pkl", "wb") as f:
        pickle.dump({"model": final, "model_name": model_name,
                     "feature_cols": list(X.columns)}, f)

    print(f"\n  Trained: {model_name}"
          f"{' (calibrated)' if len(y) >= MIN_ROWS_FOR_CALIBRATION else ' (uncalibrated)'}")
    print(f"  Saved -> data/models/final_model.pkl")
    return final


def plot_calibration(model, X, y):
    """Check: are our 70% confident predictions right ~70% of the time?"""
    try:
        proba = model.predict_proba(X)
        fig, ax = plt.subplots(figsize=(7, 6))
        class_names = ["Away win", "Draw", "Home win"]
        n_bins = min(5, max(2, len(X) // 15))
        for i, (name, color) in enumerate(zip(class_names,
                                              ["#e74c3c", "#f39c12", "#27ae60"])):
            if i >= proba.shape[1]:
                continue
            y_bin = (y == i).astype(int)
            prob_true, prob_pred = calibration_curve(y_bin, proba[:, i], n_bins=n_bins)
            ax.plot(prob_pred, prob_true, marker="o", label=name, color=color)

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect calibration")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Actual Fraction")
        ax.set_title("Model Calibration Curve\n(closer to diagonal = better)")
        ax.legend()
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(EDA_DIR / "8_calibration_curve.png", dpi=150)
        plt.close()
        print("   Saved: data/eda/8_calibration_curve.png")
    except Exception as e:
        print(f"   [!] Calibration plot skipped: {e}")


def plot_shap_importance(model, X, feature_cols, model_name: str = ""):
    """SHAP values show *why* each prediction was made.
    TreeExplainer only supports tree-based models (Random Forest, XGBoost,
    GradientBoosting) - if the best model turned out to be Logistic
    Regression, this is skipped with a clear explanation rather than
    failing with a confusing library error."""
    if not HAS_SHAP:
        print("   [!] SHAP not installed (pip install shap) - skipping SHAP plot.")
        return

    is_tree_model = any(name in model_name for name in ("Random Forest", "XGBoost", "GradientBoosting"))
    if not is_tree_model:
        print(f"   [!] Best model is '{model_name}' (not tree-based) - SHAP TreeExplainer")
        print("       doesn't apply here. Skipping SHAP plot for this run; use")
        print("       coefficient inspection instead, or a model-agnostic SHAP")
        print("       explainer (shap.LinearExplainer / shap.KernelExplainer) if needed.")
        return

    try:
        if hasattr(model, "calibrated_classifiers_"):
            base = model.calibrated_classifiers_[0].estimator
        else:
            base = model
        explainer = shap.TreeExplainer(base)
        shap_values = explainer.shap_values(X)

        class_label_order = ["Away win", "Draw", "Home win"]

        plt.figure(figsize=(10, 7))
        if isinstance(shap_values, list):
            shap.summary_plot(
                shap_values, X, feature_names=feature_cols, show=False,
                max_display=15, plot_type="bar",
                class_names=class_label_order[:len(shap_values)],
            )
        else:
            shap.summary_plot(
                shap_values, X, feature_names=feature_cols, show=False,
                max_display=15, plot_type="bar",
            )
        plt.title("SHAP Feature Importance - Match Result Prediction")
        plt.tight_layout()
        plt.savefig(EDA_DIR / "9_shap_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("   Saved: data/eda/9_shap_importance.png")
        print("   (Legend may show 'Class 0/1/2' instead of names on some SHAP")
        print("    versions - mapping is Class 0=Away win, 1=Draw, 2=Home win.)")
    except Exception as e:
        print(f"   [!] SHAP plot skipped: {e}")


def train_poisson_model(train: pd.DataFrame, feature_cols: list):
    """
    Separate Poisson regressors for home goals and away goals.
    """
    completed = train[train["home_score"].notna()].copy()
    if len(completed) < 5:
        print("  [!] Too few matches for Poisson model yet.")
        return None, None

    X_p = completed[feature_cols].fillna(completed[feature_cols].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_p)

    home_model = PoissonRegressor(alpha=1.0, max_iter=300)
    away_model = PoissonRegressor(alpha=1.0, max_iter=300)

    home_model.fit(X_scaled, completed["home_score"].astype(int))
    away_model.fit(X_scaled, completed["away_score"].astype(int))

    home_pred = home_model.predict(X_scaled)
    away_pred = away_model.predict(X_scaled)

    home_mae = np.abs(home_pred - completed["home_score"]).mean()
    away_mae = np.abs(away_pred - completed["away_score"]).mean()
    print(f"\n  Poisson model (score prediction, all years combined):")
    print(f"    Home goals MAE: {home_mae:.2f}")
    print(f"    Away goals MAE: {away_mae:.2f}")
    print(f"    (MAE is computed on training data itself - optimistic; "
          f"true out-of-sample error will be higher)")

    with open(MODELS_DIR / "poisson_model.pkl", "wb") as f:
        pickle.dump({"home_model": home_model, "away_model": away_model,
                     "scaler": scaler, "feature_cols": feature_cols}, f)
    print("  Saved -> data/models/poisson_model.pkl")
    return home_model, away_model


def predict_upcoming(model, feature_cols: list):
    """Apply trained model to upcoming 2026 matches (predict_features.csv
    only ever contains 2026 rows - historical years have no unplayed
    matches by definition).

    Defensive date filter: predict_features.csv is only as fresh as the
    last Phase 2 run. If matches.csv has since been refreshed (more games
    finished) but Phase 2 wasn't re-run, predict_features.csv can still
    contain rows for matches that have actually already been played.
    Dropping anything dated today or earlier catches that case instead of
    silently showing stale "predictions" for matches with real results.
    The correct fix when this fires is still to re-run Phase 2 (and this
    script) - this filter is a safety net, not a substitute for that."""
    predict_path = PROCESSED / "predict_features.csv"
    if not predict_path.exists():
        return

    predict_df = pd.read_csv(predict_path)
    if predict_df.empty:
        print("  No upcoming matches to predict.")
        return

    if "date" in predict_df.columns:
        from datetime import date
        today = date.today()
        predict_df["date"] = pd.to_datetime(predict_df["date"], errors="coerce")
        stale_mask = predict_df["date"].dt.date <= today
        n_stale = stale_mask.sum()
        if n_stale > 0:
            print(f"  [!] Dropped {n_stale} row(s) in predict_features.csv dated today or "
                  f"earlier - these matches have likely already been played.")
            print(f"      Re-run Phase 2 + Phase 4 to refresh predict_features.csv properly.")
        predict_df = predict_df[~stale_mask].copy()
        predict_df["date"] = predict_df["date"].dt.strftime("%Y-%m-%d")

    if predict_df.empty:
        print("  No upcoming matches to predict (all rows were stale or in the past).")
        return

    missing = [c for c in feature_cols if c not in predict_df.columns]
    if missing:
        print(f"  [!] predict_features.csv is missing columns used by the model: {missing}")
        print("      Skipping predictions - re-run Phase 2 to regenerate consistent features.")
        return

    X_pred = predict_df[feature_cols].fillna(0)
    proba  = model.predict_proba(X_pred)

    results = []
    for i, (_, row) in enumerate(predict_df.iterrows()):
        n_classes = proba[i].shape[0]

        if n_classes == 2:
            home_win = round(proba[i][0] * 100, 1)
            draw     = 0.0
            away_win = round(proba[i][1] * 100, 1)
        else:
            home_win = round(proba[i][2] * 100, 1)
            draw     = round(proba[i][1] * 100, 1)
            away_win = round(proba[i][0] * 100, 1)

        best_idx  = np.argmax(proba[i])
        if n_classes == 2:
            predicted = "Home win" if best_idx == 0 else "Away win"
        else:
            predicted = ["Away win", "Draw", "Home win"][best_idx]

        results.append({
            "home_team":  row["home_team"],
            "away_team":  row["away_team"],
            "date":       row["date"],
            "home_win_%": home_win,
            "draw_%":     draw,
            "away_win_%": away_win,
            "predicted":  predicted,
        })

    df_preds = pd.DataFrame(results)
    df_preds.to_csv(PROCESSED / "predictions.csv", index=False)

    print("\n  Upcoming match predictions (2026):")
    print(f"  {'Home':<22} {'Away':<22} {'HW%':>5} {'D%':>5} {'AW%':>5}  Prediction")
    print("  " + "-" * 72)
    for _, r in df_preds.iterrows():
        print(f"  {r['home_team']:<22} {r['away_team']:<22} "
              f"{r['home_win_%']:>5} {r['draw_%']:>5} {r['away_win_%']:>5}  {r['predicted']}")


def run():
    print("=" * 60)
    print("  FIFA World Cup 2026 - Phase 4: Model Training")
    print("=" * 60)

    train = pd.read_csv(PROCESSED / "train_features.csv")
    train = train[train["result"].notna()].copy()
    print(f"\n  Loaded {len(train)} completed matches for training")
    if "year" in train.columns:
        print(f"  By year: {train['year'].value_counts().to_dict()}")

    feature_cols = select_top_features(train, max_features=MAX_FEATURES)
    if not feature_cols:
        raise SystemExit("\n[X] No features passed the coverage threshold - "
                          "lower MIN_COVERAGE or check train_features.csv.")

    X, y = prepare_features(train, feature_cols)

    with open(MODELS_DIR / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)

    models = get_models()
    cv_results, best_name = evaluate_models(X, y, models)

    evaluate_by_year(X, y, train, models[best_name], best_name)

    final_model = train_final_model(X, y, models[best_name], best_name)

    print("\n  Generating evaluation plots...")
    plot_calibration(final_model, X, y)
    plot_shap_importance(final_model, X, feature_cols, model_name=best_name)

    train_poisson_model(train, feature_cols)
    predict_upcoming(final_model, feature_cols)

    with open(MODELS_DIR / "cv_results.json", "w") as f:
        json.dump(cv_results, f, indent=2)

    return final_model, feature_cols


if __name__ == "__main__":
    run()