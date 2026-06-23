"""
FIFA World Cup 2026 - Phase 3: Exploratory Data Analysis

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

PROCESSED = Path("data/processed")
EDA_DIR   = Path("data/eda")
EDA_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.size":        11,
})
COLORS = {"win": "#2ecc71", "draw": "#f39c12", "loss": "#e74c3c"}

# Quality proxy used throughout - no overall rating exists in this dataset,
# so we use share of squad playing in a "big 5" league as the stand-in.
# Available for every year (built from club_country).
QUALITY_COL = "top_league_ratio"
QUALITY_LABEL = "Top-League Squad Share"

# The year used for single-year plots (squad strength, age analysis) -
# these compare teams to each other within one tournament, so mixing years
# would be misleading. Historical years still get covered by
# plot_year_comparison() and the correlation/goal-distribution plots.
PRIMARY_YEAR = 2026


# ─────────────────────────────────────────────────────────────────────────────
# EDA 1 — Squad strength comparison across all teams (single year)
# ─────────────────────────────────────────────────────────────────────────────

def plot_squad_strength(tf: pd.DataFrame, year: int = PRIMARY_YEAR):
    """Bar chart of top-league squad share, sorted, with caps overlay.
    Filtered to one year - team_features.csv has one row per (team, year)."""
    tf_year = tf[tf["year"] == year]
    if tf_year.empty:
        print(f"   [!] No team_features rows for year {year} - skipping plot 1.")
        return

    has_caps = "avg_caps" in tf_year.columns and tf_year["avg_caps"].notna().any()
    cols = [QUALITY_COL] + (["avg_caps"] if has_caps else [])
    df = tf_year[cols].sort_values(QUALITY_COL, ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(8, len(df) * 0.28)))
    y = range(len(df))

    ax.barh(y, df[QUALITY_COL], color="#3498db", alpha=0.7, label=QUALITY_LABEL)

    ax.set_yticks(list(y))
    ax.set_yticklabels(df.index, fontsize=9)
    ax.set_xlabel(QUALITY_LABEL + " (0-1)")
    ax.set_title(f"Squad Strength Proxy by Team - FIFA World Cup {year}", fontsize=13, fontweight="bold")
    ax.axvline(df[QUALITY_COL].mean(), color="#888", linestyle="--", linewidth=0.8, label="Average")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "1_squad_strength.png", dpi=150)
    plt.close()
    print("   Saved: data/eda/1_squad_strength.png")


# ─────────────────────────────────────────────────────────────────────────────
# EDA 2 — Does squad quality predict match outcome? (all years combined)
# ─────────────────────────────────────────────────────────────────────────────

def plot_rating_vs_outcome(train: pd.DataFrame):
    """Scatter: diff_<quality> vs result, shows whether stronger teams win.
    Uses all years combined - top_league_ratio is available for every year,
    unlike caps/height which are 2026-only."""
    diff_col = f"diff_{QUALITY_COL}"
    if diff_col not in train.columns:
        print(f"   [!] {diff_col} not found in train_features.csv - skipping plot 2.")
        return

    plot_df = train.dropna(subset=[diff_col, "result"])
    if plot_df.empty:
        print(f"   [!] No rows with non-null {diff_col} - skipping plot 2.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    jitter = np.random.uniform(-0.08, 0.08, len(plot_df))
    color_map = {1: COLORS["win"], 0: COLORS["draw"], -1: COLORS["loss"]}
    for res, label in [(1, "Home win"), (0, "Draw"), (-1, "Away win")]:
        mask = (plot_df["result"] == res).values
        ax.scatter(
            plot_df.loc[mask, diff_col],
            plot_df.loc[mask, "result"] + jitter[mask],
            c=color_map[res], alpha=0.6, s=50, label=label, edgecolors="white", linewidth=0.4
        )
    ax.axvline(0, color="#999", linestyle="--", linewidth=0.8)
    ax.set_xlabel(f"{QUALITY_LABEL} Difference (home minus away)")
    ax.set_ylabel("Result")
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["Away win", "Draw", "Home win"])
    ax.set_title(f"{QUALITY_LABEL} Difference vs Match Result\n(all years combined)")
    ax.legend()

    ax2 = axes[1]
    plot_df = plot_df.copy()
    plot_df["quality_bucket"] = pd.cut(
        plot_df[diff_col],
        bins=[-1.01, -0.3, -0.1, 0.1, 0.3, 1.01],
        labels=["Much weaker", "Weaker", "Similar", "Stronger", "Much stronger"]
    )
    win_rates = plot_df.groupby("quality_bucket", observed=True)["result"].apply(
        lambda x: (x == 1).mean()
    )
    bars = ax2.bar(win_rates.index.astype(str), win_rates.values,
                   color=["#e74c3c","#e67e22","#95a5a6","#27ae60","#1abc9c"][:len(win_rates)])
    ax2.set_ylabel("Home Win Rate")
    ax2.set_xlabel("Home Team Relative Strength")
    ax2.set_title("Home Win Rate by Squad Quality Difference")
    ax2.set_ylim(0, 1)
    for bar, val in zip(bars, win_rates.values):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                 f"{val:.0%}", ha="center", va="bottom", fontsize=9)
    ax2.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig(EDA_DIR / "2_rating_vs_outcome.png", dpi=150)
    plt.close()
    print("   Saved: data/eda/2_rating_vs_outcome.png")


# ─────────────────────────────────────────────────────────────────────────────
# EDA 3 — Feature correlation heatmap (all years, schema-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(train: pd.DataFrame):
    """Heatmap of diff_ features vs result, all years combined."""
    diff_cols = [c for c in train.columns if c.startswith("diff_")]
    corr = train[diff_cols + ["result"]].corr()["result"].drop("result")
    corr = corr.dropna()
    top_feats = corr.abs().sort_values(ascending=False).head(12).index.tolist()

    if len(top_feats) < 2:
        print("   [!] Not enough valid diff_ features for a correlation heatmap - skipping plot 3.")
        return

    corr_matrix = train[top_feats + ["result"]].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.zeros_like(corr_matrix, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, vmin=-1, vmax=1, ax=ax,
        linewidths=0.5, cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Feature Correlation Matrix (top diff_ features, all years)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "3_correlation_heatmap.png", dpi=150)
    plt.close()
    print("   Saved: data/eda/3_correlation_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# EDA 4 — Goal distribution (all years combined)
# ─────────────────────────────────────────────────────────────────────────────

def plot_goal_distribution(train: pd.DataFrame):
    """Histograms of goals scored per match, all years combined."""
    completed = train[train["home_score"].notna()].copy()
    if completed.empty:
        print("   [!] No completed matches yet - skipping plot 4.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for ax, col, label, color in [
        (axes[0], "home_score", "Home Goals",  "#3498db"),
        (axes[1], "away_score", "Away Goals",  "#e74c3c"),
        (axes[2], "total_goals","Total Goals", "#9b59b6"),
    ]:
        vals = completed[col].dropna().astype(int)
        bins = range(0, int(vals.max()) + 2)
        ax.hist(vals, bins=bins, color=color, alpha=0.8, edgecolor="white")
        ax.axvline(vals.mean(), color="#333", linestyle="--", linewidth=1)
        ax.set_xlabel(label)
        ax.set_ylabel("Matches")
        ax.set_title(f"{label} Distribution\nMean: {vals.mean():.2f}")

    years_str = ", ".join(str(y) for y in sorted(completed["year"].unique()))
    plt.suptitle(f"Goal Distribution - World Cup {years_str}", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(EDA_DIR / "4_goal_distribution.png", dpi=150)
    plt.close()
    print("   Saved: data/eda/4_goal_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# EDA 5 — Squad age vs quality (single year)
# ─────────────────────────────────────────────────────────────────────────────

def plot_age_analysis(tf: pd.DataFrame, year: int = PRIMARY_YEAR):
    """Do younger or older squads skew toward bigger clubs? Single-year -
    same reasoning as plot_squad_strength."""
    tf_year = tf[tf["year"] == year]
    if tf_year.empty:
        print(f"   [!] No team_features rows for year {year} - skipping plot 5.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    top_teams = tf_year.nlargest(10, QUALITY_COL).index
    tf_top = tf_year.loc[top_teams].sort_values("avg_age")
    bars = ax.barh(tf_top.index, tf_top["avg_age"],
                   color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(tf_top))))
    ax.axvline(tf_year["avg_age"].mean(), color="#555", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Average Squad Age")
    ax.set_title(f"Avg Squad Age - Top 10 by {QUALITY_LABEL} ({year})")
    for bar, val in zip(bars, tf_top["avg_age"]):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}", va="center", fontsize=9)

    ax2 = axes[1]
    ax2.scatter(tf_year["peak_age_count"], tf_year[QUALITY_COL],
                alpha=0.7, s=60, color="#3498db", edgecolors="white")
    for team in tf_year.nlargest(5, QUALITY_COL).index:
        ax2.annotate(team, (tf_year.loc[team, "peak_age_count"], tf_year.loc[team, QUALITY_COL]),
                     fontsize=8, ha="left", va="bottom")
    if tf_year["peak_age_count"].nunique() > 1:
        m, b = np.polyfit(tf_year["peak_age_count"], tf_year[QUALITY_COL], 1)
        x_line = np.linspace(tf_year["peak_age_count"].min(), tf_year["peak_age_count"].max(), 50)
        ax2.plot(x_line, m * x_line + b, color="#e74c3c", linewidth=1.5, linestyle="--")
    ax2.set_xlabel("Players in Peak Age (24-29)")
    ax2.set_ylabel(QUALITY_LABEL)
    ax2.set_title(f"Peak-Age Players vs {QUALITY_LABEL} ({year})")

    plt.tight_layout()
    plt.savefig(EDA_DIR / "5_age_analysis.png", dpi=150)
    plt.close()
    print("   Saved: data/eda/5_age_analysis.png")


# ─────────────────────────────────────────────────────────────────────────────
# EDA 6 — Feature importance (all years, schema-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(train: pd.DataFrame):
    """Bar chart of feature correlations with result, all years combined."""
    diff_cols = [c for c in train.columns if c.startswith("diff_")]
    if len(train) < 4:
        print("   [!] Too few matches for reliable correlations yet - more results needed.")
        return

    corrs = train[diff_cols + ["result"]].corr()["result"].drop("result")
    corrs = corrs.dropna().sort_values()

    if corrs.empty:
        print("   [!] No valid correlations to plot - skipping plot 6.")
        return

    fig, ax = plt.subplots(figsize=(9, max(6, len(corrs) * 0.3)))
    colors = ["#e74c3c" if v < 0 else "#27ae60" for v in corrs.values]
    ax.barh(corrs.index, corrs.values, color=colors, alpha=0.8)
    ax.axvline(0, color="#333", linewidth=0.8)
    ax.set_xlabel("Pearson Correlation with Result")
    ax.set_title("Feature Correlations with Match Result (all years)\n(green = home advantage, red = away)", fontsize=12)

    pos_patch = mpatches.Patch(color="#27ae60", label="Favours home team")
    neg_patch = mpatches.Patch(color="#e74c3c", label="Favours away team")
    ax.legend(handles=[pos_patch, neg_patch], loc="lower right")

    plt.tight_layout()
    plt.savefig(EDA_DIR / "6_feature_importance.png", dpi=150)
    plt.close()
    print("   Saved: data/eda/6_feature_importance.png")


# ─────────────────────────────────────────────────────────────────────────────
# EDA 7 — Year comparison (missingness-leakage audit)
# ─────────────────────────────────────────────────────────────────────────────

def plot_year_comparison(train: pd.DataFrame):
    """
    Compares basic match-outcome statistics between 2026 and historical
    years. Exists specifically to sanity-check the missingness-leakage
    risk flagged during Phase 2: since 2026 rows have far more complete
    features than historical rows, a model could partly learn to
    distinguish years by missingness rather than real football signal.
    """
    if "year" not in train.columns or train["year"].nunique() < 2:
        print("   [!] Only one year present - skipping plot 7 (year comparison).")
        return

    completed = train[train["home_score"].notna()].copy()
    if completed.empty:
        print("   [!] No completed matches - skipping plot 7.")
        return

    years = sorted(completed["year"].unique())
    stats = []
    for yr in years:
        sub = completed[completed["year"] == yr]
        stats.append({
            "year": yr,
            "matches": len(sub),
            "home_win_rate": (sub["result"] == 1).mean(),
            "draw_rate": (sub["result"] == 0).mean(),
            "away_win_rate": (sub["result"] == -1).mean(),
            "avg_goals": sub["total_goals"].mean(),
        })
    stats_df = pd.DataFrame(stats).set_index("year")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    width = 0.25
    x = np.arange(len(stats_df))
    ax.bar(x - width, stats_df["home_win_rate"], width, label="Home win", color=COLORS["win"])
    ax.bar(x, stats_df["draw_rate"], width, label="Draw", color=COLORS["draw"])
    ax.bar(x + width, stats_df["away_win_rate"], width, label="Away win", color=COLORS["loss"])
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in stats_df.index])
    ax.set_ylabel("Rate")
    ax.set_title("Result Distribution by Year")
    ax.legend()
    ax.set_ylim(0, 1)

    ax2 = axes[1]
    ax2.bar(x, stats_df["avg_goals"], color="#9b59b6", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(y) for y in stats_df.index])
    ax2.set_ylabel("Avg Total Goals per Match")
    ax2.set_title("Scoring Rate by Year")
    for i, val in enumerate(stats_df["avg_goals"]):
        ax2.text(i, val + 0.05, f"{val:.2f}", ha="center", fontsize=9)

    plt.suptitle("Year Comparison - sanity check for combined-dataset modeling",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(EDA_DIR / "7_year_comparison.png", dpi=150)
    plt.close()
    print("   Saved: data/eda/7_year_comparison.png")

    print("\n  Year comparison table:")
    print(stats_df.round(3).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# Print text insights
# ─────────────────────────────────────────────────────────────────────────────

def print_insights(tf: pd.DataFrame, train: pd.DataFrame, year: int = PRIMARY_YEAR):
    print("\n" + "=" * 50)
    print(f"  KEY INSIGHTS - {year}")
    print("=" * 50)

    tf_year = tf[tf["year"] == year]
    if tf_year.empty:
        print(f"  [!] No team_features rows for {year}.")
    else:
        print(f"\n  Highest {QUALITY_LABEL}: {tf_year[QUALITY_COL].idxmax()} ({tf_year[QUALITY_COL].max():.2f})")
        print(f"  Lowest {QUALITY_LABEL}:  {tf_year[QUALITY_COL].idxmin()} ({tf_year[QUALITY_COL].min():.2f})")
        print(f"  Oldest squad:           {tf_year['avg_age'].idxmax()} ({tf_year['avg_age'].max():.1f} yrs)")
        print(f"  Youngest squad:        {tf_year['avg_age'].idxmin()} ({tf_year['avg_age'].min():.1f} yrs)")
        if "avg_caps" in tf_year.columns and tf_year["avg_caps"].notna().any():
            print(f"  Most experienced (caps):{tf_year['avg_caps'].idxmax()} ({tf_year['avg_caps'].max():.1f} avg caps)")
        if "top_league_count" in tf_year.columns:
            print(f"  Most top-league players: {tf_year['top_league_count'].idxmax()} ({tf_year['top_league_count'].max():.0f} players)")

    print(f"\n  --- All years combined ---")
    if len(train) > 0:
        completed = train[train["home_score"].notna()]
        if not completed.empty:
            print(f"  Matches played:    {len(completed)}")
            print(f"  Home win rate:     {(completed['result']==1).mean():.1%}")
            print(f"  Draw rate:         {(completed['result']==0).mean():.1%}")
            print(f"  Away win rate:     {(completed['result']==-1).mean():.1%}")
            print(f"  Avg goals/match:   {completed['total_goals'].mean():.2f}")

            diff_col = f"diff_{QUALITY_COL}"
            if diff_col in completed.columns:
                upset_pool = completed.dropna(subset=[diff_col])
                upsets = upset_pool[
                    ((upset_pool[diff_col] > 0.2) & (upset_pool["result"] == -1)) |
                    ((upset_pool[diff_col] < -0.2) & (upset_pool["result"] == 1))
                ]
                print(f"  Biggest upset:     ", end="")
                if not upsets.empty:
                    u = upsets.iloc[0]
                    print(f"{u['home_team']} vs {u['away_team']} ({int(u['home_score'])}-{int(u['away_score'])}, {int(u['year'])})")
                else:
                    print("None detected yet")
    print("=" * 50)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("  FIFA World Cup 2026 - Phase 3: EDA")
    print("=" * 60)

    tf    = pd.read_csv(PROCESSED / "team_features.csv", index_col="team")
    train = pd.read_csv(PROCESSED / "train_features.csv")

    years_in_tf = sorted(tf["year"].unique()) if "year" in tf.columns else ["(no year column)"]
    print(f"\n  {len(tf)} team-year rows across years: {years_in_tf}")
    print(f"  {len(train)} completed matches for analysis (all years)")

    print("\n  Generating plots...")
    plot_squad_strength(tf, year=PRIMARY_YEAR)
    plot_rating_vs_outcome(train)
    plot_correlation_heatmap(train)
    plot_goal_distribution(train)
    plot_age_analysis(tf, year=PRIMARY_YEAR)
    plot_feature_importance(train)
    plot_year_comparison(train)

    print_insights(tf, train, year=PRIMARY_YEAR)


if __name__ == "__main__":
    run()