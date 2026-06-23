"""
FIFA World Cup 2026 — Phase 2: Feature Engineering

"""

import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR        = Path("data/raw")
HISTORICAL_DIR = Path("data/raw/historical")
OUT_DIR        = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Historical years to include, beyond the current 2026 tournament. Must
# match data/raw/historical/{matches,players}_{year}.csv produced by
# Phase 1's fetch_historical_data().
HISTORICAL_YEARS = [2022, 2018, 2014]

# Real stage values as returned by football-data.org (2026) and Zafronix
# (historical years use different strings — e.g. "group_a" vs "groupstage",
# "r16" vs "last16" — both are mapped onto the same tournament-progress scale).
STAGE_ORDER = {
    "groupstage":    0,
    "last32":        1,
    "last16":        2,
    "quarterfinals": 3,
    "semifinals":    4,
    "thirdplace":    5,
    "final":         5,
    "r32":           1,
    "r16":           2,
    "qf":            3,
    "sf":            4,
}

def _stage_rank(stage_raw: str) -> int:
    stage_raw = str(stage_raw).strip().lower()
    if stage_raw.startswith("group"):
        return 0
    return STAGE_ORDER.get(stage_raw, 0)


TOP_LEAGUE_COUNTRIES = {
    "England", "Spain", "Germany", "Italy", "France",
}


def build_team_features_for_year(players: pd.DataFrame, year: int) -> pd.DataFrame:
    """Compress each squad into one feature row per team for a single year.
    Uses only the columns that actually exist in `players` — historical
    years naturally produce fewer columns than 2026."""
    df = players.copy()

    for col in ("captain", "starter"):
        if col in df.columns:
            df[col] = df[col].astype(bool)

    age_exp = df.groupby("team").agg(
        avg_age        = ("age", "mean"),
        min_age        = ("age", "min"),
        max_age        = ("age", "max"),
        peak_age_count = ("age", lambda x: ((x >= 24) & (x <= 29)).sum()),
        squad_size     = ("player", "count"),
    )
    parts = [age_exp]

    if "caps" in df.columns:
        experience = df.groupby("team").agg(
            avg_caps              = ("caps", "mean"),
            total_caps            = ("caps", "sum"),
            max_caps               = ("caps", "max"),
            total_national_goals   = ("national_goals", "sum"),
            avg_national_goals     = ("national_goals", "mean"),
        )
        parts.append(experience)

# NOTE: confirmed via direct inspection (June 2026) that Zafronix's 2026
    # roster endpoint returns goals=0 for essentially every player, not just
    # a few - this is "not yet sourced" per their own docs, not real zero
    # output. Detect this case and treat the column as NOT SOURCED (NaN)
    # rather than as a real all-zero result, so it doesn't masquerade as
    # signal in Phase 4's correlation-based feature selection.
    goals_col = "goals_2025" if "goals_2025" in df.columns else (
        "goals_tournament" if "goals_tournament" in df.columns else None
    )
    if goals_col and (df[goals_col].fillna(0) == 0).all():
        print(f"   [!] {goals_col} is 0 for every player in this dataset - "
              f"treating as not-sourced (NaN) rather than real zeros.")
        goals_col = None

    if goals_col:
        form = df.groupby("team").agg(
            total_goals_scored = (goals_col, "sum"),
            avg_goals_scored   = (goals_col, "mean"),
            top_scorer_goals   = (goals_col, "max"),
        )
        parts.append(form)

    if "club_country" in df.columns:
        def league_feats(g):
            # club_country is missing for ~15-22% of players in the data
            # we've checked (slightly more for older historical years).
            # Treating a missing value as "not top league" silently
            # understates every team's ratio by a measurable amount -
            # excluding those rows from the denominator instead means the
            # ratio only reflects players we actually have club data for.
            known = g["club_country"].dropna()
            if len(known) == 0:
                return pd.Series({"top_league_count": 0, "top_league_ratio": np.nan})
            return pd.Series({
                "top_league_count": known.isin(TOP_LEAGUE_COUNTRIES).sum(),
                "top_league_ratio": known.isin(TOP_LEAGUE_COUNTRIES).mean(),
            })
        league = df.groupby("team").apply(league_feats, include_groups=False)
        parts.append(league)

    if "height_cm" in df.columns:
        physical = df.groupby("team").agg(
            avg_height_cm = ("height_cm", "mean"),
            avg_weight_kg = ("weight_kg", "mean"),
        )
        parts.append(physical)

    def position_feats(g):
        pos = g["position"].fillna("UNK")
        return pd.Series({
            "gk_count":  (pos == "GK").sum(),
            "def_count": (pos == "DF").sum(),
            "mid_count": (pos == "MF").sum(),
            "fwd_count": (pos == "FW").sum(),
        })
    composition = df.groupby("team").apply(position_feats, include_groups=False)
    parts.append(composition)

    tf = parts[0]
    for p in parts[1:]:
        tf = tf.join(p)

    tf["year"] = year
    return tf


def build_all_team_features() -> pd.DataFrame:
    """Build team_features for 2026 and every historical year, concatenated
    with a `year` column so the same team across years gets separate rows."""
    print("\n⚙  Building team features...")

    players_2026 = pd.read_csv(RAW_DIR / "players.csv")
    tf_2026 = build_team_features_for_year(players_2026, year=2026)
    print(f"   2026: {len(tf_2026)} teams, {len(tf_2026.columns)} features")

    all_tf = [tf_2026]
    for year in HISTORICAL_YEARS:
        path = HISTORICAL_DIR / f"players_{year}.csv"
        if not path.exists():
            print(f"   ⚠ {path} not found — skipping {year}. Run Phase 1 with "
                  f"include_historical=True first.")
            continue
        players_hist = pd.read_csv(path)
        if players_hist.empty:
            print(f"   ⚠ {path} is empty — skipping {year}.")
            continue
        tf_hist = build_team_features_for_year(players_hist, year=year)
        print(f"   {year}: {len(tf_hist)} teams, {len(tf_hist.columns)} features")
        all_tf.append(tf_hist)

    combined = pd.concat(all_tf, axis=0)
    combined.to_csv(OUT_DIR / "team_features.csv")
    return combined


def build_match_features_for_year(
    matches: pd.DataFrame,
    tf_year: pd.DataFrame,
    rankings: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    """Build match feature rows for one year's matches, using that year's
    team_features slice. Rankings are only applied for 2026 — historical
    rows get NaN for fifa_rank/fifa_points rather than incorrect values."""
    has_rankings = (year == 2026) and not rankings.empty
    rank_idx = rankings.set_index("team") if has_rankings else None

    tf_idx = tf_year
    feature_cols = [c for c in tf_year.columns if c != "year"]

    rows = []
    skipped = []

    for _, m in matches.iterrows():
        home, away = m["home_team"], m["away_team"]

        if pd.isna(home) or pd.isna(away):
            continue
        if home not in tf_idx.index or away not in tf_idx.index:
            skipped.append((home, away))
            continue

        hf = tf_idx.loc[home]
        af = tf_idx.loc[away]

        row = {
            "match_id": m["match_id"], "date": m["date"],
            "home_team": home, "away_team": away, "year": year,
            "is_2026": int(year == 2026),
        }

        for col in feature_cols:
            hv, av = hf[col], af[col]
            row[f"home_{col}"] = hv
            row[f"away_{col}"] = av
            row[f"diff_{col}"] = (hv - av) if pd.notna(hv) and pd.notna(av) else np.nan

        if has_rankings:
            def get_rank(team):
                return rank_idx.loc[team, "fifa_rank"] if team in rank_idx.index else np.nan
            def get_pts(team):
                return rank_idx.loc[team, "fifa_points"] if team in rank_idx.index else np.nan

            row["home_fifa_rank"]   = get_rank(home)
            row["away_fifa_rank"]   = get_rank(away)
            row["diff_fifa_rank"]   = get_rank(home) - get_rank(away)
            row["home_fifa_points"] = get_pts(home)
            row["away_fifa_points"] = get_pts(away)
            row["diff_fifa_points"] = get_pts(home) - get_pts(away)
        else:
            row["home_fifa_rank"] = row["away_fifa_rank"] = row["diff_fifa_rank"] = np.nan
            row["home_fifa_points"] = row["away_fifa_points"] = row["diff_fifa_points"] = np.nan

        stage_raw = m.get("stage", "groupstage")
        row["is_neutral_venue"] = int(m.get("neutral_venue", True))
        row["is_knockout"]      = int(not str(stage_raw).strip().lower().startswith("group"))
        row["stage_encoded"]    = _stage_rank(stage_raw)
        row["home_rest_days"]   = m.get("home_rest_days") or 4
        row["away_rest_days"]   = m.get("away_rest_days") or 4
        row["diff_rest_days"]   = row["home_rest_days"] - row["away_rest_days"]

        hs = m.get("home_score")
        as_ = m.get("away_score")
        if pd.notna(hs) and pd.notna(as_):
            row["home_score"]  = int(hs)
            row["away_score"]  = int(as_)
            row["total_goals"] = int(hs) + int(as_)
            row["result"]      = 1 if hs > as_ else (-1 if hs < as_ else 0)
        else:
            row["home_score"] = row["away_score"] = row["total_goals"] = None
            row["result"] = None

        rows.append(row)

    if skipped:
        print(f"  ⚠ [{year}] Skipped {len(skipped)} match(es) — missing team features:")
        for h, a in skipped[:5]:
            print(f"    - {h} vs {a}")
        if len(skipped) > 5:
            print(f"    ... and {len(skipped) - 5} more")

    return pd.DataFrame(rows)


def build_all_match_features(tf_combined: pd.DataFrame) -> pd.DataFrame:
    """Build match features for 2026 + every historical year, concatenated
    into one DataFrame. Columns missing for a given year are filled NaN
    automatically by pandas' concat — no manual reindexing needed."""
    print("\n⚙  Building match features...")

    rankings = pd.read_csv(RAW_DIR / "rankings.csv")

    matches_2026 = pd.read_csv(RAW_DIR / "matches.csv")
    tf_2026 = tf_combined[tf_combined["year"] == 2026].drop(columns=["year"])
    df_2026 = build_match_features_for_year(matches_2026, tf_2026, rankings, year=2026)
    print(f"   2026: {len(df_2026)} match rows built")

    all_match_dfs = [df_2026]
    for year in HISTORICAL_YEARS:
        path = HISTORICAL_DIR / f"matches_{year}.csv"
        if not path.exists():
            print(f"   ⚠ {path} not found — skipping {year}.")
            continue
        matches_hist = pd.read_csv(path)
        if matches_hist.empty:
            print(f"   ⚠ {path} is empty — skipping {year}.")
            continue
        tf_hist = tf_combined[tf_combined["year"] == year].drop(columns=["year"])
        df_hist = build_match_features_for_year(matches_hist, tf_hist, rankings, year=year)
        print(f"   {year}: {len(df_hist)} match rows built")
        all_match_dfs.append(df_hist)

    combined = pd.concat(all_match_dfs, axis=0, ignore_index=True)
    return combined


def run():
    print("=" * 60)
    print("  FIFA World Cup 2026 — Phase 2: Feature Engineering")
    print("=" * 60)
    print(f"\n  Including historical years: {HISTORICAL_YEARS}")

    tf_combined = build_all_team_features()
    df = build_all_match_features(tf_combined)

    train_df   = df[df["result"].notna()].copy()
    predict_df = df[df["result"].isna()].copy()

    train_df.to_csv(OUT_DIR / "train_features.csv",   index=False)
    predict_df.to_csv(OUT_DIR / "predict_features.csv", index=False)

    print(f"\n  Features per match: {len(df.columns)}")
    print(f"  Training rows:      {len(train_df)} (completed matches, all years)")
    print(f"  Prediction rows:    {len(predict_df)} (upcoming 2026 matches)")
    print(f"  By year: {train_df['year'].value_counts().to_dict()}")

    diff_cols = [c for c in train_df.columns if c.startswith("diff_")]
    completeness = train_df[diff_cols].notna().mean().sort_values()
    sparse = completeness[completeness < 1.0]
    if not sparse.empty:
        print("\n  ⚠ Features with missing values across years (expected for "
              "2026-only / ranking features on historical rows):")
        for feat, frac in sparse.items():
            print(f"    {feat:<35} {frac:.0%} complete")

    if len(train_df) > 3:
        corrs = train_df[diff_cols + ["result"]].corr()["result"].drop("result").dropna()
        print("\n  Top predictive features (correlation with result, all years):")
        for feat, val in corrs.abs().sort_values(ascending=False).head(10).items():
            print(f"    {feat:<35} {val:.3f}")
    else:
        print("\n  ⚠ Not enough completed matches yet to compute correlations.")

    return train_df, predict_df, tf_combined


if __name__ == "__main__":
    run()