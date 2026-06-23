"""
FIFA World Cup 2026 — Phase 1: Data Collection
Collects all raw data needed for the project:
  1. Match results & fixtures  → football-data.org API (free)
  2. Player rosters (2026)    → Zafronix WC API
  3. Squad lists               → transfermarkt.com (scraped, unused — kept for reference)
  4. FIFA rankings             → hardcoded from official site (June 2026)
  5. Historical WC data (2022) → Zafronix WC API (matches + rosters)

"""
import os
from dotenv import load_dotenv
import time
import json
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
load_dotenv()
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

HISTORICAL_DIR = Path("data/raw/historical")
HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. MATCH RESULTS & FIXTURES — football-data.org free API
# ─────────────────────────────────────────────────────────────────────────────

def fetch_matches(api_key: str) -> pd.DataFrame:
    """
    Fetch all WC 2026 matches from football-data.org.
    Free tier: 10 requests/minute. Get your free key at:
      https://www.football-data.org/client/register
    """
    print("\n[1/5] Fetching match data from football-data.org...")

    url = "https://api.football-data.org/v4/competitions/WC/matches"
    resp = requests.get(url, headers={"X-Auth-Token": api_key}, timeout=15)
    resp.raise_for_status()

    matches = resp.json().get("matches", [])
    rows = []
    for m in matches:
        ft = m["score"]["fullTime"]
        rows.append({
            "match_id":       m["id"],
            "date":           m["utcDate"][:10],
            "stage":          m["stage"].lower().replace("_", ""),
            "group":          m.get("group", ""),
            "home_team":      m["homeTeam"]["name"],
            "away_team":      m["awayTeam"]["name"],
            "home_score":     ft.get("home"),
            "away_score":     ft.get("away"),
            "status":         m["status"],
            "neutral_venue":  True,
            "home_rest_days": None,
            "away_rest_days": None,
        })

    df = pd.DataFrame(rows)
    df.to_csv(RAW_DIR / "matches.csv", index=False)
    print(f"   Saved {len(df)} matches → data/raw/matches.csv")
    return df


def compute_rest_days(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Add rest days between matches for each team."""
    df = matches_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    last_match = {}
    home_rest, away_rest = [], []

    for _, row in df.iterrows():
        for team, col in [(row["home_team"], home_rest), (row["away_team"], away_rest)]:
            if team in last_match:
                rest = (row["date"] - last_match[team]).days
            else:
                rest = 7
            col.append(rest)
            last_match[team] = row["date"]

    df["home_rest_days"] = home_rest
    df["away_rest_days"] = away_rest
    df.to_csv(RAW_DIR / "matches.csv", index=False)
    return df

# 2. PLAYER STATS

def fetch_fbref_player_stats() -> pd.DataFrame:
    """
    Scrape player stats from FBref World Cup 2026 stats page.
    Respects robots.txt — 3s delay between requests.
    """
    if BeautifulSoup is None:
        print("   ⚠ BeautifulSoup not installed. Run: pip install beautifulsoup4 lxml")
        return pd.DataFrame()

    print("\n[2/5] Fetching player stats from FBref...")

    url = "https://fbref.com/en/comps/1/stats/World-Cup-Stats"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        time.sleep(3)

        tables = pd.read_html(resp.text, header=1)
        df = tables[0].copy()

        # Clean multi-level headers that fbref sometimes returns
        df.columns = [c[1] if isinstance(c, tuple) else c for c in df.columns]

        # Rename to standard schema
        rename_map = {
            "Player":    "player",
            "Squad":     "team",
            "Nation":    "nation",
            "Pos":       "position",
            "Age":       "age",
            "MP":        "matches_played",
            "Gls":       "goals_wc",
            "Ast":       "assists_wc",
            "G+A":       "g_a_wc",
            "Min":       "minutes_wc",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        df = df[df["player"].notna() & (df["player"] != "Player")]
        df["age"] = pd.to_numeric(df.get("age", 25), errors="coerce").fillna(25).astype(int)

        df.to_csv(RAW_DIR / "fbref_player_stats.csv", index=False)
        print(f"   Saved {len(df)} player records → data/raw/fbref_player_stats.csv")
        return df

    except Exception as e:
        print(f"   ⚠ FBref scrape failed: {e}")
        return pd.DataFrame()


def fetch_sofifa_ratings(team_names: list[str]) -> pd.DataFrame:
    """
    Scrape EA FC 26 player ratings from sofifa.com.
    Tip: sofifa allows you to filter by nationality — much faster.
    """
    if BeautifulSoup is None:
        return pd.DataFrame()

    print("\n[3/5] Fetching player ratings from SoFIFA...")

    all_rows = []
    for team in tqdm(team_names, desc="Teams"):
        # Skip missing/non-string team names (e.g. NaN from unplayed matches)
        if not isinstance(team, str) or not team.strip():
            continue
        # sofifa nationality filter — team name must match sofifa's URL slug
        team_slug = team.lower().replace(" ", "+")
        url = f"https://sofifa.com/players?keyword=&nationality={team_slug}&col=oa&sort=desc"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            time.sleep(2)   # be respectful to the server

            soup = BeautifulSoup(resp.text, "lxml")
            table = soup.find("table", {"id": "player-table"})
            if not table:
                continue

            for row in table.find("tbody").find_all("tr"):
                cols = row.find_all("td")
                if len(cols) < 4:
                    continue
                all_rows.append({
                    "player":         cols[1].get_text(strip=True),
                    "team":           team,
                    "sofifa_rating":  int(cols[3].get_text(strip=True) or 0),
                    "position":       cols[2].get_text(strip=True),
                })
        except Exception as e:
            print(f"   ⚠ SoFIFA failed for {team}: {e}")
            continue

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df.to_csv(RAW_DIR / "sofifa_ratings.csv", index=False)
        print(f"   Saved {len(df)} ratings → data/raw/sofifa_ratings.csv")
    return df



ZAFRONIX_BASE = "https://api.zafronix.com/fifa/worldcup/v1"
WC_YEAR = 2026

HISTORICAL_YEARS = [2022, 2018, 2014]


def _zafronix_get(endpoint: str, params: dict, api_key: str, cache_prefix: str = "zafronix"):
    """
    GET a single Zafronix endpoint, with disk caching.
    Cache key is built from the endpoint + sorted params.
    cache_prefix lets historical and 2026 calls use distinct cache
    namespaces even if endpoint/params happen to collide.
    """
    cache_key = endpoint.strip("/").replace("/", "_") + "__" + "_".join(
        f"{k}={v}" for k, v in sorted(params.items())
    )
    cache_file = CACHE_DIR / f"{cache_prefix}_{cache_key}.json"

    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    url = f"{ZAFRONIX_BASE}/{endpoint.strip('/')}"
    headers = {"X-API-Key": api_key}
    resp = requests.get(url, headers=headers, params=params, timeout=20)

    if resp.status_code == 404:
        # Team/roster not found for this name — not fatal, just skip
        return None

    resp.raise_for_status()
    data = resp.json()

    cache_file.write_text(json.dumps(data), encoding="utf-8")
    time.sleep(0.3)  # well under the 250/day cap, but be polite
    return data


def fetch_zafronix_rosters(api_key: str, team_names: list[str]) -> pd.DataFrame:
    """
    Fetch WC 2026 squad lists for every team via Zafronix's
    /teams/{name}/roster?year=2026 endpoint.
    Cached per team so re-running the script costs zero extra requests
    once everything has been fetched once.
    """
    clean_teams = [t for t in team_names if isinstance(t, str) and t.strip()]
    if not clean_teams:
        print("   ⚠ No team names available — skipping Zafronix rosters.")
        return pd.DataFrame()

    print("\n[3b] Fetching player rosters from Zafronix WC API...")

    all_rows = []
    for team in tqdm(clean_teams, desc="Teams"):
        try:
            data = _zafronix_get(
                f"teams/{team}/roster", {"year": WC_YEAR}, api_key
            )
        except Exception as e:
            print(f"   ⚠ Zafronix failed for {team}: {e}")
            continue

        if not data:
            continue

        for p in data:
            club = p.get("club") or {}
            all_rows.append({
                "player":          p.get("name"),
                "team":            team,
                "jersey":          p.get("jersey"),
                "position":        p.get("position"),
                "age":             p.get("ageAtTournament"),
                "club":            club.get("name"),
                "club_country":    club.get("country"),
                "goals_2025":      p.get("goals", 0),
                "captain":         p.get("captain", False),
                "starter":         p.get("starter", False),
                "height_cm":       p.get("heightCm"),
                "weight_kg":       p.get("weightKg"),
                "dominant_foot":   p.get("dominantFoot"),
                "caps":            p.get("caps"),
                "national_goals":  p.get("nationalGoals"),
            })

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df.to_csv(RAW_DIR / "zafronix_rosters.csv", index=False)
        print(f"   Saved {len(df)} player records → data/raw/zafronix_rosters.csv")
    return df

# 4. SQUAD LISTS 

def fetch_squad_lists() -> pd.DataFrame:
    """
    Scrape WC 2026 squad lists from transfermarkt.
    Returns player name, team, age, position, club, club_league, market_value.
    """
    if BeautifulSoup is None:
        return pd.DataFrame()

    print("\n[4/5] Fetching squad lists from Transfermarkt...")

    url = "https://www.transfermarkt.com/weltmeisterschaft-2026/teilnehmer/pokalwettbewerb/WM26"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        time.sleep(3)

        soup = BeautifulSoup(resp.text, "lxml")
        # Transfermarkt has team links — follow each to get squad
        team_links = []
        for a in soup.select("table.items tbody tr td.hauptlink a"):
            href = a.get("href", "")
            if "/startseite/verein/" not in href:
                team_links.append(("https://www.transfermarkt.com" + href, a.get_text(strip=True)))

        rows = []
        for team_url, team_name in tqdm(team_links[:10], desc="Squads"):  # limit for demo
            try:
                r = requests.get(team_url, headers=HEADERS, timeout=15)
                r.raise_for_status()
                time.sleep(2)
                squad_tables = pd.read_html(r.text)
                if squad_tables:
                    sq = squad_tables[0]
                    sq["team"] = team_name
                    rows.append(sq)
            except Exception:
                continue

        if rows:
            df = pd.concat(rows, ignore_index=True)
            df.to_csv(RAW_DIR / "squads_raw.csv", index=False)
            print(f"   Saved squad data → data/raw/squads_raw.csv")
            return df

    except Exception as e:
        print(f"   ⚠ Transfermarkt scrape failed: {e}")

    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# 5. FIFA RANKINGS — hardcoded from official FIFA site (June 2026)
# ─────────────────────────────────────────────────────────────────────────────

def save_fifa_rankings() -> pd.DataFrame:
    """
    FIFA World Rankings as of June 2026.
    Update from: https://www.fifa.com/fifa-world-ranking/men
    """
    print("\n[5/5] Saving FIFA rankings...")

    rankings = [
        ("Argentina",             1, 1867),
        ("France",                2, 1851),
        ("England",               3, 1794),
        ("Brazil",                4, 1782),
        ("Belgium",               5, 1761),
        ("Portugal",              6, 1750),
        ("Netherlands",           7, 1742),
        ("Spain",                 8, 1735),
        ("Germany",               9, 1728),
        ("Uruguay",              10, 1710),
        ("Colombia",             11, 1697),
        ("Morocco",              12, 1685),
        ("USA",                  13, 1670),
        ("Japan",                14, 1658),
        ("Switzerland",          15, 1641),
        ("Mexico",               16, 1629),
        ("Turkiye",              17, 1618),
        ("South Korea",          18, 1607),
        ("Canada",               19, 1595),
        ("Australia",            20, 1584),
        ("Ghana",                25, 1530),
        ("Scotland",             30, 1490),
        ("Ivory Coast",          33, 1460),
        ("Ecuador",              35, 1445),
        ("Sweden",               18, 1607),
        ("Austria",              28, 1510),
        ("South Africa",         60, 1310),
        ("Saudi Arabia",         56, 1340),
        ("IR Iran",              21, 1573),
        ("Iraq",                 58, 1325),
        ("Cape Verde",           45, 1395),
        ("Panama",               73, 1240),
        ("Haiti",                85, 1180),
        ("Qatar",                37, 1440),
        ("Uzbekistan",           69, 1270),
        ("Bosnia and Herzegovina",55, 1345),
        ("Paraguay",             50, 1370),
        ("Curacao",             100, 1105),
        ("New Zealand",          95, 1125),
        ("Egypt",                34, 1455),
        ("Tunisia",              27, 1515),
    ]

    df = pd.DataFrame(rankings, columns=["team", "fifa_rank", "fifa_points"])
    df.to_csv(RAW_DIR / "rankings.csv", index=False)
    print(f"   Saved {len(df)} team rankings → data/raw/rankings.csv")
    return df

# 6. HISTORICAL DATA 


def fetch_historical_matches(api_key: str, year: int) -> pd.DataFrame:
    """Fetch all matches for one historical WC year, same row-per-match
    shape as fetch_matches() above, so Phase 2 can treat them uniformly."""
    print(f"\n[6a] Fetching {year} historical matches...")
    data = _zafronix_get("matches", {"year": year}, api_key, cache_prefix="zafronix_hist")
    if not data or not data.get("data"):
        print(f"   ⚠ No match data returned for {year}.")
        return pd.DataFrame()

    rows = []
    for m in data["data"]:
        home = (m.get("homeTeam") or "").strip()
        away = (m.get("awayTeam") or "").strip()
        if not home or not away:
            continue

        rows.append({
            "match_id":       m.get("id"),
            "date":           m.get("date"),
            "stage":          m.get("stage", "").strip().lower(),
            "group":          m.get("stage", "") if "group" in str(m.get("stage", "")) else "",
            "home_team":      home,
            "away_team":      away,
            "home_score":     m.get("homeScore"),
            "away_score":     m.get("awayScore"),
            "status":         m.get("status"),
            "neutral_venue":  True,
            "home_rest_days": None,
            "away_rest_days": None,
            "year":           year,
        })

    df = pd.DataFrame(rows)
    out_path = HISTORICAL_DIR / f"matches_{year}.csv"
    df.to_csv(out_path, index=False)
    print(f"   Saved {len(df)} matches → {out_path}")
    return df


def fetch_historical_rosters(api_key: str, year: int, team_names: list) -> pd.DataFrame:
    """Fetch squad rosters for every team in a historical WC year.
    Only includes fields that actually exist for historical squads —
    no caps/height/weight/captain/starter, unlike fetch_zafronix_rosters()."""
    print(f"\n[6b] Fetching {year} historical rosters for {len(team_names)} teams...")

    all_rows = []
    for team in tqdm(team_names, desc=f"Teams ({year})"):
        try:
            data = _zafronix_get(
                f"teams/{team}/roster", {"year": year}, api_key, cache_prefix="zafronix_hist"
            )
        except Exception as e:
            print(f"   ⚠ Zafronix failed for {team} ({year}): {e}")
            continue

        if not data:
            print(f"   ⚠ No roster found for {team} in {year} — possible name mismatch.")
            continue

        for p in data:
            club = p.get("club") or {}
            all_rows.append({
                "player":            p.get("name"),
                "team":              team,
                "jersey":            p.get("jersey"),
                "position":          p.get("position"),
                "age":               p.get("ageAtTournament"),
                "club":              club.get("name"),
                "club_country":      club.get("country"),
                "goals_tournament":  p.get("goals", 0),
                "year":              year,
            })

    df = pd.DataFrame(all_rows)
    out_path = HISTORICAL_DIR / f"players_{year}.csv"
    df.to_csv(out_path, index=False)
    print(f"   Saved {len(df)} player records → {out_path}")
    return df


def fetch_historical_data(api_key: str, years: list = None):
    """Run historical match + roster collection for each year in `years`."""
    years = years or HISTORICAL_YEARS

    for year in years:
        matches_df = fetch_historical_matches(api_key, year)
        if matches_df.empty:
            print(f"   Skipping roster fetch for {year} — no matches found.")
            continue

        teams = sorted(set(
            matches_df["home_team"].dropna().tolist() +
            matches_df["away_team"].dropna().tolist()
        ))
        fetch_historical_rosters(api_key, year, teams)


# ─────────────────────────────────────────────────────────────────────────────
# 7. BUILD MASTER PLAYERS CSV — merge all sources (2026 only)
# ─────────────────────────────────────────────────────────────────────────────

def build_master_players_csv(zafronix_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the final players.csv from Zafronix WC API roster data.
    Falls back to sensible defaults for missing data.
    """
    print("\n[6/6] Building master players CSV...")

    if zafronix_df.empty:
        print("   ⚠ No player data was collected from Zafronix.")
        print("   → Saving an empty players.csv. Check your API key, quota,")
        print("     or whether team names matched Zafronix's naming.")
        empty_df = pd.DataFrame(columns=[
            "player", "team", "jersey", "position", "age", "club",
            "club_country", "goals_2025", "captain", "starter",
            "height_cm", "weight_kg", "dominant_foot", "caps", "national_goals",
        ])
        empty_df.to_csv(RAW_DIR / "players.csv", index=False)
        return empty_df

    merged = zafronix_df.copy()

    # Fill mandatory columns with sensible defaults if missing
    for col, default in [
        ("age", 26), ("position", "MF"), ("goals_2025", 0),
        ("captain", False), ("starter", False),
    ]:
        if col not in merged.columns:
            merged[col] = default
        else:
            merged[col] = merged[col].fillna(default)

    merged.to_csv(RAW_DIR / "players.csv", index=False)
    print(f"   Saved {len(merged)} players → data/raw/players.csv")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run(api_key: str = None, zafronix_key: str = None, include_historical: bool = True):
    print("=" * 60)
    print("  FIFA World Cup 2026 — Phase 1: Data Collection")
    print("=" * 60)

    if not api_key:
        raise SystemExit(
            "\n❌ No football-data.org API key found.\n"
            "   Set FOOTBALL_DATA_API_KEY in your .env file, e.g.:\n"
            "   FOOTBALL_DATA_API_KEY=your_key_here\n"
            "   Get a free key at: https://www.football-data.org/client/register"
        )

    matches_df = fetch_matches(api_key)
    matches_df = compute_rest_days(matches_df)
    teams = list(set(matches_df["home_team"].dropna().tolist() + matches_df["away_team"].dropna().tolist()))

    if zafronix_key:
        zafronix_df = fetch_zafronix_rosters(zafronix_key, teams)
    else:
        print("\n⚠ No ZAFRONIX_API_KEY found in .env — skipping player rosters.")
        print("   Get a free key at: https://api.zafronix.com/signup")
        zafronix_df = pd.DataFrame()

    build_master_players_csv(zafronix_df)
    save_fifa_rankings()

    if include_historical:
        if zafronix_key:
            print("\n" + "-" * 60)
            print("  Collecting historical World Cup data (for model training)")
            print("-" * 60)
            fetch_historical_data(zafronix_key, HISTORICAL_YEARS)
        else:
            print("\n⚠ Skipping historical data collection — no ZAFRONIX_API_KEY.")

 

if __name__ == "__main__":
    API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
    ZAFRONIX_KEY = os.getenv("ZAFRONIX_API_KEY")
    run(api_key=API_KEY, zafronix_key=ZAFRONIX_KEY, include_historical=True)