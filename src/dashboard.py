import json, pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

PROCESSED  = Path("data/processed")
MODELS_DIR = Path("data/models")

st.set_page_config(page_title="WC 2026 Predictor", page_icon="⚽", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;600;700&family=Inter:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Oswald', sans-serif !important; letter-spacing: 1px; }
.stButton > button { background: #0d2b0d; color: #7fce82; border: 1px solid #2d6e2d; border-radius: 8px; font-family: 'Oswald', sans-serif; letter-spacing: 1px; padding: 0.5rem 1.5rem; }
.stButton > button:hover { background: #1a5c1a; color: #fff; border-color: #4caf50; }
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
div[data-testid="stMetricValue"] { color: #2f8f33 !important; font-family: 'Oswald', sans-serif !important; }
div[data-testid="stMetricLabel"] { color: #6b766b !important; }
div[data-testid="stHorizontalBlock"] .stButton > button {
    background: transparent !important; color: #888 !important; border: none !important;
    border-bottom: 2px solid transparent !important; border-radius: 0 !important;
    font-family: 'Oswald', sans-serif !important; font-size: 13px !important;
    letter-spacing: 1px !important; padding: 10px 4px !important; width: 100% !important;
}
div[data-testid="stHorizontalBlock"] .stButton > button:hover { color: #4caf50 !important; border-bottom: 2px solid #4caf50 !important; background: transparent !important; }
div[data-testid="stHorizontalBlock"] .stButton > button[kind="primary"] { color: #4caf50 !important; border-bottom: 2px solid #4caf50 !important; background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ── Top bar ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
</style>
<div style="display:flex;align-items:center;justify-content:space-between;padding:0 8px;height:56px;
     border-bottom:1px solid #e0e8e0;margin-bottom:24px;background:#fff;
     box-shadow:0 1px 8px rgba(20,50,20,0.06);">
    <div style="display:flex;align-items:center;gap:10px;font-family:Oswald,sans-serif;font-size:17px;
         color:#16331a;letter-spacing:2px;font-weight:700;">
        <div style="width:8px;height:8px;border-radius:50%;background:#4caf50;animation:pulse 1.8s ease infinite;"></div>
        WC 2026 PREDICTOR
    </div>
    <div style="display:flex;align-items:center;gap:5px;background:#f0f8f0;border:1px solid #b8e0b8;
         border-radius:20px;padding:4px 12px;font-size:11px;color:#2f8f33;letter-spacing:1px;font-weight:600;">
        <div style="width:6px;height:6px;border-radius:50%;background:#4caf50;animation:pulse 1.8s ease infinite;"></div>
        LIVE
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tab navigation ─────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "🎮  Kick & Predict"

tab_labels = ["🎮  Kick & Predict","🔮  Match Predictions","🏆  Who Wins the Cup?","⚔️  Team vs Team"]
cols = st.columns(4)
for col, label in zip(cols, tab_labels):
    with col:
        is_active = st.session_state.page == label
        if st.button(label, key=f"nav_{label}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.page = label
            st.rerun()

st.markdown("<hr style='border:none;border-top:1px solid #e0e8e0;margin:4px 0 28px'>", unsafe_allow_html=True)
page = st.session_state.page

# ── Data ───────────────────────────────────────────────────────────────────
# NOTE: only loads what's actually used by the four pages below.
# (The original draft also loaded train_features.csv into a "matches" key
# that no page ever referenced - dropped here to avoid the unnecessary read.)
@st.cache_data(ttl=1800)
def load_all():
    paths = {"preds": PROCESSED/"predictions.csv",
             "sim": PROCESSED/"simulation_results.csv",
             "teams": PROCESSED/"team_features.csv"}
    return {k: pd.read_csv(p) if p.exists() else pd.DataFrame() for k, p in paths.items()}

@st.cache_resource
def load_model():
    mp, fp = MODELS_DIR/"final_model.pkl", MODELS_DIR/"feature_cols.json"
    if mp.exists() and fp.exists():
        with open(mp,"rb") as f: bundle = pickle.load(f)
        with open(fp) as f: cols = json.load(f)
        return bundle["model"], cols
    return None, None

data  = load_all()
model, feature_cols = load_model()


# ══════════════════════════════════════════════════════════════
# PAGE: KICK & PREDICT
# ══════════════════════════════════════════════════════════════
if page == "🎮  Kick & Predict":
    st.markdown("<h1 style='color:#16331a;'>Kick & Predict</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6b766b;margin-top:-8px;'>Pick a match, click the ball, see who wins.</p>", unsafe_allow_html=True)

    preds = data["preds"]

    home_team    = "Team A"
    away_team    = "Team B"
    match_date   = "TBD"
    hw_pct       = 40.0
    draw_pct     = 25.0
    aw_pct       = 35.0
    winner       = "TBD"
    winner_emoji = "⚽"
    commentary   = "Pick a match above to get a prediction."

    if not preds.empty:
        match_options = [
            f"{r['home_team']}  vs  {r['away_team']}  ({r.get('date','TBD')})"
            for _, r in preds.iterrows()
        ]
        selected   = st.selectbox("Pick a scheduled match", match_options)
        idx        = match_options.index(selected)
        sel_row    = preds.iloc[idx]
        home_team  = sel_row["home_team"]
        away_team  = sel_row["away_team"]
        match_date = sel_row.get("date", "TBD")
        hw_pct     = float(sel_row.get("home_win_%", 40))
        draw_pct   = float(sel_row.get("draw_%", 25))
        aw_pct     = float(sel_row.get("away_win_%", 35))

    if hw_pct >= aw_pct and hw_pct >= draw_pct:
        winner, winner_emoji = home_team, "🏠"
    elif aw_pct >= hw_pct and aw_pct >= draw_pct:
        winner, winner_emoji = away_team, "✈️"
    else:
        winner, winner_emoji = "Draw", "🤝"

    if winner not in ("Draw", "TBD"):
        diff = abs(hw_pct - aw_pct)
        if diff > 40:   commentary = f"Comfortable win expected for {winner}."
        elif diff > 20: commentary = f"{winner} are the clear favourites here."
        elif diff > 10: commentary = f"Close match but {winner} have the edge."
        else:           commentary = f"Really tight one — {winner} just shade it on form."
    elif winner == "Draw":
        commentary = "Too close to call. This one could go either way."

    st.markdown("<br>", unsafe_allow_html=True)

    # Match header
    st.markdown(f"""
    <div style='display:flex;align-items:center;justify-content:space-between;
         background:#fff;border:1px solid #e0e8e0;border-radius:16px;
         padding:20px 28px;margin-bottom:20px;box-shadow:0 2px 12px rgba(20,50,20,0.05);'>
      <div>
        <div style='font-size:11px;color:#888;letter-spacing:2px;margin-bottom:4px;'>HOME</div>
        <div style='font-family:Oswald;font-size:26px;font-weight:700;color:#2f8f33;'>{home_team}</div>
      </div>
      <div style='text-align:center;'>
        <div style='font-family:Oswald;font-size:13px;color:#aaa;letter-spacing:3px;'>VS</div>
        <div style='font-size:12px;color:#888;margin-top:4px;'>📅 {match_date}</div>
      </div>
      <div style='text-align:right;'>
        <div style='font-size:11px;color:#888;letter-spacing:2px;margin-bottom:4px;'>AWAY</div>
        <div style='font-family:Oswald;font-size:26px;font-weight:700;color:#cf5454;'>{away_team}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Pitch + ball ──────────────────────────────────────────────────────
    components.html(f"""
<!DOCTYPE html><html><head>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Oswald:wght@700;800&display=swap" rel="stylesheet">
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:transparent;font-family:'Oswald',sans-serif;}}
.scene{{position:relative;width:100%;height:380px;border-radius:20px;overflow:hidden;cursor:pointer;
  box-shadow:0 16px 40px rgba(20,50,20,0.18);
  background:linear-gradient(#bfe3ff 0%,#d9f0ff 36%,#5fa83f 36%,#4e9433 100%);}}
.sun{{position:absolute;top:-60px;right:-40px;width:220px;height:220px;border-radius:50%;
  background:radial-gradient(circle,rgba(255,247,214,0.9),rgba(255,247,214,0));}}
.stripes{{position:absolute;left:0;right:0;top:36%;bottom:0;
  background:repeating-linear-gradient(90deg,rgba(255,255,255,0.05) 0 60px,rgba(0,0,0,0.04) 60px 120px);}}
.arc{{position:absolute;left:50%;transform:translateX(-50%);bottom:40px;width:300px;height:120px;
  border:3px solid rgba(255,255,255,0.55);border-radius:50%;border-bottom-color:transparent;}}
.spot{{position:absolute;left:50%;transform:translateX(-50%);bottom:90px;width:10px;height:10px;
  border-radius:50%;background:rgba(255,255,255,0.7);}}
.goal{{position:absolute;left:50%;top:36%;transform:translateX(-50%);width:360px;height:150px;}}
.net{{position:absolute;inset:0;transform-origin:top center;border-radius:4px;
  background-color:rgba(10,22,40,0.18);
  background-image:linear-gradient(rgba(255,255,255,0.5) 1px,transparent 1px),
                   linear-gradient(90deg,rgba(255,255,255,0.5) 1px,transparent 1px);
  background-size:18px 18px;}}
.post{{position:absolute;background:#F8F9FA;border-radius:4px;box-shadow:0 0 8px rgba(0,0,0,0.2);}}
.post-l{{left:0;top:0;bottom:0;width:10px;}} .post-r{{right:0;top:0;bottom:0;width:10px;}}
.post-t{{left:0;right:0;top:0;height:10px;}}
.ball{{position:absolute;left:50%;top:78%;width:56px;height:56px;transform:translate(-50%,-50%);
  z-index:5;animation:nudge 1.6s ease-in-out infinite;}}
.ball-skin{{width:100%;height:100%;border-radius:50%;position:relative;overflow:hidden;
  background:radial-gradient(circle at 34% 30%,#ffffff,#e8e8e8 60%,#c4c4c4);
  box-shadow:0 6px 12px rgba(0,0,0,0.35);}}
.p0{{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);width:42%;height:42%;
  background:#1a1a1a;clip-path:polygon(50% 0%,95% 35%,78% 90%,22% 90%,5% 35%);}}
.pp{{position:absolute;width:21%;height:21%;background:#1a1a1a;
  clip-path:polygon(50% 0%,100% 60%,50% 100%,0% 60%);}}
.goal-text{{position:absolute;left:50%;top:30%;transform:translate(-50%,-50%) scale(0.4);opacity:0;
  z-index:8;pointer-events:none;font-size:84px;font-weight:800;color:#F8F9FA;letter-spacing:-0.03em;
  text-shadow:0 6px 24px rgba(0,0,0,0.5);white-space:nowrap;}}
.hint{{text-align:center;margin-top:12px;font-size:13px;color:#6b766b;font-family:'Oswald',sans-serif;
  letter-spacing:2px;transition:opacity 0.3s;}}
@keyframes nudge{{0%,100%{{transform:translate(-50%,-50%) scale(1);}}50%{{transform:translate(-50%,-50%) scale(1.06);}}}}
@keyframes shootToNet{{0%{{top:78%;width:56px;height:56px;}}100%{{top:40%;width:34px;height:34px;}}}}
@keyframes netRipple{{0%{{transform:scaleY(1);}}30%{{transform:scaleY(0.88);}}60%{{transform:scaleY(1.04);}}100%{{transform:scaleY(1);}}}}
@keyframes goalPop{{0%{{transform:translate(-50%,-50%) scale(0.4);opacity:0;}}30%{{transform:translate(-50%,-50%) scale(1.12);opacity:1;}}70%{{transform:translate(-50%,-50%) scale(1);opacity:1;}}100%{{transform:translate(-50%,-50%) scale(1);opacity:1;}}}}
@keyframes fadeUp{{from{{opacity:0;transform:translateY(18px);}}to{{opacity:1;transform:translateY(0);}}}}

#result{{display:none;margin-top:16px;}}
.result-in{{display:block !important;animation:fadeUp 0.55s ease forwards;}}
</style></head><body>

<div class="scene" id="pitch" onclick="kickBall()">
  <div class="sun"></div><div class="stripes"></div>
  <div class="arc"></div><div class="spot"></div>
  <div class="goal">
    <div class="net" id="net"></div>
    <div class="post post-l"></div><div class="post post-r"></div><div class="post post-t"></div>
  </div>
  <div class="ball" id="ball">
    <div class="ball-skin">
      <div class="p0"></div>
      <div class="pp" style="left:8%;top:14%"></div>
      <div class="pp" style="right:6%;top:18%"></div>
      <div class="pp" style="left:14%;bottom:8%"></div>
      <div class="pp" style="right:12%;bottom:10%"></div>
    </div>
  </div>
  <div class="goal-text" id="gt">GOAL!</div>
</div>
<div class="hint" id="hint">CLICK TO SHOOT</div>

<div id="result">
  <!-- WINNER HERO -->
  <div style="position:relative;border-radius:20px;overflow:hidden;background:#0d2b0d;
       box-shadow:0 16px 40px rgba(13,43,13,0.28);padding:32px 28px;margin-bottom:14px;text-align:center;">
    <div style="position:absolute;inset:0;background:radial-gradient(90% 160% at 50% -30%,rgba(76,175,80,0.4),transparent 60%);"></div>
    <div style="position:relative;">
      <div style="font-size:11px;letter-spacing:0.24em;color:#7fce82;font-weight:600;font-family:sans-serif;">PREDICTED WINNER</div>
      <div style="font-family:'Oswald',sans-serif;font-size:48px;font-weight:800;color:#fff;letter-spacing:1px;line-height:1.05;margin-top:8px;">{winner_emoji} {winner.upper()}</div>
      <div style="display:inline-flex;align-items:center;gap:10px;margin-top:16px;background:rgba(255,255,255,0.08);
           border:1px solid rgba(127,206,130,0.4);border-radius:9999px;padding:8px 18px;">
        <span style="font-family:'Oswald',sans-serif;font-size:22px;font-weight:800;color:#7fce82;line-height:1;">{max(hw_pct,draw_pct,aw_pct):.0f}%</span>
        <span style="width:1px;height:16px;background:rgba(127,206,130,0.4);"></span>
        <span style="font-size:11px;letter-spacing:0.14em;color:rgba(255,255,255,0.65);font-family:sans-serif;">CONFIDENCE</span>
      </div>
      <div style="font-size:13px;color:rgba(255,255,255,0.6);margin-top:14px;font-family:sans-serif;">{commentary}</div>
    </div>
  </div>
  <!-- PROBABILITY CARD -->
  <div style="background:#fff;border:1px solid #e0e8e0;border-radius:20px;
       box-shadow:0 4px 18px rgba(20,50,20,0.06);padding:24px 28px;">
    <div style="display:flex;height:12px;border-radius:9999px;overflow:hidden;background:#f0f4f0;margin-bottom:18px;">
      <div style="width:{min(hw_pct,100):.0f}%;background:linear-gradient(90deg,#3fa844,#4caf50);"></div>
      <div style="width:{min(draw_pct,100):.0f}%;background:#f4c543;"></div>
      <div style="width:{min(aw_pct,100):.0f}%;background:#e57373;"></div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0;">
      <div style="text-align:left;">
        <div style="display:flex;align-items:center;gap:6px;">
          <span style="width:8px;height:8px;border-radius:50%;background:#4caf50;"></span>
          <span style="font-size:12px;color:#6b766b;font-family:sans-serif;">{home_team}</span>
        </div>
        <div style="font-family:'Oswald',sans-serif;font-size:28px;font-weight:800;color:#2f8f33;margin-top:4px;">{hw_pct:.0f}%</div>
      </div>
      <div style="text-align:center;">
        <div style="display:flex;align-items:center;justify-content:center;gap:6px;">
          <span style="width:8px;height:8px;border-radius:50%;background:#f4c543;"></span>
          <span style="font-size:12px;color:#6b766b;font-family:sans-serif;">Draw</span>
        </div>
        <div style="font-family:'Oswald',sans-serif;font-size:28px;font-weight:800;color:#c79a14;margin-top:4px;">{draw_pct:.0f}%</div>
      </div>
      <div style="text-align:right;">
        <div style="display:flex;align-items:center;justify-content:flex-end;gap:6px;">
          <span style="font-size:12px;color:#6b766b;font-family:sans-serif;">{away_team}</span>
          <span style="width:8px;height:8px;border-radius:50%;background:#e57373;"></span>
        </div>
        <div style="font-family:'Oswald',sans-serif;font-size:28px;font-weight:800;color:#cf5454;margin-top:4px;">{aw_pct:.0f}%</div>
      </div>
    </div>
  </div>
</div>

<script>
let kicked=false;
function kickBall(){{
  if(kicked)return; kicked=true;
  const ball=document.getElementById('ball'),pitch=document.getElementById('pitch'),
        net=document.getElementById('net'),hint=document.getElementById('hint'),
        gt=document.getElementById('gt');
  hint.style.opacity='0'; pitch.style.cursor='default';
  ball.style.animation='shootToNet 0.62s cubic-bezier(0.3,-0.2,0.5,1) forwards';
  setTimeout(()=>{{net.style.animation='netRipple 0.6s cubic-bezier(0.2,0,0,1)';}},560);
  setTimeout(()=>{{gt.style.animation='goalPop 1.4s ease forwards';}},620);
  setTimeout(()=>{{pitch.style.transition='opacity 0.4s,transform 0.4s';pitch.style.opacity='0';pitch.style.transform='translateY(-10px)';}},1900);
  setTimeout(()=>{{pitch.style.display='none';hint.style.display='none';document.getElementById('result').classList.add('result-in');}},2300);
}}
</script>
</body></html>
""", height=680, scrolling=False)


# ══════════════════════════════════════════════════════════════
# PAGE: MATCH PREDICTIONS
# ══════════════════════════════════════════════════════════════
elif page == "🔮  Match Predictions":
    st.markdown("<h1 style='color:#16331a;'>Match Predictions</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6b766b;margin-top:-8px;'>Upcoming WC 2026 matches with win probabilities.</p>", unsafe_allow_html=True)

    preds = data["preds"]
    if preds.empty:
        st.warning("No predictions yet. Run modeltraining.py first.")
        st.stop()

    for _, row in preds.iterrows():
        hw  = float(row.get("home_win_%", 0))
        d   = float(row.get("draw_%", 0))
        aw  = float(row.get("away_win_%", 0))
        predicted = row.get("predicted", "?")
        date      = row.get("date", "")
        winner_color = "#2f8f33" if hw >= aw else "#cf5454"

        st.markdown(f"""
        <div style='background:#fff;border:1px solid #e0e8e0;border-radius:16px;
             box-shadow:0 2px 14px rgba(20,50,20,0.06);padding:22px 26px;margin-bottom:14px;'>
          <div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:16px;'>
            <div>
              <div style='font-family:Oswald;font-size:22px;font-weight:700;color:#16331a;letter-spacing:0.5px;'>
                {row['home_team']} <span style='color:#bbb;font-weight:400;font-size:16px;'>vs</span> {row['away_team']}
              </div>
              <div style='font-size:11px;color:#999;margin-top:3px;'>📅 {date}</div>
            </div>
            <div style='background:#f0f8f0;color:#2f8f33;font-family:Oswald;font-size:11px;
                 letter-spacing:1.5px;padding:5px 14px;border-radius:20px;border:1px solid #b8e0b8;
                 white-space:nowrap;'>
              {predicted}
            </div>
          </div>
          <div style='display:flex;height:10px;border-radius:9999px;overflow:hidden;background:#f0f4f0;margin-bottom:16px;'>
            <div style='width:{min(hw,100):.0f}%;background:linear-gradient(90deg,#3fa844,#4caf50);'></div>
            <div style='width:{min(d,100):.0f}%;background:#f4c543;'></div>
            <div style='width:{min(aw,100):.0f}%;background:#e57373;'></div>
          </div>
          <div style='display:grid;grid-template-columns:1fr 1fr 1fr;'>
            <div style='text-align:left;'>
              <div style='display:flex;align-items:center;gap:5px;'>
                <span style='width:7px;height:7px;border-radius:50%;background:#4caf50;'></span>
                <span style='font-size:11px;color:#888;'>{row['home_team']}</span>
              </div>
              <div style='font-family:Oswald;font-size:24px;font-weight:700;color:#2f8f33;'>{hw:.0f}%</div>
            </div>
            <div style='text-align:center;'>
              <div style='display:flex;align-items:center;justify-content:center;gap:5px;'>
                <span style='width:7px;height:7px;border-radius:50%;background:#f4c543;'></span>
                <span style='font-size:11px;color:#888;'>Draw</span>
              </div>
              <div style='font-family:Oswald;font-size:24px;font-weight:700;color:#c79a14;'>{d:.0f}%</div>
            </div>
            <div style='text-align:right;'>
              <div style='display:flex;align-items:center;justify-content:flex-end;gap:5px;'>
                <span style='font-size:11px;color:#888;'>{row['away_team']}</span>
                <span style='width:7px;height:7px;border-radius:50%;background:#e57373;'></span>
              </div>
              <div style='font-family:Oswald;font-size:24px;font-weight:700;color:#cf5454;'>{aw:.0f}%</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE: WHO WINS THE CUP
# ══════════════════════════════════════════════════════════════
elif page == "🏆  Who Wins the Cup?":
    st.markdown("<h1 style='color:#16331a;'>Who Wins the World Cup?</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6b766b;margin-top:-8px;'>Based on Monte Carlo simulations of the full bracket.</p>", unsafe_allow_html=True)

    sim = data["sim"]
    if sim.empty:
        st.warning("Run phase5_tournament_simulator.py first.")
        st.stop()

    top   = sim[sim["win_prob"] > 0].sort_values("win_prob", ascending=False)
    if top.empty:
        st.warning("No teams with non-zero win probability in simulation_results.csv.")
        st.stop()
    champ = top.iloc[0]

    n_sims_label = int(sim["simulations"].iloc[0]) if "simulations" in sim.columns and not sim.empty else "N"

    # Champion hero card
    st.markdown(f"""
    <div style='position:relative;border-radius:20px;overflow:hidden;background:#0d2b0d;
         box-shadow:0 16px 40px rgba(13,43,13,0.22);padding:36px 28px;margin-bottom:24px;text-align:center;'>
      <div style='position:absolute;inset:0;background:radial-gradient(90% 160% at 50% -30%,rgba(76,175,80,0.4),transparent 60%);'></div>
      <div style='position:relative;'>
        <div style='font-size:11px;letter-spacing:0.24em;color:#7fce82;font-weight:600;'>PREDICTED CHAMPION</div>
        <div style='font-family:Oswald;font-size:56px;font-weight:800;color:#fff;letter-spacing:2px;line-height:1.05;margin-top:8px;'>
          🏆 {champ['team'].upper()}
        </div>
        <div style='display:inline-flex;align-items:center;gap:10px;margin-top:16px;background:rgba(255,255,255,0.08);
             border:1px solid rgba(127,206,130,0.4);border-radius:9999px;padding:8px 18px;'>
          <span style='font-family:Oswald;font-size:22px;font-weight:800;color:#7fce82;line-height:1;'>{champ['win_prob']:.1f}%</span>
          <span style='width:1px;height:16px;background:rgba(127,206,130,0.4);'></span>
          <span style='font-size:11px;letter-spacing:0.14em;color:rgba(255,255,255,0.65);'>WIN PROBABILITY</span>
        </div>
        <div style='font-size:13px;color:rgba(255,255,255,0.55);margin-top:14px;'>
          Across {n_sims_label} simulations of the full bracket
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1.6, 1])
    with c1:
        st.markdown("<div style='background:#fff;border:1px solid #e0e8e0;border-radius:16px;padding:20px;box-shadow:0 2px 14px rgba(20,50,20,0.05);'>", unsafe_allow_html=True)
        st.markdown("#### Championship Odds")
        fig = px.bar(top.head(16), x="win_prob", y="team", orientation="h",
                     color="win_prob", color_continuous_scale=["#b8e0b8","#2f8f33"],
                     labels={"win_prob":"Win %","team":""},
                     text=top.head(16)["win_prob"].map(lambda x: f"{x:.1f}%"))
        fig.update_layout(plot_bgcolor="#fff", paper_bgcolor="#fff", font_color="#16331a",
                          coloraxis_showscale=False, yaxis={"categoryorder":"total ascending"},
                          margin=dict(l=0,r=60,t=10,b=0), height=480)
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div style='background:#fff;border:1px solid #e0e8e0;border-radius:16px;padding:20px;box-shadow:0 2px 14px rgba(20,50,20,0.05);'>", unsafe_allow_html=True)
        st.markdown("#### Full Table")
        display = top[["team","win_prob"]].copy()
        display.columns = ["Team","Win %"]
        display["Win %"] = display["Win %"].map(lambda x: f"{x:.1f}%")
        display.index = range(1, len(display)+1)
        st.dataframe(display, use_container_width=True, height=460)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='background:#fff;border:1px solid #e0e8e0;border-radius:16px;padding:20px;box-shadow:0 2px 14px rgba(20,50,20,0.05);'>", unsafe_allow_html=True)
    st.markdown("#### Probability Map")
    fig2 = px.treemap(top, path=["team"], values="win_prob",
                      color="win_prob", color_continuous_scale=["#e8f5e8","#2f8f33"])
    fig2.update_layout(paper_bgcolor="#fff", font_color="#16331a",
                       margin=dict(t=10,b=0), coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE: TEAM VS TEAM
# ══════════════════════════════════════════════════════════════
elif page == "⚔️  Team vs Team":
    st.markdown("<h1 style='color:#16331a;'>Team vs Team</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6b766b;margin-top:-8px;'>Compare squad quality between any two teams.</p>", unsafe_allow_html=True)

    teams_df = data["teams"]
    if teams_df.empty:
        st.warning("Run phase2_feature_engineering.py first.")
        st.stop()

    # team_features.csv has one row per (team, year) since Phase 2 combines
    # 2026 with historical years (e.g. 2022). Filter to a single year before
    # indexing by team - otherwise teams_df.loc[team] returns a DataFrame
    # (not a Series) for any team appearing in multiple years, which breaks
    # every float(fa.get(...)) call below with a TypeError.
    DASHBOARD_YEAR = 2026
    if "year" in teams_df.columns:
        teams_df = teams_df[teams_df["year"] == DASHBOARD_YEAR].drop(columns=["year"])

    if "team" in teams_df.columns:
        teams_df = teams_df.set_index("team")

    if teams_df.empty:
        st.warning(f"No team_features rows found for year {DASHBOARD_YEAR}.")
        st.stop()

    all_teams = sorted(teams_df.index.unique().tolist())
    c1, c2 = st.columns(2)
    with c1: team_a = st.selectbox("Home team", all_teams, index=0)
    with c2: team_b = st.selectbox("Away team", all_teams, index=min(1, len(all_teams)-1))

    if team_a == team_b:
        st.warning("Pick two different teams!")
        st.stop()

    fa, fb = teams_df.loc[team_a], teams_df.loc[team_b]

    # Real column names from our Phase 2 team_features.csv. No SoFIFA-style
    # overall rating exists in this dataset (Zafronix doesn't provide one),
    # so top_league_ratio (share of squad at a "big 5" league club) and
    # avg_caps (international experience) are used as the closest available
    # squad-quality proxies, matching Phase 3's EDA.
    FEATS = {
        "top_league_ratio": "Top-League Share",
        "avg_caps": "Avg Caps (Experience)",
        "avg_age": "Avg Squad Age",
        "peak_age_count": "Peak Age (24-29)",
        "top_scorer_goals": "Top Scorer Goals",
        "gk_count": "Goalkeepers",
        "def_count": "Defenders",
        "mid_count": "Midfielders",
        "fwd_count": "Forwards",
    }
    # Only include features that exist AND have a non-null value for both
    # selected teams (avg_caps etc. may be NaN if Zafronix never sourced it
    # for a given squad).
    available = {
        k: v for k, v in FEATS.items()
        if k in fa.index and k in fb.index
        and pd.notna(fa.get(k)) and pd.notna(fb.get(k))
    }

    if not available:
        st.warning("No comparable metrics available for these two teams.")
        st.stop()

    labels = list(available.values())
    keys   = list(available.keys())
    vals_a = [float(fa.get(k, 0)) for k in keys]
    vals_b = [float(fb.get(k, 0)) for k in keys]
    maxs   = [max(a, b, 0.01) for a, b in zip(vals_a, vals_b)]
    norm_a = [v/m for v, m in zip(vals_a, maxs)]
    norm_b = [v/m for v, m in zip(vals_b, maxs)]

    # Radar chart in white card
    st.markdown("<div style='background:#fff;border:1px solid #e0e8e0;border-radius:16px;padding:20px;box-shadow:0 2px 14px rgba(20,50,20,0.05);margin-bottom:16px;'>", unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=norm_a+[norm_a[0]], theta=labels+[labels[0]],
                                   fill="toself", name=team_a, line_color="#4caf50",
                                   fillcolor="rgba(76,175,80,0.12)"))
    fig.add_trace(go.Scatterpolar(r=norm_b+[norm_b[0]], theta=labels+[labels[0]],
                                   fill="toself", name=team_b, line_color="#e57373",
                                   fillcolor="rgba(229,115,115,0.12)"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,1],color="#ccc",gridcolor="#eee"),
                                  bgcolor="#fff", angularaxis=dict(color="#888")),
                       paper_bgcolor="#fff", font_color="#16331a",
                       legend=dict(bgcolor="#fff",bordercolor="#eee",borderwidth=1),
                       height=420, margin=dict(t=20,b=20))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Head to head stats
    st.markdown("<div style='background:#fff;border:1px solid #e0e8e0;border-radius:16px;padding:24px 28px;box-shadow:0 2px 14px rgba(20,50,20,0.05);'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;'>
      <div style='font-family:Oswald;font-size:20px;font-weight:700;color:#16331a;'>{team_a}</div>
      <div style='font-size:11px;color:#aaa;letter-spacing:2px;'>HEAD TO HEAD</div>
      <div style='font-family:Oswald;font-size:20px;font-weight:700;color:#16331a;text-align:right;'>{team_b}</div>
    </div>
    """, unsafe_allow_html=True)

    for key, label in available.items():
        va, vb = float(fa.get(key,0)), float(fb.get(key,0))
        total  = max(va+vb, 0.01)
        pct_a  = va/total*100
        w = team_a if va > vb else (team_b if vb > va else None)
        a_col = "#2f8f33" if w == team_a else "#16331a"
        b_col = "#cf5454" if w == team_b else "#16331a"
        st.markdown(f"""
        <div style='margin-bottom:18px;'>
          <div style='display:flex;justify-content:space-between;font-size:13px;margin-bottom:6px;'>
            <span style='font-weight:600;color:{a_col};'>{va:.1f}</span>
            <span style='color:#888;font-size:11px;letter-spacing:1px;'>{label}</span>
            <span style='font-weight:600;color:{b_col};'>{vb:.1f}</span>
          </div>
          <div style='background:#f0f4f0;border-radius:9999px;height:8px;overflow:hidden;'>
            <div style='float:left;width:{pct_a:.0f}%;height:8px;background:#4caf50;border-radius:9999px 0 0 9999px;'></div>
            <div style='float:left;width:{100-pct_a:.0f}%;height:8px;background:#e57373;border-radius:0 9999px 9999px 0;'></div>
          </div>
        </div>""", unsafe_allow_html=True)

    a_wins = sum(1 for k in keys if float(fa.get(k,0)) > float(fb.get(k,0)))
    b_wins = len(keys) - a_wins
    if a_wins > b_wins:
        verdict = f"{team_a} look stronger on paper ({a_wins}/{len(keys)} metrics)"
        verdict_color = "#2f8f33"
    elif b_wins > a_wins:
        verdict = f"{team_b} look stronger on paper ({b_wins}/{len(keys)} metrics)"
        verdict_color = "#cf5454"
    else:
        verdict = "These two teams are very evenly matched!"
        verdict_color = "#c79a14"

    st.markdown(f"""
    <div style='margin-top:8px;padding:14px 20px;background:#f6fbf6;border:1px solid #c8e6c9;
         border-radius:12px;text-align:center;font-family:Oswald;font-size:15px;color:{verdict_color};letter-spacing:0.5px;'>
      {verdict}
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
