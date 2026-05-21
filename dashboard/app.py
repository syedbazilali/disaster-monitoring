import streamlit as st
import pandas as pd
import numpy as np
import json, time, re, os, unicodedata, datetime
from collections import deque
import plotly.graph_objects as go
import plotly.express as px
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_OK = True
except Exception:
    FOLIUM_OK = False

# PAGE CONFIG
st.set_page_config(
    page_title="Disaster Tweet Monitor",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #07090f !important;
    color: #e2e8f0 !important;
    font-family: 'IBM Plex Sans', sans-serif;
}
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1a2332;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* ── metric cards ── */
.mc {
    background: #0d1117;
    border: 1px solid #1a2332;
    border-radius: 10px;
    padding: 13px 16px;
    text-align: center;
}
.mv { font-family:'Space Mono',monospace; font-size:1.7rem; font-weight:700; line-height:1.1; }
.ml { font-size:0.65rem; color:#4b5563; text-transform:uppercase; letter-spacing:.1em; margin-top:3px; }

/* ── tweet feed ── */
.tw {
    background: #0d1117;
    border-left: 3px solid #1a2332;
    border-radius: 0 6px 6px 0;
    padding: 8px 12px;
    margin-bottom: 4px;
    font-size: 0.78rem;
    line-height: 1.5;
}
.tw-d { border-left-color: #ef4444 !important; }
.tw-s { border-left-color: #10b981 !important; }
.pb {
    display:inline-block; font-family:'Space Mono',monospace;
    font-size:.66rem; padding:1px 5px; border-radius:3px; margin-right:5px;
}
.pb-h { background:rgba(239,68,68,.15); color:#ef4444; }
.pb-l { background:rgba(16,185,129,.15); color:#10b981; }

/* ── section label ── */
.sl {
    font-family:'Space Mono',monospace; font-size:.6rem; color:#4b5563;
    text-transform:uppercase; letter-spacing:.12em;
    margin-bottom:8px; padding-bottom:4px;
    border-bottom:1px solid #1a2332;
}

/* ── alert card (rich) ── */
.alert-rich {
    background: linear-gradient(135deg, rgba(239,68,68,.1), rgba(239,68,68,.03));
    border: 1px solid #ef4444;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 8px;
    animation: pulse 1.6s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.7} }

.alert-title {
    font-family:'Space Mono',monospace;
    font-size:.85rem; font-weight:700;
    color:#ef4444; margin-bottom:8px;
}
.alert-grid {
    display:grid; grid-template-columns:1fr 1fr 1fr;
    gap:8px; margin-bottom:8px;
}
.alert-field {
    background:rgba(239,68,68,.08);
    border:1px solid rgba(239,68,68,.2);
    border-radius:6px; padding:7px 10px;
}
.alert-field-label {
    font-family:'Space Mono',monospace;
    font-size:.58rem; color:#9ca3af;
    text-transform:uppercase; letter-spacing:.08em;
    margin-bottom:3px;
}
.alert-field-value {
    font-size:.82rem; font-weight:600; color:#f1f5f9;
}
.alert-tweet {
    font-size:.75rem; color:#9ca3af;
    border-top:1px solid rgba(239,68,68,.15);
    padding-top:7px; margin-top:4px;
    line-height:1.5;
}
.conf-bar-wrap {
    background:rgba(0,0,0,.3); border-radius:3px;
    height:5px; margin-top:4px; overflow:hidden;
}
.conf-bar {
    height:100%; border-radius:3px;
    background:linear-gradient(90deg,#f59e0b,#ef4444);
}

/* ── type badge ── */
.type-badge {
    display:inline-block; border-radius:4px;
    padding:2px 8px; font-size:.7rem;
    font-family:'Space Mono',monospace; font-weight:700;
}
.type-eq   { background:rgba(251,191,36,.15); color:#fbbf24; }
.type-fl   { background:rgba(59,130,246,.15);  color:#3b82f6; }
.type-fi   { background:rgba(249,115,22,.15);  color:#f97316; }
.type-hu   { background:rgba(139,92,246,.15);  color:#8b5cf6; }
.type-ex   { background:rgba(239,68,68,.2);    color:#ef4444; }
.type-ot   { background:rgba(107,114,128,.15); color:#6b7280; }

/* misc */
div[data-testid="stPlotlyChart"] > div { background:transparent !important; }
.stProgress > div > div > div > div { background:#ef4444 !important; }
</style>
""", unsafe_allow_html=True)

# CONSTANTS & PATHS
IS_COLAB = os.path.exists("/content/drive/MyDrive/Thesis_Stage2/distilbert_model")

if IS_COLAB:
    STAGE1         = "/content/drive/MyDrive/Thesis_Stage1"
    STAGE2         = "/content/drive/MyDrive/Thesis_Stage2"
    MODEL_PATH     = os.path.join(STAGE2, "distilbert_model")
    DATA_CSV       = os.path.join(STAGE1, "df_clean.csv")
    SPLIT_JSON     = os.path.join(STAGE1, "split_indices.json")
    THRESH_JSON    = os.path.join(STAGE2, "transformer_results.json")
    GAZETTEER_PATH = "/content/geonames_cities.json"
else:
    # HuggingFace Spaces — all assets are in the repo
    MODEL_PATH     = "model"
    DATA_CSV       = "data/stream_sample.csv"
    SPLIT_JSON     = None
    THRESH_JSON    = "data/transformer_results.json"
    GAZETTEER_PATH = "data/geonames_cities.json"

DTYPE_LABELS = {
    "earthquake": ("Earthquake", "type-eq", "🌍"),
    "flood":      ("Flood",      "type-fl", "🌊"),
    "fire":       ("Wildfire",   "type-fi", "🔥"),
    "hurricane":  ("Hurricane",  "type-hu", "🌀"),
    "explosion":  ("Explosion",  "type-ex", "💥"),
    "other":      ("Other",      "type-ot", "⚠️"),
}

EVENT_TO_TYPE = {
    "earthquake":"earthquake","quake":"earthquake",
    "flood":"flood","floods":"flood",
    "hurricane":"hurricane","cyclone":"hurricane","typhoon":"hurricane",
    "wildfire":"fire","fire":"fire",
    "explosion":"explosion","bombing":"explosion","bombings":"explosion","shooting":"explosion",
}

DISASTER_BLOCKLIST = {
    "hurricane","flood","fire","earthquake","storm","tornado","tsunami",
    "cyclone","typhoon","eruption","landslide","wildfire","drought",
    "avalanche","blizzard","heatwave","explosion","bombing","shooting",
    "harvey","irma","katrina","sandy","maria","dorian","michael",
    "relief","rescue","damage","victims","debris","evacuation",
    "emergency","crisis","disaster","tragedy","death","casualty",
    "injured","missing","trapped","shelter","aid","response",
    "the","and","for","are","but","not","you","all","can","had",
    "her","was","one","our","out","day","get","has","him","his",
}

# RESOURCE LOADING (cached)
@st.cache_resource(show_spinner="Loading DistilBERT model…")
def load_model():
    tok   = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    dev   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev).eval()
    thresh = 0.50
    try:
        with open(THRESH_JSON) as f:
            thresh = float(json.load(f)["distilbert"]["threshold"])
    except Exception:
        pass
    return tok, model, dev, thresh

@st.cache_data(show_spinner="Loading tweet stream…")
def load_data():
    df = pd.read_csv(DATA_CSV)
    if "disaster_type" not in df.columns:
        def infer(e):
            e = str(e).lower()
            for k,v in EVENT_TO_TYPE.items():
                if k in e: return v
            return "other"
        event_col = "event" if "event" in df.columns else None
        df["disaster_type"] = df[event_col].apply(infer) if event_col else "other"
    if SPLIT_JSON and os.path.exists(SPLIT_JSON):
        with open(SPLIT_JSON) as f:
            idx = json.load(f)
        key = "test_idx" if "test_idx" in idx else "test"
        return df.iloc[idx[key]].reset_index(drop=True)
    return df.reset_index(drop=True)

@st.cache_data(show_spinner="Loading gazetteer…")
def load_gazetteer():
    if not os.path.exists(GAZETTEER_PATH):
        return {}
    with open(GAZETTEER_PATH) as f:
        return json.load(f)

# PREPROCESSING  (byte-identical to Stage 1/2)
try:
    import wordsegment; wordsegment.load(); WSEG = True
except Exception:
    WSEG = False
try:
    import emoji as emoji_lib; EMOJI = True
except Exception:
    EMOJI = False

_RT  = re.compile(r"^RT\s+@\w+:\s*")
_URL = re.compile(r"https?://\S+|www\.\S+")
_MEN = re.compile(r"@\w+")
_HTG = re.compile(r"#(\w+)")
_SPC = re.compile(r"\s+")

def preprocess(text):
    if not isinstance(text, str): text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = _RT.sub("", text)
    text = _URL.sub("[URL]", text)
    text = _MEN.sub("@USER", text)
    if WSEG:
        def _ex(m):
            w = wordsegment.segment(m.group(1))
            return " ".join(w) if w else m.group(1)
        text = _HTG.sub(_ex, text)
    else:
        text = _HTG.sub(lambda m: m.group(1), text)
    if EMOJI:
        text = emoji_lib.demojize(text, delimiters=(" ", " "))
    return _SPC.sub(" ", text.lower()).strip()

# CLASSIFICATION
def classify_batch(texts, tok, model, dev, thresh):
    enc = tok(texts, padding=True, truncation=True,
              max_length=128, return_tensors="pt")
    enc = {k: v.to(dev) for k, v in enc.items()
            if k != "token_type_ids"}
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)[:, 1].cpu().numpy()
    return (probs >= thresh).astype(int), probs

# GEOPARSING  (Geoparser A — in-memory GeoNames)
def geoparse(text, gaz):
    tokens = [t for t in text.lower().split()
              if len(t) >= 4 and t not in DISASTER_BLOCKLIST]
    best, best_pop = None, -1
    for n in [3, 2, 1]:
        for i in range(len(tokens) - n + 1):
            cand = " ".join(tokens[i:i+n])
            if cand in gaz:
                e = gaz[cand]
                if n == 1 and e["pop"] < 10000:
                    continue
                if e["pop"] > best_pop:
                    best, best_pop = e, e["pop"]
        if best:
            break
    if best:
        return best["lat"], best["lon"], best.get("name", cand)
    return None, None, None

# DISASTER TYPE INFERENCE FROM WINDOW OF TWEETS
def infer_window_type(tweets_window):
    """Return the most common disaster_type in the current alert window."""
    counts = {}
    for t in tweets_window:
        dtype = t.get("dtype", "other") or "other"
        if dtype != "other":
            counts[dtype] = counts.get(dtype, 0) + 1
    if not counts:
        return "other"
    return max(counts, key=counts.get)

def dominant_location(tweets_window):
    """Return the most frequently mentioned location name in the window."""
    loc_counts = {}
    for t in tweets_window:
        loc = t.get("loc_name")
        if loc:
            loc_counts[loc] = loc_counts.get(loc, 0) + 1
    if not loc_counts:
        return None, None, None
    best_name = max(loc_counts, key=loc_counts.get)
    for t in tweets_window:
        if t.get("loc_name") == best_name:
            return best_name, t.get("lat"), t.get("lon")
    return best_name, None, None

def avg_confidence(tweets_window):
    """Average disaster probability of classified tweets in the window."""
    dis = [t["p"] for t in tweets_window if t["d"]]
    return round(float(np.mean(dis)), 3) if dis else 0.0

# DETECTORS
class CUSUM:
    def __init__(self, drift=0.5, H=5.0, n=50):
        self.drift = drift; self.H = H; self.S = 0.0
        self.buf = deque(maxlen=n)
    def update(self, v):
        self.buf.append(v)
        if len(self.buf) < 10: return self.S, False
        mu = np.mean(list(self.buf)[:-1])
        self.S = max(0.0, self.S + (v - mu - self.drift))
        fired = self.S >= self.H
        if fired: self.S = 0.0
        return round(self.S, 3), fired

class ZScore:
    def __init__(self, k=2.5, n=50):
        self.k = k; self.buf = deque(maxlen=n)
    def update(self, v):
        self.buf.append(v)
        if len(self.buf) < 10: return 0.0, False
        arr = list(self.buf)
        mu = np.mean(arr[:-1]); sigma = np.std(arr[:-1]) + 1e-9
        z = (v - mu) / sigma
        return round(z, 3), z >= self.k

# SESSION STATE
_D = dict(
    running=False, idx=0, tweets=[], geo=[],
    cusum_ser=[], rate_ser=[], zscore_ser=[],
    n=0, n_dis=0, n_geo=0, n_alerts=0,
    alerts=[],         # rich alert dicts
    cusum=None, zs=None, last_alert=-9999,
    window_buf=[],     # last N tweets for alert context
)
for _k, _v in _D.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# SIDEBAR
with st.sidebar:
    st.markdown("<div class='sl'>Stream Controls</div>", unsafe_allow_html=True)

    speed  = st.slider("Tweets per refresh",         5,  120,  25,  5)
    thresh = st.slider("Classification threshold", 0.30, 0.90, 0.50, 0.05,
                       help="Probability above which a tweet is flagged as disaster")
    H_val  = st.slider("CUSUM alert threshold (H)",  2.0, 12.0, 5.0, 0.5,
                       help="Higher = less sensitive CUSUM detector")
    k_val  = st.slider("Z-score sensitivity (k)",    1.5,  4.0, 2.5, 0.25,
                       help="Higher = less sensitive Z-score detector")

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    start  = c1.button("▶ Start", use_container_width=True)
    stop   = c2.button("⏹ Stop",  use_container_width=True)
    reset  = st.button("↺ Reset stream", use_container_width=True)

    if start:
        st.session_state.running = True
        if st.session_state.cusum is None:
            st.session_state.cusum = CUSUM(H=H_val)
            st.session_state.zs    = ZScore(k=k_val)
    if stop:
        st.session_state.running = False
    if reset:
        for _k in list(st.session_state.keys()):
            del st.session_state[_k]
        st.rerun()

    st.markdown("<div class='sl' style='margin-top:18px;'>Model Info</div>",
                unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:.7rem;color:#4b5563;line-height:1.9;'>"
        "<b style='color:#cbd5e1;'>Model</b> &nbsp;DistilBERT-base-uncased<br>"
        "<b style='color:#cbd5e1;'>Corpus</b> &nbsp;CrisisBench + HumAID<br>"
        "<b style='color:#cbd5e1;'>Train set</b> &nbsp;225,589 tweets<br>"
        "<b style='color:#cbd5e1;'>Test macro F1</b> &nbsp;0.880<br>"
        "<b style='color:#cbd5e1;'>Geoparser</b> &nbsp;GeoNames A (in-memory)<br>"
        "<b style='color:#cbd5e1;'>Detectors</b> &nbsp;CUSUM + Z-score"
        "</div>", unsafe_allow_html=True)

# LOAD RESOURCES
tok, model, dev, default_thresh = load_model()
stream_df = load_data()
gaz       = load_gazetteer()
TOTAL     = len(stream_df)

# HEADER
st.markdown(
    "<div style='display:flex;align-items:baseline;gap:14px;margin-bottom:2px;'>"
    "<span style='font-family:Space Mono,monospace;font-size:1.45rem;"
    "font-weight:700;color:#f1f5f9;letter-spacing:-.01em;'>🚨 DISASTER MONITOR</span>"
    "<span style='font-family:Space Mono,monospace;font-size:.6rem;"
    "color:#4b5563;text-transform:uppercase;letter-spacing:.1em;'>"
    "Real-Time Tweet Classification &middot; Thesis Pipeline</span>"
    "</div>"
    "<div style='height:1px;background:linear-gradient(90deg,"
    "#ef4444 0%,#3b82f6 60%,transparent 100%);margin-bottom:16px;'></div>",
    unsafe_allow_html=True)

# TOP METRICS
pct   = round(st.session_state.n / TOTAL * 100, 1) if TOTAL else 0
drate = round(st.session_state.n_dis / st.session_state.n * 100, 1) \
        if st.session_state.n else 0.0

for col, val, lbl, clr in zip(
    st.columns(5),
    [f"{st.session_state.n:,}", f"{drate}%",
     f"{st.session_state.n_geo:,}", f"{st.session_state.n_alerts}", f"{pct}%"],
    ["Tweets processed", "Disaster rate", "Geolocated", "Alerts fired", "Progress"],
    ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#4b5563"]
):
    col.markdown(
        f"<div class='mc'><div class='mv' style='color:{clr};'>{val}</div>"
        f"<div class='ml'>{lbl}</div></div>",
        unsafe_allow_html=True)

st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

# ALERT PANEL (top, full-width, only when active)
alert_zone = st.empty()

# MAIN LAYOUT  — left feed | middle charts | right map+alerts
L, M, R = st.columns([1.0, 1.55, 1.45])

with L:
    st.markdown("<div class='sl'>Live Tweet Feed</div>", unsafe_allow_html=True)
    feed_ph = st.empty()

with M:
    st.markdown("<div class='sl'>CUSUM Detector Statistic</div>", unsafe_allow_html=True)
    cusum_ph = st.empty()
    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sl'>Disaster Posts per Refresh Window</div>", unsafe_allow_html=True)
    rate_ph = st.empty()
    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sl'>Disaster Type Breakdown</div>", unsafe_allow_html=True)
    pie_ph = st.empty()

with R:
    st.markdown("<div class='sl'>Geolocated Disaster Posts</div>", unsafe_allow_html=True)
    map_ph = st.empty()
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sl'>Event Alert Log</div>", unsafe_allow_html=True)
    log_ph = st.empty()

# RENDER HELPERS

def type_badge_html(dtype):
    label, css, icon = DTYPE_LABELS.get(dtype, ("Unknown", "type-ot", "⚠️"))
    return f"<span class='type-badge {css}'>{icon} {label}</span>"

def render_active_alert(alert):
    """Rich full-width banner for the most recent alert."""
    dtype     = alert["dtype"]
    loc       = alert["loc"] or "Unknown location"
    conf      = alert["conf"]
    det       = alert["det"]
    ts        = alert["ts"]
    tweet_txt = alert["tweet"][:120]
    lat, lon  = alert.get("lat"), alert.get("lon")
    label, css, icon = DTYPE_LABELS.get(dtype, ("Other", "type-ot", "⚠️"))
    conf_pct  = int(conf * 100)
    coord_str = f"{lat:.3f}, {lon:.3f}" if lat else "—"

    alert_zone.markdown(
        f"<div class='alert-rich'>"
        f"<div class='alert-title'>⚠ DISASTER EVENT DETECTED &nbsp;·&nbsp; "
        f"Detector: {det} &nbsp;·&nbsp; {ts}</div>"
        f"<div class='alert-grid'>"

        f"<div class='alert-field'>"
        f"<div class='alert-field-label'>Disaster Type</div>"
        f"<div class='alert-field-value'>{icon} {label}</div>"
        f"</div>"

        f"<div class='alert-field'>"
        f"<div class='alert-field-label'>Location</div>"
        f"<div class='alert-field-value'>{loc}</div>"
        f"<div style='font-size:.65rem;color:#6b7280;margin-top:2px;'>{coord_str}</div>"
        f"</div>"

        f"<div class='alert-field'>"
        f"<div class='alert-field-label'>Confidence</div>"
        f"<div class='alert-field-value'>{conf_pct}%</div>"
        f"<div class='conf-bar-wrap'><div class='conf-bar' style='width:{conf_pct}%;'></div></div>"
        f"</div>"

        f"</div>"
        f"<div class='alert-tweet'>\"{tweet_txt}\"</div>"
        f"</div>",
        unsafe_allow_html=True)

def render_feed(tweets):
    html = ""
    for tw in reversed(tweets[-22:]):
        css   = "tw-d" if tw["d"] else "tw-s"
        badge = "pb-h" if tw["d"] else "pb-l"
        txt   = tw["text"][:105].replace("<", "&lt;").replace(">", "&gt;")
        dtype_badge = type_badge_html(tw.get("dtype","other")) if tw["d"] else ""
        html += (f"<div class='tw {css}'>"
                 f"<span class='pb {badge}'>{tw['p']:.2f}</span>"
                 f"{dtype_badge} {txt}</div>")
    feed_ph.markdown(html, unsafe_allow_html=True)

def render_cusum(series, h):
    if not series: return
    y = series[-300:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=y, mode="lines",
        line=dict(color="#ef4444", width=1.6),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.06)"))
    fig.add_hline(y=h, line_dash="dash",
                  line=dict(color="#f59e0b", width=1.2),
                  annotation_text=f"H={h}",
                  annotation_font_color="#f59e0b",
                  annotation_font_size=9)
    fig.update_layout(
        height=170, margin=dict(t=4,b=4,l=2,r=2),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,17,23,0.8)",
        showlegend=False,
        xaxis=dict(showticklabels=False, gridcolor="#1a2332", zeroline=False),
        yaxis=dict(gridcolor="#1a2332", color="#4b5563", tickfont=dict(size=8)))
    cusum_ph.plotly_chart(fig, use_container_width=True,
                          config={"displayModeBar": False})

def render_rate(series):
    if not series: return
    arr = np.array(series[-100:])
    mu  = arr[:-1].mean() if len(arr)>1 else 0
    sig = arr[:-1].std()  if len(arr)>1 else 1
    colors = ["#ef4444" if v > mu+sig else "#3b82f6" for v in arr]
    fig = go.Figure(go.Bar(y=arr, marker_color=colors,
                           marker_line_width=0))
    fig.update_layout(
        height=140, margin=dict(t=2,b=2,l=2,r=2),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,17,23,0.8)",
        showlegend=False,
        xaxis=dict(showticklabels=False, gridcolor="#1a2332"),
        yaxis=dict(gridcolor="#1a2332", color="#4b5563", tickfont=dict(size=8)))
    rate_ph.plotly_chart(fig, use_container_width=True,
                         config={"displayModeBar": False})

def render_pie(tweets):
    dis = [t for t in tweets if t["d"]]
    if not dis: return
    counts = {}
    for t in dis:
        k = t.get("dtype","other") or "other"
        counts[k] = counts.get(k,0)+1
    labels = [DTYPE_LABELS[k][0] if k in DTYPE_LABELS else k for k in counts]
    colors = ["#fbbf24","#3b82f6","#f97316","#8b5cf6","#ef4444","#6b7280"]
    fig = go.Figure(go.Pie(
        labels=labels, values=list(counts.values()), hole=0.55,
        marker=dict(colors=colors[:len(counts)],
                    line=dict(color="#07090f", width=2)),
        textfont=dict(size=9, color="#e2e8f0"),
        textinfo="label+percent"))
    fig.update_layout(
        height=165, margin=dict(t=2,b=2,l=2,r=2),
        paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
    pie_ph.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})

def render_map(geo):
    if not geo:
        map_ph.caption("No geolocated posts yet.")
        return
    recent = geo[-800:]
    if FOLIUM_OK:
        m = folium.Map(location=[20,0], zoom_start=2,
                       tiles="CartoDB dark_matter", prefer_canvas=True)
        for pt in recent:
            clr = "#ef4444" if pt["d"] else "#3b82f6"
            r   = 5 if pt["d"] else 3
            folium.CircleMarker(
                location=[pt["lat"], pt["lon"]], radius=r,
                color=clr, fill=True, fill_opacity=0.65, weight=0,
                popup=f"{pt.get('loc','')}: {pt.get('text','')[:60]}"
            ).add_to(m)
        with map_ph:
            st_folium(m, width=None, height=260, returned_objects=[])
    else:
        pts = pd.DataFrame([{"lat":p["lat"],"lon":p["lon"]} for p in recent])
        map_ph.map(pts)

def render_alert_log(alerts):
    if not alerts:
        log_ph.caption("No alerts yet — stream is quiet.")
        return
    html = ""
    for a in reversed(alerts[-6:]):
        label, css, icon = DTYPE_LABELS.get(a["dtype"], ("Other","type-ot","⚠️"))
        conf_pct = int(a["conf"]*100)
        loc_str  = a["loc"] or "Unknown"
        html += (
            f"<div style='background:#0d1117;border:1px solid #1a2332;"
            f"border-left:3px solid #ef4444;border-radius:0 6px 6px 0;"
            f"padding:9px 12px;margin-bottom:6px;'>"
            f"<div style='display:flex;justify-content:space-between;"
            f"align-items:center;margin-bottom:5px;'>"
            f"<span class='type-badge {css}'>{icon} {label}</span>"
            f"<span style='font-family:Space Mono,monospace;font-size:.6rem;"
            f"color:#4b5563;'>{a['ts']}</span></div>"
            f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:4px;"
            f"font-size:.72rem;'>"
            f"<div><span style='color:#4b5563;'>Location</span><br>"
            f"<b style='color:#cbd5e1;'>{loc_str}</b></div>"
            f"<div><span style='color:#4b5563;'>Confidence</span><br>"
            f"<b style='color:#cbd5e1;'>{conf_pct}%</b>"
            f"<div style='background:#1a2332;border-radius:2px;height:3px;"
            f"margin-top:3px;'><div style='background:linear-gradient(90deg,"
            f"#f59e0b,#ef4444);width:{conf_pct}%;height:100%;border-radius:2px;'>"
            f"</div></div></div></div>"
            f"<div style='font-size:.68rem;color:#6b7280;margin-top:5px;"
            f"border-top:1px solid #1a2332;padding-top:5px;'>"
            f"[{a['det']}] {a['tweet'][:70]}…</div>"
            f"</div>"
        )
    log_ph.markdown(html, unsafe_allow_html=True)

# STREAM PROCESSING LOOP
if st.session_state.running:
    i   = st.session_state.idx
    end = min(i + speed, TOTAL)

    if i >= TOTAL:
        st.session_state.running = False
        st.balloons()
        st.success("✓ Stream complete — all test tweets processed.")
    else:
        batch    = stream_df.iloc[i:end]
        raw_list = batch["text"].fillna("").tolist()
        cln_list = [preprocess(t) for t in raw_list]
        preds, probs = classify_batch(cln_list, tok, model, dev, thresh)

        window_count = 0
        for j, (_, row) in enumerate(batch.iterrows()):
            is_dis = bool(preds[j])
            prob   = float(probs[j])
            raw    = raw_list[j]
            cln    = cln_list[j]
            dtype  = str(row.get("disaster_type", "other"))

            if is_dis:
                window_count += 1
                st.session_state.n_dis += 1

            lat, lon, loc_name = geoparse(cln, gaz)
            if lat is not None:
                st.session_state.n_geo += 1
                st.session_state.geo.append({
                    "lat": lat, "lon": lon, "d": is_dis,
                    "text": raw[:80], "loc": loc_name or ""
                })

            tw = {
                "text": raw, "p": prob, "d": is_dis,
                "dtype": dtype if is_dis else "other",
                "lat": lat, "lon": lon, "loc_name": loc_name,
            }
            st.session_state.tweets.append(tw)
            st.session_state.window_buf.append(tw)
            st.session_state.n += 1

        # keep window_buf at last 200 tweets for alert context
        st.session_state.window_buf = st.session_state.window_buf[-200:]

        # trim main buffers
        if len(st.session_state.tweets) > 350:
            st.session_state.tweets = st.session_state.tweets[-350:]
        if len(st.session_state.geo) > 1500:
            st.session_state.geo = st.session_state.geo[-1500:]

        # init detectors on first run
        if st.session_state.cusum is None:
            st.session_state.cusum = CUSUM(H=H_val)
            st.session_state.zs    = ZScore(k=k_val)

        cs, cf = st.session_state.cusum.update(window_count)
        zv, zf = st.session_state.zs.update(window_count)

        st.session_state.cusum_ser.append(cs)
        st.session_state.rate_ser.append(window_count)
        st.session_state.zscore_ser.append(zv)

        for _key in ["cusum_ser", "rate_ser", "zscore_ser"]:
            if len(st.session_state[_key]) > 500:
                st.session_state[_key] = st.session_state[_key][-500:]

        # ── Fire alert if threshold crossed ──────────────────────────────
        cooldown = 30 * speed
        if (cf or zf) and (i - st.session_state.last_alert > cooldown):
            wb = st.session_state.window_buf

            # infer disaster type, location, confidence from recent window
            dtype_alert = infer_window_type(wb)
            loc_name, lat_a, lon_a = dominant_location(wb)
            conf_alert  = avg_confidence(wb)
            tweet_txt   = next(
                (t["text"] for t in reversed(wb) if t["d"]), raw_list[-1]
            )
            ts_str = datetime.datetime.utcnow().strftime("%H:%M:%S UTC")

            alert = {
                "dtype": dtype_alert,
                "loc":   loc_name,
                "lat":   lat_a,
                "lon":   lon_a,
                "conf":  conf_alert,
                "det":   "CUSUM" if cf else "Z-score",
                "ts":    ts_str,
                "tweet": tweet_txt,
                "n":     st.session_state.n,
            }
            st.session_state.alerts.append(alert)
            st.session_state.n_alerts   += 1
            st.session_state.last_alert  = i

        st.session_state.idx = end

# RENDER ALL PANELS

# active alert banner
if st.session_state.alerts:
    last_alert = st.session_state.alerts[-1]
    gap = st.session_state.n - last_alert["n"]
    if gap < speed * 8:
        render_active_alert(last_alert)
    else:
        alert_zone.empty()
else:
    alert_zone.empty()

render_feed(st.session_state.tweets)
render_cusum(st.session_state.cusum_ser, H_val)
render_rate(st.session_state.rate_ser)
render_pie(st.session_state.tweets)
render_map(st.session_state.geo)
render_alert_log(st.session_state.alerts)

# progress bar
prog = st.session_state.n / TOTAL if TOTAL else 0
st.progress(min(prog, 1.0))
st.caption(
    f"Processed {st.session_state.n:,} / {TOTAL:,} tweets  ·  "
    f"Alerts: {st.session_state.n_alerts}  ·  "
    f"Device: {'CUDA GPU' if torch.cuda.is_available() else 'CPU'}"
)

# auto-rerun while streaming
if st.session_state.running:
    time.sleep(0.04)
    st.rerun()
