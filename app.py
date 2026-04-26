"""
CineMatch — Movie Recommendation System  v3.0
==============================================
v3 Fixes:
  1. Empty ghost box removed — search-card no longer wraps native Streamlit widgets
     in a raw HTML div (caused the visible empty container at top)
  2. Poster fetching upgraded to use TMDB /movie/{id} endpoint (direct, reliable)
     + title-search fallback + beautiful initial-letter SVG placeholder (never broken)
  3. Sidebar collapse fully fixed — zero CSS interference with native toggle
  4. All results persisted in session_state
  5. Movie cards always show an image — no empty spaces
"""

import streamlit as st
import joblib
import pandas as pd
import requests
import os
import urllib.parse
from typing import List, Optional

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch · Movie Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── DATA LOADING ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_similarity():
    return joblib.load("similarity.joblib")

@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv("movie_dataset.csv")
    df.columns = [c.strip().lower() for c in df.columns]
    return df

simi_df    = load_similarity()
ALL_MOVIES: List[str] = list(simi_df.index)
df_meta    = load_dataset()

@st.cache_data(show_spinner=False)
def build_meta_lookup(_df: pd.DataFrame) -> dict:
    """title → row dict, built once and cached."""
    return {
        str(row.get("title", "")).strip(): row.to_dict()
        for _, row in _df.iterrows()
        if str(row.get("title", "")).strip()
    }

META_LOOKUP: dict = build_meta_lookup(df_meta)

# ── BACKEND — unchanged ───────────────────────────────────────────────────────
def recommend(movie_name: str) -> List[str]:
    score  = simi_df[movie_name]
    top11  = score.sort_values(ascending=False).head(11)
    return [m for m in top11.index if m != movie_name][:10]

# ── POSTER FETCHING ───────────────────────────────────────────────────────────
TMDB_KEY      = os.environ.get("TMDB_API_KEY", "688874a314034a394be1ab5fe83a69f5")
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w342"
TMDB_BASE     = "https://api.themoviedb.org/3"


def _make_initials_svg(title: str) -> str:
    """
    Generate a beautiful initial-letter SVG placeholder.
    Uses the first letter of the title so each movie gets a unique look.
    Encoded as data URI — works offline, never a broken image.
    """
    initial = title[0].upper() if title else "?"
    # Pick a deterministic accent shade from the palette
    shades  = ["#C2A56D", "#547A95", "#7a9db5", "#d4b87e", "#4a6a85"]
    color   = shades[ord(initial) % len(shades)]
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="300" height="450" viewBox="0 0 300 450">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#1e2b38"/>
      <stop offset="100%" stop-color="#2C3947"/>
    </linearGradient>
  </defs>
  <rect width="300" height="450" fill="url(#g)"/>
  <rect x="0" y="0" width="300" height="4" fill="{color}" opacity="0.8"/>
  <rect x="0" y="446" width="300" height="4" fill="{color}" opacity="0.4"/>
  <text x="150" y="210" font-size="110" text-anchor="middle"
        dominant-baseline="middle" fill="{color}" opacity="0.25"
        font-family="Georgia,serif" font-weight="bold">{initial}</text>
  <text x="150" y="210" font-size="72" text-anchor="middle"
        dominant-baseline="middle" fill="{color}" opacity="0.9"
        font-family="Georgia,serif" font-weight="bold">{initial}</text>
  <text x="150" y="290" font-size="13" text-anchor="middle"
        fill="#547A95" font-family="sans-serif" letter-spacing="2">NO POSTER</text>
</svg>"""
    encoded = urllib.parse.quote(svg)
    return f"data:image/svg+xml,{encoded}"


@st.cache_data(show_spinner=False, ttl=86_400)
def get_movie_poster(title: str, tmdb_id: Optional[int] = None) -> str:
    """
    Returns a guaranteed-working poster URL.
    Strategy (most → least accurate):
      1. TMDB /movie/{id}  — direct ID lookup, no title mismatch possible
      2. TMDB /search/movie — title + year search
      3. TMDB /search/movie — title only (broader)
      4. Initials SVG placeholder — always works, offline-safe
    """
    if not TMDB_KEY:
        return _make_initials_svg(title)

    headers = {"Accept": "application/json"}

    # Strategy 1: direct by TMDB ID (most accurate)
    if tmdb_id and str(tmdb_id).isdigit():
        try:
            url  = f"{TMDB_BASE}/movie/{int(tmdb_id)}"
            resp = requests.get(url, params={"api_key": TMDB_KEY}, headers=headers, timeout=5)
            if resp.status_code == 200:
                path = resp.json().get("poster_path")
                if path:
                    return f"{TMDB_IMG_BASE}{path}"
        except Exception:
            pass

    # Strategies 2 & 3: search by title (with/without year)
    row  = META_LOOKUP.get(title, {})
    year = str(row.get("release_date", "") or "")[:4]

    search_params = [
        {"api_key": TMDB_KEY, "query": title, **({"year": year} if year.isdigit() else {})},
        {"api_key": TMDB_KEY, "query": title},
    ]
    for params in search_params:
        try:
            resp = requests.get(f"{TMDB_BASE}/search/movie", params=params,
                                headers=headers, timeout=5)
            if resp.status_code == 200:
                for result in resp.json().get("results", []):
                    if result.get("poster_path"):
                        return f"{TMDB_IMG_BASE}{result['poster_path']}"
        except Exception:
            continue

    # Strategy 4: fallback — always renders
    return _make_initials_svg(title)


def get_meta(title: str) -> dict:
    """Null-safe metadata — never raises."""
    row = META_LOOKUP.get(title, {})
    if not row:
        return {"genre": "—", "rating": "—", "year": "—", "tmdb_id": None}

    raw_date = str(row.get("release_date") or "")
    year     = raw_date[:4] if len(raw_date) >= 4 else "—"

    try:
        rating = f"{float(row.get('vote_average', 0)):.1f}"
    except (TypeError, ValueError):
        rating = "—"

    raw_genre = str(row.get("genre") or "—").strip()
    genre     = raw_genre.split(",")[0].strip() if raw_genre != "—" else "—"

    try:
        tmdb_id = int(row.get("id", 0)) or None
    except (TypeError, ValueError):
        tmdb_id = None

    return {"genre": genre, "rating": rating, "year": year, "tmdb_id": tmdb_id}

# ── SESSION STATE ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "recommendations": [],
        "selected_movie":  None,
        "show_results":    False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ── CSS ───────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,600;1,9..40,300&display=swap');

:root {
  --bg:      #2C3947;
  --bg2:     #1e2b38;
  --bg3:     #243040;
  --mid:     #547A95;
  --accent:  #C2A56D;
  --text:    #E8EDF2;
  --dim:     #8fa8bf;
  --radius:  14px;
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stMain"] { background: var(--bg) !important; }

/* Film grain */
[data-testid="stAppViewContainer"]::before {
  content: ''; position: fixed; inset: 0;
  pointer-events: none; z-index: 0; opacity: .018;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
}

/* ── Sidebar ──
   Original: do NOT interfere with native collapse toggle ── */
[data-testid="stSidebar"] {
  background: linear-gradient(175deg, var(--bg3) 0%, #18232f 100%) !important;
  border-right: 1px solid rgba(84,122,149,.2) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Sidebar collapse arrow — make it visible ── */
[data-testid="stSidebarCollapsedControl"] {
  background: #1a2535 !important;
  border-right: 2px solid var(--accent) !important;
}
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="collapsedControl"] button {
  background: var(--accent) !important;
  border-radius: 50% !important;
  color: #1a2535 !important;
  opacity: 1 !important;
  visibility: visible !important;
  box-shadow: 0 0 10px rgba(194,165,109,.5) !important;
}
[data-testid="stSidebarCollapsedControl"] button svg,
[data-testid="collapsedControl"] button svg {
  fill: #1a2535 !important;
  color: #1a2535 !important;
}

.sb-logo {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 2rem; letter-spacing: .1em;
  color: var(--accent) !important;
}
.sb-divider {
  border: none; border-top: 1px solid rgba(84,122,149,.25);
  margin: 1rem 0;
}
.sb-label {
  font-size: .68rem; letter-spacing: .2em; text-transform: uppercase;
  color: var(--accent); font-weight: 700; margin-bottom: .8rem;
}
.sb-step {
  display: flex; gap: .7rem; align-items: flex-start;
  margin-bottom: .8rem; font-size: .86rem;
  color: var(--dim) !important; line-height: 1.5;
}
.sb-step .n {
  background: var(--accent); color: var(--bg3) !important;
  font-weight: 700; border-radius: 50%;
  width: 1.5rem; height: 1.5rem; min-width: 1.5rem;
  display: flex; align-items: center; justify-content: center;
  font-size: .72rem; flex-shrink: 0;
}

/* ── Top accent strip ── */
.top-strip {
  height: 3px;
  background: linear-gradient(90deg, transparent, var(--accent), var(--mid), transparent);
  margin-bottom: 0;
}

/* ── Hero ── */
.hero {
  text-align: center;
  padding: 2.8rem 1rem 2rem;
}
.hero-eyebrow {
  font-size: .72rem; letter-spacing: .28em; text-transform: uppercase;
  color: var(--accent); margin-bottom: .6rem; font-weight: 600;
}
.hero-title {
  font-family: 'Bebas Neue', sans-serif;
  font-size: clamp(3rem, 6.5vw, 5.2rem);
  letter-spacing: .06em; line-height: 1;
  color: var(--text); margin: 0;
}
.hero-title .acc { color: var(--accent); }
.hero-sub {
  margin-top: .8rem; font-size: .98rem;
  color: var(--dim); font-weight: 300; font-style: italic;
}

/* ── Search container — targets the middle column block directly ── */
/* No HTML div wrapper needed. The column provides the centering. */
.search-label {
  font-size: .72rem; letter-spacing: .2em; text-transform: uppercase;
  color: var(--accent); font-weight: 600; margin-bottom: .5rem;
}

/* ── Selectbox styling ── */
[data-testid="stSelectbox"] label { display: none !important; }
[data-testid="stSelectbox"] > div > div {
  background: rgba(44,57,71,.95) !important;
  border: 1.5px solid rgba(84,122,149,.6) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
  font-size: 1rem !important;
  font-family: 'DM Sans', sans-serif !important;
  transition: border-color .2s, box-shadow .2s;
}
[data-testid="stSelectbox"] > div > div:focus-within {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(194,165,109,.15) !important;
}
[data-testid="stSelectbox"] span,
[data-testid="stSelectbox"] input { color: var(--text) !important; }
[data-baseweb="popover"] {
  background: var(--bg2) !important;
  border: 1px solid rgba(84,122,149,.28) !important;
  border-radius: 10px !important;
}
[data-baseweb="menu"] li {
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: .95rem !important;
}
[data-baseweb="menu"] li:hover,
[data-baseweb="menu"] [aria-selected="true"] {
  background: rgba(84,122,149,.22) !important;
}

/* ── Recommend button ── */
.stButton > button {
  width: 100% !important;
  background: linear-gradient(135deg, #c9ae78 0%, #a8853c 100%) !important;
  color: #1a2535 !important;
  font-family: 'Bebas Neue', sans-serif !important;
  font-size: 1.25rem !important;
  letter-spacing: .15em !important;
  border: none !important;
  border-radius: 12px !important;
  padding: .9rem 2rem !important;
  margin-top: .9rem !important;
  box-shadow: 0 4px 22px rgba(194,165,109,.3) !important;
  transition: transform .15s ease, box-shadow .15s ease, filter .15s ease !important;
  cursor: pointer !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 32px rgba(194,165,109,.5) !important;
  filter: brightness(1.06) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Selected movie banner ── */
.sel-banner {
  display: flex; align-items: center; gap: .85rem;
  padding: 1rem 1.5rem; margin-bottom: 1.8rem;
  background: linear-gradient(135deg, rgba(84,122,149,.15), rgba(44,57,71,.5));
  border: 1px solid rgba(194,165,109,.25);
  border-radius: 14px;
}
.sel-banner .lbl {
  font-size: .67rem; letter-spacing: .2em;
  text-transform: uppercase; color: var(--dim);
}
.sel-banner .ttl {
  font-size: 1.1rem; font-weight: 600; color: var(--accent);
  margin-top: .15rem;
}

/* ── Section header ── */
.sec-header {
  display: flex; align-items: center; gap: .75rem; margin-bottom: 1.4rem;
}
.sec-header h2 {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 1.8rem; letter-spacing: .08em; color: var(--text); margin: 0;
}
.pill {
  background: var(--accent); color: var(--bg3);
  font-size: .67rem; font-weight: 700;
  letter-spacing: .12em; text-transform: uppercase;
  padding: .22rem .65rem; border-radius: 999px;
}

/* ── Movie card ── */
.movie-card {
  background: var(--bg2);
  border-radius: var(--radius);
  overflow: hidden;
  border: 1px solid rgba(84,122,149,.15);
  transition: transform .22s cubic-bezier(.22,1,.36,1),
              box-shadow  .22s cubic-bezier(.22,1,.36,1),
              border-color .22s;
  box-shadow: 0 6px 20px rgba(0,0,0,.35);
}
.movie-card:hover {
  transform: translateY(-8px) scale(1.02);
  box-shadow: 0 20px 48px rgba(0,0,0,.6),
              0 0 0 1.5px var(--accent);
  border-color: var(--accent);
}
.movie-card img {
  width: 100%;
  aspect-ratio: 2 / 3;
  object-fit: cover;
  display: block;
  background: var(--bg3);
}
.card-body { padding: .85rem 1rem 1rem; }
.card-rank {
  font-size: .62rem; letter-spacing: .2em; text-transform: uppercase;
  color: var(--accent); font-weight: 700; margin-bottom: .22rem;
}
.card-title {
  font-size: .92rem; font-weight: 600; color: var(--text);
  line-height: 1.35; margin-bottom: .45rem;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.card-meta { display: flex; gap: .36rem; flex-wrap: wrap; }
.chip {
  font-size: .66rem; padding: .16rem .5rem;
  border-radius: 999px; font-weight: 500;
  letter-spacing: .04em; white-space: nowrap;
}
.chip-year  { background: rgba(84,122,149,.2);  color: var(--dim); }
.chip-star  { background: rgba(194,165,109,.18); color: var(--accent); }
.chip-genre { background: rgba(84,122,149,.13);  color: var(--dim);
              max-width: 100px; overflow: hidden; text-overflow: ellipsis; }

/* ── Footer ── */
.footer {
  text-align: center; padding: 2.5rem 1rem 1.5rem;
  color: var(--dim); font-size: .82rem;
  border-top: 1px solid rgba(84,122,149,.12); margin-top: 3rem;
}
.footer .heart { color: var(--accent); }
.footer strong  { color: var(--text); font-weight: 600; }

/* ── Spinner ── */
[data-testid="stSpinner"] > div { border-top-color: var(--accent) !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, [data-testid="stDecoration"] { visibility: hidden !important; }
.block-container { padding-top: 0 !important; }

/* ── Premium Top Header Bar ── */
header[data-testid="stHeader"] {
  background: linear-gradient(90deg, #0d1b2a 0%, #1b2838 50%, #0d1b2a 100%) !important;
  border-bottom: 1px solid rgba(194,165,109,.4) !important;
  box-shadow: 0 2px 20px rgba(0,0,0,.6) !important;
  backdrop-filter: blur(8px) !important;
}

/* ── Status widget (cycling logo) ── */
[data-testid="stStatusWidget"] * {
  color: var(--accent) !important;
  opacity: 1 !important;
  visibility: visible !important;
}
[data-testid="stStatusWidget"] svg {
  fill: var(--accent) !important;
  filter: drop-shadow(0 0 3px rgba(194,165,109,.6)) !important;
}

/* ── Toolbar buttons: Stop, Deploy, menu ── */
[data-testid="stToolbar"] {
  background: transparent !important;
}
[data-testid="stToolbar"] button,
[data-testid="stToolbar"] a {
  color: var(--text) !important;
  opacity: 1 !important;
  visibility: visible !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: .82rem !important;
  font-weight: 500 !important;
  letter-spacing: .04em !important;
  border-radius: 8px !important;
  padding: .3rem .75rem !important;
  transition: all .18s ease !important;
}
[data-testid="stToolbar"] button svg,
[data-testid="stToolbar"] a svg {
  fill: var(--accent) !important;
  opacity: 1 !important;
}
/* Stop button — subtle red pill */
[data-testid="stToolbar"] button:first-of-type {
  background: rgba(220,80,80,.12) !important;
  border: 1px solid rgba(220,80,80,.35) !important;
  color: #f08080 !important;
}
[data-testid="stToolbar"] button:first-of-type:hover {
  background: rgba(220,80,80,.25) !important;
  border-color: #f08080 !important;
  box-shadow: 0 0 12px rgba(220,80,80,.3) !important;
}
/* Deploy button — gold pill */
[data-testid="stToolbar"] a,
[data-testid="stToolbar"] button:last-of-type {
  background: linear-gradient(135deg, rgba(194,165,109,.18), rgba(168,133,60,.18)) !important;
  border: 1px solid rgba(194,165,109,.45) !important;
  color: var(--accent) !important;
}
[data-testid="stToolbar"] a:hover,
[data-testid="stToolbar"] button:last-of-type:hover {
  background: linear-gradient(135deg, rgba(194,165,109,.32), rgba(168,133,60,.32)) !important;
  box-shadow: 0 0 14px rgba(194,165,109,.35) !important;
}

/* ── FIX: remove stVerticalBlock gap ── */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
</style>
""", unsafe_allow_html=True)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="sb-logo">🎬 CineMatch</div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

        # About
        st.markdown('<div class="sb-label">About</div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:.84rem;color:#8fa8bf;line-height:1.65;margin-bottom:.5rem;">'
            'CineMatch is a personalised movie recommendation engine powered by '
            'cosine similarity. Pick any film and instantly discover 10 movies '
            'you\'re likely to love.</p>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

        # How to use
        st.markdown('<div class="sb-label">How to Use</div>', unsafe_allow_html=True)
        for num, icon, text in [
            ("1", "🎥", "Type or select a movie from the dropdown"),
            ("2", "✦",  "Click <b>Get Recommendations</b>"),
            ("3", "🍿",  "Scroll down to see your top 10 picks"),
        ]:
            st.markdown(
                f'<div class="sb-step">'
                f'<span class="n">{num}</span>'
                f'<span>{icon} {text}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

        # Stats
        st.markdown(
            f'<p style="font-size:.78rem;color:#8fa8bf;line-height:1.8;">'
            f'📽 <b style="color:#E8EDF2">{len(ALL_MOVIES):,}</b> movies available<br>'
            f'🤖 Cosine similarity engine<br>'
            f'👨‍💻 Built by <b style="color:#E8EDF2">Mahesh Putta</b>'
            f'</p>',
            unsafe_allow_html=True,
        )


# ── HERO ──────────────────────────────────────────────────────────────────────
def render_hero():
    st.markdown('<div class="top-strip"></div>', unsafe_allow_html=True)
    st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">✦ Personalised for you ✦</div>
  <h1 class="hero-title"><span class="acc">Cine</span>Match</h1>
  <p class="hero-sub">Discover movies you'll love — powered by intelligent similarity matching</p>
</div>""", unsafe_allow_html=True)


# ── SEARCH SECTION ────────────────────────────────────────────────────────────
def render_search():
    """
    Ghost box fix: Never wrap st widgets inside raw HTML divs.
    Instead, use st.columns() to centre the content, and target the
    widget containers with CSS via their data-testid selectors.
    """
    # Centre using columns — no HTML div wrapper needed
    _, mid, _ = st.columns([1, 2.5, 1])

    with mid:
        st.markdown('<p class="search-label">🎥 Choose a movie you enjoy</p>',
                    unsafe_allow_html=True)

        selected = st.selectbox(
            label="movie",
            options=ALL_MOVIES,
            label_visibility="hidden",
            key="movie_select",
            help="Type to search through all movies",
        )

        clicked = st.button("✦  Get Recommendations", use_container_width=True)

    return selected, clicked


# ── MOVIE CARD ────────────────────────────────────────────────────────────────
def render_card(rank: int, title: str):
    """Renders one movie card. Poster is always a valid image."""
    meta   = get_meta(title)
    poster = get_movie_poster(title, tmdb_id=meta["tmdb_id"])

    # Safe HTML encoding
    safe_t = title.replace('"', "&quot;").replace("<", "&lt;")
    # onerror encodes the SVG fallback inline — no broken images
    fallback = _make_initials_svg(title).replace("'", "%27")

    st.markdown(f"""
<div class="movie-card">
  <img src="{poster}"
       alt="{safe_t}"
       loading="lazy"
       onerror="this.onerror=null;this.src='{fallback}';">
  <div class="card-body">
    <div class="card-rank">#{rank} Pick</div>
    <div class="card-title">{title}</div>
    <div class="card-meta">
      <span class="chip chip-year">📅 {meta['year']}</span>
      <span class="chip chip-star">⭐ {meta['rating']}</span>
      <span class="chip chip-genre" title="{meta['genre']}">{meta['genre']}</span>
    </div>
  </div>
</div>""", unsafe_allow_html=True)


# ── RECOMMENDATIONS ───────────────────────────────────────────────────────────
def render_recommendations(source: str, movies: List[str]):
    # "Because you liked" banner
    st.markdown(f"""
<div class="sel-banner">
  <span style="font-size:1.6rem">🎯</span>
  <div>
    <div class="lbl">Because you liked</div>
    <div class="ttl">{source}</div>
  </div>
</div>""", unsafe_allow_html=True)

    # Section header
    st.markdown("""
<div class="sec-header">
  <h2>Top 10 Picks</h2>
  <span class="pill">Recommended for you</span>
</div>""", unsafe_allow_html=True)

    # 5-column grid, 2 rows
    COLS = 5
    for row_start in range(0, len(movies), COLS):
        batch = movies[row_start: row_start + COLS]
        cols  = st.columns(len(batch), gap="small")
        for i, movie in enumerate(batch):
            with cols[i]:
                render_card(row_start + i + 1, movie)


# ── FOOTER ────────────────────────────────────────────────────────────────────
def render_footer():
    st.markdown("""
<div class="footer">
  Built with <span class="heart">♥</span> by <strong>Mahesh Putta</strong>
  &nbsp;·&nbsp; Powered by Streamlit &amp; TMDB
</div>""", unsafe_allow_html=True)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    init_state()
    inject_css()
    render_sidebar()
    render_hero()

    selected, clicked = render_search()

    # Store results in session_state so they persist across Streamlit reruns
    # (sidebar toggles, widget interactions, etc. — FIX 4)
    if clicked and selected:
        with st.spinner("🎬  Finding your perfect matches…"):
            st.session_state.recommendations = recommend(selected)
            st.session_state.selected_movie  = selected
            st.session_state.show_results    = True

    if st.session_state.show_results and st.session_state.recommendations:
        render_recommendations(
            st.session_state.selected_movie,
            st.session_state.recommendations,
        )

    render_footer()


if __name__ == "__main__":
    main()
