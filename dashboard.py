# dashboard.py
# The Streamlit web dashboard — run this to see the live monitoring UI.
#
# HOW TO RUN:
#   streamlit run dashboard.py
#
# Then open your browser at: http://localhost:8501
#
# The dashboard reads from inference_results.csv which inference.py writes to.
# Run inference.py in one terminal and dashboard.py in another.

import os
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from preprocessor import process_dataframe

# ── PATH RESOLUTION ───────────────────────────────────────────────────────────
# Anchor every data file to the folder this script lives in, NOT to whatever
# directory the terminal happens to be in when you run `streamlit run dashboard.py`.
# Without this, dashboard.py and inference.py can silently read/write two
# different inference_results.csv files if launched from different terminals,
# which sends the dashboard into demo mode even while inference.py is running fine.
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH        = os.path.join(BASE_DIR, 'model.pkl')
DATASET_PATH      = os.path.join(BASE_DIR, 'ai4i2020.csv')
RESULTS_PATH      = os.path.join(BASE_DIR, 'inference_results.csv')

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EngineIQ — Live Monitor",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── SUPABASE AUTH ─────────────────────────────────────────────────────────────
# Same project inference.py logs to. This is the public 'anon' key — safe to
# ship client-side, it only grants what Supabase Row Level Security allows.
# The actual access control is the sign-in gate below, not the key itself.
SUPABASE_URL = "https://vpudvhanmyggzimwcgqo.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZwdWR2aGFubXlnZ3ppbXdjZ3FvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODA1MTA2MTQsImV4cCI6MjA5NjA4NjYxNH0.iXIR8-UE5SIaZfAY-Uc6-LaUzYkqfI4LdSY_71V9WQs"

# Where Supabase should send the browser back to after Google auth completes.
# MUST exactly match an entry in Supabase → Authentication → URL
# Configuration → Redirect URLs, or Google sign-in will fail with a
# "redirect_to not allowed" error. Points at the deployed dashboard; switch
# back to "http://localhost:8501" (and re-add that to Redirect URLs) when
# testing locally.
APP_URL = "https://enginemonitor.streamlit.app/"


@st.cache_resource
def get_supabase_client():
    from supabase import create_client
    # create_client() defaults to PKCE for OAuth: sign_in_with_oauth() below
    # generates a code_verifier and holds it in this client's own in-memory
    # storage, and exchange_code_for_session() reads it back on the redirect
    # return. Because this client is @st.cache_resource-cached, it's the
    # SAME object across reruns (needed for that verifier to survive the
    # round trip to Google and back) but also shared across every visitor
    # on this server process — two people mid-Google-sign-in at the exact
    # same moment would clobber each other's verifier. Fine for a small
    # single-user dissertation deployment; would need per-session storage
    # (e.g. backed by st.session_state) for a real multi-user rollout.
    return create_client(SUPABASE_URL, SUPABASE_KEY)


@st.cache_data(ttl=30)
def check_supabase_connection():
    """
    Lightweight reachability check for the System Status sidebar. Cached for
    30s so the dashboard's auto-refresh (every 1-10s) doesn't hit Supabase
    on every single rerun — failures aren't cached (Streamlit re-raises
    without caching on error), so a real outage still shows up promptly.
    """
    get_supabase_client().table("anomaly_logs").select("timestamp").limit(1).execute()


def require_login():
    """
    Gate the entire dashboard behind Supabase auth (email/password or
    Google). Renders a login form and calls st.stop() until sign-in
    succeeds — every line below this call in the script simply never runs
    for a visitor who hasn't authenticated, so there's no code path that
    leaks data pre-login.
    """
    client = get_supabase_client()

    # Landing back from Google: Supabase appends ?code=... to APP_URL.
    code = st.query_params.get('code')
    if code and not st.session_state.get('user_email'):
        try:
            result = client.auth.exchange_code_for_session({"auth_code": code})
            st.session_state['user_email']   = result.user.email
            st.session_state['access_token'] = result.session.access_token
        except Exception as e:
            st.session_state['oauth_error'] = str(e)
        st.query_params.clear()
        st.rerun()

    if st.session_state.get('user_email'):
        return

    st.markdown("# 🔧 EngineIQ")
    st.markdown("*Sign in to view the live engine monitor*")

    with st.form("login_form"):
        email     = st.text_input("Email")
        password  = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

    if submitted:
        try:
            result = client.auth.sign_in_with_password(
                {"email": email, "password": password}
            )
            st.session_state['user_email']   = result.user.email
            st.session_state['access_token'] = result.session.access_token
            st.rerun()
        except Exception as e:
            st.error(f"Login failed: {e}")

    if st.session_state.pop('oauth_error', None):
        st.error("Google sign-in failed. Try again.")

    st.divider()
    oauth = client.auth.sign_in_with_oauth({
        "provider": "google",
        "options": {"redirect_to": APP_URL},
    })
    st.link_button("Continue with Google", oauth.url, use_container_width=True)

    st.stop()


require_login()

# ── CUSTOM STYLING ────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Dark theme */
.stApp { background-color: #0a0a0a; color: #f0f0f0; }
.main .block-container { padding-top: 1rem; padding-bottom: 1rem; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: #111111;
    border: 1px solid #222222;
    padding: 1rem;
    border-radius: 4px;
}

/* Alert banners */
.alert-normal  { background:#0d2b1a; border:1px solid #39ff7a; border-radius:4px; padding:1rem; color:#39ff7a; font-family:monospace; }
.alert-low     { background:#2b2700; border:1px solid #ffb545; border-radius:4px; padding:1rem; color:#ffb545; font-family:monospace; }
.alert-medium  { background:#2b1800; border:1px solid #ff6b35; border-radius:4px; padding:1rem; color:#ff6b35; font-family:monospace; }
.alert-high    { background:#2b0000; border:1px solid #ff3f3f; border-radius:4px; padding:1rem; color:#ff3f3f; font-family:monospace; animation: blink 1s step-end infinite; }
@keyframes blink { 50% { opacity: 0.5; } }

/* Section headers */
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    color: #666;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* Sidebar */
section[data-testid="stSidebar"] { background: #111111; border-right: 1px solid #222; }
</style>
""", unsafe_allow_html=True)


# ── HELPERS ───────────────────────────────────────────────────────────────────
def severity_colour(s):
    return {'NORMAL':'#39ff7a','LOW':'#ffb545','MEDIUM':'#ff6b35','HIGH':'#ff3f3f'}.get(s,'#888')

def classify_severity(score, threshold):
    """
    Bucket a raw anomaly score into severity relative to the sensitivity
    threshold. Mirrors inference.py's get_severity(), but takes the
    threshold as a parameter — inference.py's severity/is_anomaly columns
    are baked in at scoring time against a fixed threshold, so the sidebar
    slider used to only move the chart's dashed line without changing what
    anything downstream (metrics, distribution, history) actually counted
    as an anomaly. Reclassifying here from the raw score is what makes the
    slider real.
    """
    if score > threshold: return 'NORMAL'
    if score > threshold - 0.02: return 'LOW'
    if score > threshold - 0.05: return 'MEDIUM'
    return 'HIGH'

def load_results():
    """Load inference results if available."""
    try:
        df = pd.read_csv(RESULTS_PATH)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except FileNotFoundError:
        return None

def load_dataset_preview():
    """Load a preview of the raw dataset for the data explorer tab."""
    try:
        return pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        return None


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔧 EngineIQ")
    st.markdown("*IoT & ML Predictive Engine Monitor*")
    st.divider()

    st.markdown(f"**Signed in as:** {st.session_state['user_email']}")
    if st.button("Log out"):
        get_supabase_client().auth.sign_out()
        st.session_state.pop('user_email', None)
        st.session_state.pop('access_token', None)
        st.rerun()
    st.divider()

    st.markdown("**Project Info**")
    st.markdown("""
    - **Group:** 11
    - **Dept:** Computer Science, KNUST
    - **Supervisor:** Dr. R.O.M Gyening
    - **Year:** 2025/2026
    """)
    st.divider()

    st.markdown("**Model Settings**")
    threshold = st.slider(
        "Anomaly Threshold",
        min_value=-0.70, max_value=-0.30,
        value=-0.50, step=0.01,
        help="Scores below this are flagged as anomalies"
    )

    refresh_rate = st.selectbox(
        "Auto-refresh interval",
        options=[1, 2, 5, 10],
        index=1,
        format_func=lambda x: f"{x} seconds"
    )

    auto_refresh = st.toggle("Auto-refresh", value=True)
    st.divider()

    st.markdown("**System Status**")
    try:
        joblib.load(MODEL_PATH)
        st.success("✓ Model loaded")
    except:
        st.error("✗ model.pkl not found — run train.py first")

    try:
        pd.read_csv(DATASET_PATH)
        st.success("✓ Dataset found")
    except:
        st.warning("⚠ ai4i2020.csv not found")

    try:
        pd.read_csv(RESULTS_PATH)
        st.success("✓ Inference running")
    except:
        st.info("ℹ Run inference.py to start")

    try:
        check_supabase_connection()
        st.success("✓ Supabase connected")
    except Exception as e:
        st.error(f"✗ Supabase unreachable — {e}")


# ── MAIN CONTENT ──────────────────────────────────────────────────────────────
st.markdown("# 🔧 EngineIQ — Live Engine Health Monitor")
st.markdown(
    "<p style='color:#666;font-size:0.85rem;margin-top:-0.5rem'>"
    "IoT & ML-Based Predictive Car Engine Health Monitoring System — Group 11, KNUST</p>",
    unsafe_allow_html=True
)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📡 Live Monitor",
    "📊 Anomaly History",
    "🗃 Dataset Explorer",
    "ℹ About"
])


# ── TAB 1: LIVE MONITOR ───────────────────────────────────────────────────────
with tab1:
    results_df = load_results()

    if results_df is None or len(results_df) == 0:
        st.info("⏳ Waiting for inference data... Run `python inference.py` in your terminal.")
        
        # Show a demo with fake data so the dashboard looks good.
        # IMPORTANT: samples are shuffled into timestamp order (not grouped
        # normal-then-anomaly) so the "latest" row isn't structurally biased
        # towards always being an anomaly — that was the original bug.
        st.markdown("### Preview (Demo mode — no live data yet)")
        np.random.seed(int(time.time()) % 100)
        demo_scores = np.concatenate([
            np.random.normal(-0.45, 0.03, 80),   # normal   (matches real model scale)
            np.random.normal(-0.58, 0.03, 20),   # anomalies
        ])
        np.random.shuffle(demo_scores)
        demo_df = pd.DataFrame({
            'timestamp': pd.date_range(end=datetime.now(), periods=100, freq='s'),
            'score': demo_scores,
            'actual_failure': np.random.choice([0,1], 100, p=[0.9,0.1])
        })
        results_df = demo_df

    # Reclassify from the raw score against the current sensitivity
    # threshold — applies to both real inference results and demo data, so
    # the sidebar slider actually changes what's flagged, not just where
    # the chart's dashed line is drawn.
    results_df['severity']   = results_df['score'].apply(lambda s: classify_severity(s, threshold))
    results_df['is_anomaly'] = results_df['score'] < threshold

    # ── STATUS BANNER ────────────────────────────────────────────────────────
    latest = results_df.iloc[-1]
    latest_score    = float(latest['score'])
    latest_severity = str(latest['severity'])
    is_anomaly      = bool(latest['is_anomaly'])

    if latest_severity == 'NORMAL':
        st.markdown(f'<div class="alert-normal">● ENGINE STATUS: NORMAL &nbsp;|&nbsp; Score: {latest_score:.4f}</div>', unsafe_allow_html=True)
    elif latest_severity == 'LOW':
        st.markdown(f'<div class="alert-low">⚠ ENGINE STATUS: LOW ANOMALY &nbsp;|&nbsp; Score: {latest_score:.4f} &nbsp;|&nbsp; Monitor closely</div>', unsafe_allow_html=True)
    elif latest_severity == 'MEDIUM':
        st.markdown(f'<div class="alert-medium">⚠ ENGINE STATUS: MEDIUM ANOMALY &nbsp;|&nbsp; Score: {latest_score:.4f} &nbsp;|&nbsp; Recommend inspection</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-high">🔴 ENGINE STATUS: HIGH ANOMALY &nbsp;|&nbsp; Score: {latest_score:.4f} &nbsp;|&nbsp; SEEK MECHANIC IMMEDIATELY</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── METRICS ROW ──────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    total      = len(results_df)
    anomalies  = int(results_df['is_anomaly'].sum())
    fpr_pct    = f"{anomalies/max(total,1)*100:.1f}%"
    high_count = int((results_df['severity'] == 'HIGH').sum())

    col1.metric("Latest Score",     f"{latest_score:.4f}",  help="Lower = more anomalous. Threshold: -0.50")
    col2.metric("Current Status",   latest_severity)
    col3.metric("Anomalies Found",  f"{anomalies}/{total}", help="Windows flagged as anomalous")
    col4.metric("High Severity",    high_count,             help="Windows flagged as HIGH severity")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SCORE CHART ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Anomaly Score — Rolling History</div>', unsafe_allow_html=True)

    last_n = min(100, len(results_df))
    plot_df = results_df.tail(last_n).copy()

    fig = go.Figure()

    # Score line
    fig.add_trace(go.Scatter(
        x=plot_df['timestamp'], y=plot_df['score'],
        mode='lines',
        line=dict(color='#e8ff47', width=1.5),
        name='Anomaly Score',
        fill='tozeroy',
        fillcolor='rgba(232,255,71,0.04)'
    ))

    # Threshold line
    fig.add_hline(
        y=threshold,
        line_dash='dash', line_color='#ff3f3f', line_width=1.5,
        annotation_text=f'Threshold ({threshold})',
        annotation_font_color='#ff3f3f'
    )

    # Mark anomaly points
    anomaly_points = plot_df[plot_df['is_anomaly']]
    if len(anomaly_points) > 0:
        fig.add_trace(go.Scatter(
            x=anomaly_points['timestamp'], y=anomaly_points['score'],
            mode='markers',
            marker=dict(color='#ff3f3f', size=8, symbol='x'),
            name='Anomaly'
        ))

    fig.update_layout(
        paper_bgcolor='#0a0a0a', plot_bgcolor='#111111',
        font=dict(color='#888888', family='monospace'),
        xaxis=dict(gridcolor='#1a1a1a', showgrid=True),
        yaxis=dict(gridcolor='#1a1a1a', showgrid=True, title='Score'),
        legend=dict(bgcolor='#111111', bordercolor='#222222'),
        margin=dict(l=0, r=0, t=20, b=0),
        height=280
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── SEVERITY DISTRIBUTION ────────────────────────────────────────────────
    st.markdown('<div class="section-label">Severity Distribution</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns([1, 2])

    with col_a:
        sev_counts = results_df['severity'].value_counts()
        for sev in ['HIGH', 'MEDIUM', 'LOW', 'NORMAL']:
            count = sev_counts.get(sev, 0)
            pct   = count / max(total, 1) * 100
            colour = severity_colour(sev)
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:6px 0;border-bottom:1px solid #1a1a1a'>"
                f"<span style='color:{colour};font-family:monospace;font-size:0.8rem'>{sev}</span>"
                f"<span style='color:#888;font-size:0.8rem'>{count} ({pct:.0f}%)</span></div>",
                unsafe_allow_html=True
            )

    with col_b:
        fig2 = go.Figure(go.Bar(
            x=[sev_counts.get(s, 0) for s in ['NORMAL','LOW','MEDIUM','HIGH']],
            y=['NORMAL','LOW','MEDIUM','HIGH'],
            orientation='h',
            marker_color=['#39ff7a','#ffb545','#ff6b35','#ff3f3f']
        ))
        fig2.update_layout(
            paper_bgcolor='#0a0a0a', plot_bgcolor='#111111',
            font=dict(color='#888', family='monospace'),
            xaxis=dict(gridcolor='#1a1a1a'),
            yaxis=dict(gridcolor='#1a1a1a'),
            margin=dict(l=0,r=0,t=10,b=0),
            height=160, showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)


# ── TAB 2: ANOMALY HISTORY ────────────────────────────────────────────────────
with tab2:
    results_df2 = load_results()

    if results_df2 is None:
        st.info("No inference results yet. Run `python inference.py` first.")
    else:
        # Same reclassification as Tab 1, so "Anomalies only" / "HIGH only"
        # filtering here respects the sidebar's sensitivity slider too.
        results_df2['severity']   = results_df2['score'].apply(lambda s: classify_severity(s, threshold))
        results_df2['is_anomaly'] = results_df2['score'] < threshold
        st.markdown(f"### Anomaly Event Log ({len(results_df2)} windows scored)")
        
        # Filter options
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            show_only = st.selectbox("Show", ["All windows", "Anomalies only", "HIGH only"])
        with col_f2:
            sort_order = st.selectbox("Sort by", ["Newest first", "Worst score first"])
        
        display_df = results_df2.copy()
        
        if show_only == "Anomalies only":
            display_df = display_df[display_df['is_anomaly'] == True]
        elif show_only == "HIGH only":
            display_df = display_df[display_df['severity'] == 'HIGH']
        
        if sort_order == "Newest first":
            display_df = display_df.sort_values('timestamp', ascending=False)
        else:
            display_df = display_df.sort_values('score', ascending=True)
        
        # Style the table
        def highlight_severity(row):
            colours = {'HIGH':'#2b0000','MEDIUM':'#2b1800','LOW':'#2b2700','NORMAL':''}
            return [f'background-color:{colours.get(row["severity"],"")}'] * len(row)
        
        display_cols = ['timestamp','score','severity','is_anomaly']
        if 'actual_failure' in display_df.columns:
            display_cols.append('actual_failure')
        
        st.dataframe(
            display_df[display_cols].head(200).style.apply(highlight_severity, axis=1),
            use_container_width=True, height=400
        )
        
        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            "⬇ Download as CSV",
            data=csv,
            file_name=f"anomaly_log_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime='text/csv'
        )


# ── TAB 3: DATASET EXPLORER ───────────────────────────────────────────────────
with tab3:
    raw_df = load_dataset_preview()
    
    if raw_df is None:
        st.info("ai4i2020.csv not found. Download it and place it in the project folder.")
    else:
        st.markdown(f"### AI4I 2020 Dataset — {len(raw_df):,} rows, {len(raw_df.columns)} columns")
        
        col_d1, col_d2, col_d3 = st.columns(3)
        col_d1.metric("Total rows",    f"{len(raw_df):,}")
        col_d2.metric("Normal rows",   f"{(raw_df['Machine failure']==0).sum():,}")
        col_d3.metric("Fault rows",    f"{(raw_df['Machine failure']==1).sum():,}")
        
        st.markdown("**Raw data preview (first 100 rows)**")
        st.dataframe(raw_df.head(100), use_container_width=True)
        
        st.markdown("**Column statistics**")
        st.dataframe(raw_df.describe(), use_container_width=True)
        
        # Feature correlation chart
        st.markdown("**Temperature over time (first 500 rows)**")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            y=raw_df['Air temperature [K]'].head(500),
            mode='lines', line=dict(color='#ff6b35', width=1),
            name='Air Temp [K]'
        ))
        fig3.add_trace(go.Scatter(
            y=raw_df['Process temperature [K]'].head(500),
            mode='lines', line=dict(color='#e8ff47', width=1),
            name='Process Temp [K]'
        ))
        fig3.update_layout(
            paper_bgcolor='#0a0a0a', plot_bgcolor='#111111',
            font=dict(color='#888', family='monospace'),
            xaxis=dict(gridcolor='#1a1a1a'),
            yaxis=dict(gridcolor='#1a1a1a', title='Temperature [K]'),
            height=260, margin=dict(l=0,r=0,t=20,b=0)
        )
        st.plotly_chart(fig3, use_container_width=True)


# ── TAB 4: ABOUT ─────────────────────────────────────────────────────────────
with tab4:
    st.markdown("""
    ## IoT & ML-Based Predictive Car Engine Health Monitoring System

    **Group 11 — Department of Computer Science, KNUST, Kumasi**  
    **Supervisor:** Dr. R.O.M Gyening | **Academic Year:** 2025/2026

    ---

    ### What this system does
    This system monitors car engine health in real-time using two sensors:
    - **MPU6050 Accelerometer** — measures engine vibration on X, Y, Z axes at 50Hz
    - **DS18B20 Temperature Sensor** — measures engine surface temperature at 1Hz

    An **Isolation Forest** machine learning model, trained on normal engine operational 
    data, continuously scores incoming sensor readings. When behaviour deviates from 
    normal, an alert is raised before serious damage occurs.

    ---

    ### System Architecture
    ```
    SENSORS → ESP32 (BLE) → Python Inference Engine → Streamlit Dashboard
    MPU6050    Bluetooth LE     Isolation Forest           Alert + Log
    DS18B20    GATT Notify      Butterworth + Features      Supabase DB
    ```

    ---

    ### Technology Stack
    | Layer | Technology |
    |---|---|
    | Hardware | ESP32, MPU6050, DS18B20 |
    | Firmware | C++ / Arduino (ESP32 BLE Arduino) |
    | Communication | Bluetooth Low Energy (BLE) |
    | ML Model | Isolation Forest (scikit-learn) |
    | Processing | Python, NumPy, Pandas, SciPy |
    | Dashboard | Streamlit + Plotly |
    | Database | Supabase (PostgreSQL) |

    ---

    ### Team
    - Iddrisu Faadila Wumpini
    - Lartey-Mensah Emmanuel Tetteh
    """)

# ── AUTO-REFRESH ──────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()