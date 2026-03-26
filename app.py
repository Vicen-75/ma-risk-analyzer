# =============================================================================
# app.py
# M&A Risk & Synergy Analyzer - Main Streamlit Application
# BA870/AC820 - Boston University
# Team: Sutikshna Tiwari, Kelvin Nlebemchukwu, Vicente Llinares Llata
# =============================================================================

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from data_fetcher import fetch_financials, manual_input_form
from distress_models import calculate_z_score, calculate_g_score
from manipulation_detector import calculate_m_score
from probability_models import calculate_logit_probability, calculate_overall_risk_score
from synergy_analyzer import calculate_synergy_score

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="M&A Risk & Synergy Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1F3864;
        text-align: center;
        padding: 1rem 0;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #595959;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .footer {
        text-align: center;
        color: #595959;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding: 1rem;
        border-top: 1px solid #e0e0e0;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def color_badge(text: str, color: str) -> str:
    """Returns an HTML colored badge for risk zones."""
    colors = {
        "green":  "#28a745",
        "orange": "#fd7e14",
        "red":    "#dc3545"
    }
    bg = colors.get(color, "#6c757d")
    return f'<span style="background-color:{bg}; color:white; padding:4px 12px; border-radius:20px; font-weight:bold;">{text}</span>'


def plot_risk_gauge(score: float, title: str) -> go.Figure:
    """Creates a gauge chart for the overall risk score (1-10)."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": title, "font": {"size": 16}},
        gauge={
            "axis": {"range": [1, 10]},
            "bar":  {"color": "#1F3864"},
            "steps": [
                {"range": [1, 3.5],  "color": "#d4edda"},
                {"range": [3.5, 6.5], "color": "#fff3cd"},
                {"range": [6.5, 10],  "color": "#f8d7da"}
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": score
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(t=40, b=0, l=20, r=20))
    return fig


def plot_scores_bar(z_score: float, g_score: float, title: str) -> go.Figure:
    """Creates a bar chart comparing Z-Score and G-Score."""
    fig = go.Figure(data=[
        go.Bar(name="Altman Z\"-Score", x=["Z-Score"], y=[z_score],
               marker_color="#1F3864"),
        go.Bar(name="Grover G-Score",  x=["G-Score"], y=[g_score],
               marker_color="#2E75B6")
    ])
    fig.update_layout(
        title=title,
        barmode="group",
        height=300,
        margin=dict(t=40, b=20, l=20, r=20)
    )
    return fig


def display_results(data: dict, prev_data: dict, label: str):
    """
    Runs all models on the provided data and displays results.
    Used for both Single Target and Merger Mode.
    """
    st.subheader(f"📊 Results for {label}")

    # Run all models
    z_result     = calculate_z_score(data)
    g_result     = calculate_g_score(data)
    m_result     = calculate_m_score(data, prev_data)
    logit_result = calculate_logit_probability(data)
    risk_result  = calculate_overall_risk_score(z_result, g_result,
                                                 m_result, logit_result)

    # ── Row 1: Key metrics ────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Altman Z\"-Score", z_result["score"])
        st.markdown(color_badge(z_result["zone"], z_result["color"]),
                    unsafe_allow_html=True)

    with col2:
        st.metric("Grover G-Score", g_result["score"])
        st.markdown(color_badge(g_result["zone"], g_result["color"]),
                    unsafe_allow_html=True)

    with col3:
        st.metric("Beneish M-Score", m_result["m_score"])
        st.markdown(color_badge(m_result["flag"], m_result["color"]),
                    unsafe_allow_html=True)

    with col4:
        st.metric("Bankruptcy Probability", f"{logit_result['probability_pct']}%")
        st.markdown(color_badge(risk_result["zone"], risk_result["color"]),
                    unsafe_allow_html=True)

    st.divider()

    # ── Row 2: Gauge + Bar chart ──────────────────────────────────────────
    col_gauge, col_bar = st.columns(2)

    with col_gauge:
        st.plotly_chart(
            plot_risk_gauge(risk_result["score"], f"Overall Risk Score – {label}"),
            use_container_width=True
        )

    with col_bar:
        st.plotly_chart(
            plot_scores_bar(z_result["score"], g_result["score"],
                            f"Distress Scores – {label}"),
            use_container_width=True
        )

    # ── Row 3: Recommendation ─────────────────────────────────────────────
    st.info(f"💡 **Recommendation:** {risk_result['recommendation']}")
    st.info(f"💰 **Suggested Price Discount:** {risk_result['price_discount']}")

    # ── Row 4: M-Score indices table ──────────────────────────────────────
    with st.expander("🔍 View Beneish M-Score Indices"):
        indices = m_result["indices"]
        df_idx = pd.DataFrame({
            "Index": list(indices.keys()),
            "Value": list(indices.values())
        })
        st.dataframe(df_idx, use_container_width=True)

    return z_result, g_result, m_result, logit_result, risk_result


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f5/Boston_University_seal.svg/200px-Boston_University_seal.svg.png",
             width=80)
    st.title("M&A Risk Analyzer")
    st.caption("BA870/AC820 – Boston University")
    st.divider()

    # Mode selection
    mode = st.radio(
        "Select Analysis Mode",
        ["📋 Single Target Assessment", "🤝 Merger Analysis"],
        index=0
    )

    st.divider()

    # Data input type
    input_type = st.radio(
        "Data Input Method",
        ["🌐 Public Company (Auto-fetch)", "✏️ Private Company (Manual Entry)"],
        index=0
    )

    st.divider()
    st.caption("Powered by WRDS Compustat & yfinance")


# =============================================================================
# MAIN PAGE
# =============================================================================
st.markdown('<div class="main-title">📊 M&A Risk & Synergy Analyzer</div>',
            unsafe_allow_html=True)
st.markdown('<div class="subtitle">Bankruptcy Risk · Earnings Manipulation · Merger Synergy</div>',
            unsafe_allow_html=True)

# =============================================================================
# SINGLE TARGET MODE
# =============================================================================
if "Single Target" in mode:

    st.header("📋 Single Target Assessment")
    st.markdown("Analyze a single company for bankruptcy risk and earnings manipulation.")

    # ── Data Input ────────────────────────────────────────────────────────
    if "Auto-fetch" in input_type:
        ticker = st.text_input("Enter Company Ticker Symbol (e.g. AAPL, MSFT, TSLA)",
                               placeholder="AAPL").upper()

        if st.button("🔍 Fetch & Analyze", type="primary"):
            if ticker:
                with st.spinner(f"Fetching financial data for {ticker}..."):
                    data = fetch_financials(ticker)

                if data:
                    # Store in session state
                    st.session_state["data_single"] = data
                    st.session_state["ticker_single"] = ticker
                    st.success(f"✅ Data loaded for {ticker}")
            else:
                st.warning("Please enter a ticker symbol.")

    else:  # Manual entry
        data = manual_input_form("Target Company")
        prev_data = manual_input_form("Target Company (Previous Year)")

        if st.button("🔍 Run Analysis", type="primary"):
            st.session_state["data_single"] = data
            st.session_state["prev_single"] = prev_data

    # ── Display Results ───────────────────────────────────────────────────
    if "data_single" in st.session_state:
        data      = st.session_state["data_single"]
        prev_data = st.session_state.get("prev_single", data)
        label     = st.session_state.get("ticker_single", "Target Company")

        st.divider()
        display_results(data, prev_data, label)


# =============================================================================
# MERGER MODE
# =============================================================================
elif "Merger" in mode:

    st.header("🤝 Merger Analysis")
    st.markdown("Compare two companies and evaluate their merger synergy potential.")

    col_a, col_b = st.columns(2)

    # ── Company A Input ───────────────────────────────────────────────────
    with col_a:
        st.subheader("🏢 Company A")
        if "Auto-fetch" in input_type:
            ticker_a = st.text_input("Ticker Symbol – Company A",
                                     placeholder="AAPL").upper()
        else:
            data_a      = manual_input_form("Company A")
            prev_data_a = manual_input_form("Company A (Previous Year)")

    # ── Company B Input ───────────────────────────────────────────────────
    with col_b:
        st.subheader("🏢 Company B")
        if "Auto-fetch" in input_type:
            ticker_b = st.text_input("Ticker Symbol – Company B",
                                     placeholder="MSFT").upper()
        else:
            data_b      = manual_input_form("Company B")
            prev_data_b = manual_input_form("Company B (Previous Year)")

    # ── Run Analysis Button ───────────────────────────────────────────────
    if st.button("🔍 Run Merger Analysis", type="primary"):
        if "Auto-fetch" in input_type:
            if ticker_a and ticker_b:
                with st.spinner("Fetching data for both companies..."):
                    data_a = fetch_financials(ticker_a)
                    data_b = fetch_financials(ticker_b)
                    prev_data_a = data_a
                    prev_data_b = data_b

                if data_a and data_b:
                    st.session_state["data_a"]    = data_a
                    st.session_state["data_b"]    = data_b
                    st.session_state["prev_a"]    = prev_data_a
                    st.session_state["prev_b"]    = prev_data_b
                    st.session_state["ticker_a"]  = ticker_a
                    st.session_state["ticker_b"]  = ticker_b
                    st.success(f"✅ Data loaded for {ticker_a} and {ticker_b}")
            else:
                st.warning("Please enter both ticker symbols.")
        else:
            st.session_state["data_a"]   = data_a
            st.session_state["data_b"]   = data_b
            st.session_state["prev_a"]   = prev_data_a
            st.session_state["prev_b"]   = prev_data_b
            st.session_state["ticker_a"] = "Company A"
            st.session_state["ticker_b"] = "Company B"

    # ── Display Merger Results ────────────────────────────────────────────
    if "data_a" in st.session_state and "data_b" in st.session_state:

        data_a    = st.session_state["data_a"]
        data_b    = st.session_state["data_b"]
        prev_a    = st.session_state["prev_a"]
        prev_b    = st.session_state["prev_b"]
        label_a   = st.session_state.get("ticker_a", "Company A")
        label_b   = st.session_state.get("ticker_b", "Company B")

        st.divider()

        # Individual results for each company
        tab1, tab2, tab3 = st.tabs([f"📊 {label_a}", f"📊 {label_b}", "🤝 Synergy Score"])

        with tab1:
            display_results(data_a, prev_a, label_a)

        with tab2:
            display_results(data_b, prev_b, label_b)

        with tab3:
            st.subheader("🤝 Merger Synergy Analysis")

            synergy = calculate_synergy_score(data_a, data_b, prev_a, prev_b)

            # Synergy score display
            col_score, col_interp = st.columns([1, 2])

            with col_score:
                fig_syn = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=synergy["score"],
                    title={"text": "Synergy Score", "font": {"size": 16}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar":  {"color": "#1F3864"},
                        "steps": [
                            {"range": [0, 35],   "color": "#f8d7da"},
                            {"range": [35, 65],  "color": "#fff3cd"},
                            {"range": [65, 100], "color": "#d4edda"}
                        ]
                    }
                ))
                fig_syn.update_layout(height=250,
                                      margin=dict(t=40, b=0, l=20, r=20))
                st.plotly_chart(fig_syn, use_container_width=True)

                st.markdown(
                    color_badge(synergy["category"], synergy["color"]),
                    unsafe_allow_html=True
                )

            with col_interp:
                st.subheader("Interpretation")
                st.write(synergy["interpretation"])

                st.subheader("Score Breakdown")
                components = synergy["components"]
                df_comp = pd.DataFrame({
                    "Component": ["Financial Health", "Ratio Complementarity",
                                  "Size Compatibility", "Penalty (Distress/Manipulation)"],
                    "Points": [components["health_score"],
                               components["complementarity"],
                               components["size_score"],
                               -components["penalty"]]
                })
                st.dataframe(df_comp, use_container_width=True)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("""
    <div class="footer">
        BA870/AC820 Financial & Accounting Analytics · Boston University<br>
        Team: Sutikshna Tiwari · Kelvin Nlebemchukwu · Vicente Llinares Llata<br>
        Data: WRDS Compustat (model training) · yfinance (real-time)
    </div>
""", unsafe_allow_html=True)