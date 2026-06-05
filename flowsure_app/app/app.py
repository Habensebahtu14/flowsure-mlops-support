from pathlib import Path

import pandas as pd
import plotly.express as px   # pyright: ignore[reportMissingImports]
import plotly.graph_objects as go  # pyright: ignore[reportMissingImports]
import streamlit as st

from classifier import IntentClassifier

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FlowSure Intent Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "baseline_model.onnx"
MAPPING_PATH = BASE_DIR / "models" / "intent_mapping.json"

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Base */
    [data-testid="stAppViewContainer"] { background: #F8F9FB; }
    [data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E5E7EB; }

    /* Cards */
    .card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 1px 4px rgba(0,0,0,.06), 0 4px 16px rgba(0,0,0,.04);
        margin-bottom: 16px;
    }
    .card-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: .08em;
        text-transform: uppercase;
        color: #9CA3AF;
        margin-bottom: 6px;
    }
    .card-value {
        font-size: 22px;
        font-weight: 700;
        color: #111827;
        line-height: 1.2;
    }
    .card-sub {
        font-size: 13px;
        color: #6B7280;
        margin-top: 4px;
    }

    /* Priority badges */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
    }
    .badge-high   { background: #FEE2E2; color: #DC2626; }
    .badge-medium { background: #FEF3C7; color: #D97706; }
    .badge-low    { background: #DCFCE7; color: #16A34A; }

    /* Inference time pill */
    .time-pill {
        display: inline-block;
        background: #EFF6FF;
        color: #3B82F6;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }

    /* Header */
    .app-header {
        padding: 8px 0 24px 0;
    }
    .app-title {
        font-size: 28px;
        font-weight: 800;
        color: #111827;
        margin: 0;
    }
    .app-subtitle {
        font-size: 15px;
        color: #6B7280;
        margin-top: 4px;
    }

    /* Example chips */
    div[data-testid="stButton"] > button {
        border-radius: 20px;
        font-size: 13px;
        padding: 4px 14px;
        border: 1px solid #E5E7EB;
        background: #FFFFFF;
        color: #374151;
        transition: all .15s;
    }
    div[data-testid="stButton"] > button:hover {
        border-color: #3B82F6;
        color: #3B82F6;
        background: #EFF6FF;
    }

    /* Sidebar stats */
    .stat-row {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid #F3F4F6;
        font-size: 13px;
        color: #374151;
    }
    .stat-row span { font-weight: 600; color: #111827; }

    /* Alert / warning messages */
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] span {
        color: #111827 !important;
        font-weight: 500 !important;
        font-size: 14px !important;
    }

    /* Similar Tickets expander headers */
    [data-testid="stExpander"] summary {
        background: #F3F4F6;
        border-radius: 8px;
        padding: 10px 14px;
    }
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span {
        color: #111827 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    [data-testid="stExpander"] summary:hover {
        background: #E5E7EB;
    }

    /* Hide streamlit branding */
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Load model (cached) ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_classifier() -> IntentClassifier:
    return IntentClassifier(str(MODEL_PATH), str(MAPPING_PATH))


clf = load_classifier()

# ── Session state ──────────────────────────────────────────────────────────────
if "inference_times" not in st.session_state:
    st.session_state.inference_times = []
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ FlowSure")
    st.markdown("**AI Customer Support Suite**")
    st.divider()

    st.markdown("#### Model Information")
    model_size = MODEL_PATH.stat().st_size / (1024 * 1024)
    for label, value in [
        ("Type", "Sklearn Pipeline (ONNX)"),
        ("Size", f"{model_size:.2f} MB"),
        ("Intents", str(clf.num_intents)),
        ("Runtime", "ONNX Runtime"),
    ]:
        st.markdown(
            f'<div class="stat-row">{label}<span>{value}</span></div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("#### Performance")
    if st.session_state.inference_times:
        avg_ms = sum(st.session_state.inference_times) / len(st.session_state.inference_times)
        st.markdown(
            f'<div class="stat-row">Avg inference<span>{avg_ms:.1f} ms</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="stat-row">Predictions<span>{len(st.session_state.inference_times)}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("No predictions yet.")

    st.divider()
    st.caption("FlowSure MLOps · v1.0.0")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="app-header">
        <p class="app-title">🛡️ FlowSure Intent Classifier</p>
        <p class="app-subtitle">
            AI-Powered Customer Intent Classifier —
            automatically routes customer messages to the right team.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🎯 Classify Ticket", "📊 Batch Classification"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single classification
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    EXAMPLES = [
        "I want to cancel my insurance policy immediately.",
        "My payment was declined, what should I do?",
        "Where is my refund? It has been 2 weeks.",
        "I need to update my home address for shipping.",
        "I would like to file a formal complaint.",
    ]

    st.markdown("##### Quick examples")
    cols = st.columns(len(EXAMPLES))
    for i, (col, example) in enumerate(zip(cols, EXAMPLES)):
        with col:
            if st.button(example[:35] + "…", key=f"ex_{i}"):
                st.session_state.input_text = example

    st.markdown("")

    left, right = st.columns([3, 2], gap="large")

    with left:
        text_input = st.text_area(
            "Customer message",
            value=st.session_state.input_text,
            height=140,
            placeholder="Type or paste a customer message…",
            label_visibility="collapsed",
        )
        classify_btn = st.button("Classify →", type="primary", use_container_width=True)

    # ── Results ────────────────────────────────────────────────────────────────
    if classify_btn:
        if not text_input.strip():
            st.warning("Please enter a customer message before classifying.")
        else:
            with st.spinner("Running inference…"):
                result = clf.predict(text_input)
            st.session_state.inference_times.append(result["inference_time_ms"])

            priority = result["priority"]
            badge_cls = f"badge-{priority}"

            with left:
                # Confidence bar
                conf_pct = result["confidence"] * 100
                st.markdown("**Confidence**")
                st.progress(result["confidence"])
                st.caption(f"{conf_pct:.1f}% confident · "
                           f'<span class="time-pill">⚡ {result["inference_time_ms"]} ms</span>',
                           unsafe_allow_html=True)

                # Top 3
                st.markdown("**Top 3 predictions**")
                for item in result["top_3"]:
                    pct = item["confidence"] * 100
                    st.markdown(
                        f"`{item['intent'].replace('_', ' ').title()}` — {pct:.1f}%"
                    )
                    st.progress(item["confidence"])

            with right:
                # Intent card
                st.markdown(
                    f"""
                    <div class="card">
                        <div class="card-label">Predicted Intent</div>
                        <div class="card-value">{result['intent'].replace('_', ' ').title()}</div>
                        <div class="card-sub">{result['intent']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                # Category card
                st.markdown(
                    f"""
                    <div class="card">
                        <div class="card-label">Category</div>
                        <div class="card-value">{result['category']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                # Priority card
                st.markdown(
                    f"""
                    <div class="card">
                        <div class="card-label">Priority</div>
                        <div class="card-value">
                            <span class="badge {badge_cls}">{priority.upper()}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch classification
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("Upload a CSV with a **`text`** column, or paste messages (one per line).")

    col_upload, col_paste = st.columns(2, gap="large")

    with col_upload:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    with col_paste:
        pasted = st.text_area(
            "Or paste messages (one per line)",
            height=160,
            placeholder="I want to cancel my policy.\nWhere is my refund?\n…",
        )

    run_batch = st.button("Run Batch Classification →", type="primary")

    if run_batch:
        texts: list[str] = []

        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                if "text" not in df_upload.columns:
                    st.error("CSV must contain a column named **`text`**.")
                    st.stop()
                texts = df_upload["text"].dropna().astype(str).tolist()
            except Exception as e:
                st.error(f"Could not parse CSV: {e}")
                st.stop()
        elif pasted.strip():
            texts = [line.strip() for line in pasted.strip().splitlines() if line.strip()]
        else:
            st.warning("Provide a CSV file or paste some messages.")
            st.stop()

        if not texts:
            st.warning("No valid texts found.")
            st.stop()

        with st.spinner(f"Classifying {len(texts)} messages…"):
            results = clf.predict_batch(texts)
            st.session_state.inference_times.extend(
                [r["inference_time_ms"] for r in results]
            )

        df = pd.DataFrame(
            [
                {
                    "Text": t,
                    "Intent": r["intent"],
                    "Category": r["category"],
                    "Priority": r["priority"],
                    "Confidence": round(r["confidence"], 4),
                    "Inference (ms)": r["inference_time_ms"],
                }
                for t, r in zip(texts, results)
            ]
        )

        st.success(f"Classified **{len(df)}** messages.")

        # ── Summary stats ──────────────────────────────────────────────────────
        avg_conf = df["Confidence"].mean()
        s1, s2, s3 = st.columns(3)
        s1.metric("Total Messages", len(df))
        s2.metric("Avg Confidence", f"{avg_conf*100:.1f}%")
        s3.metric("Unique Intents", df["Intent"].nunique())

        st.divider()

        chart_col1, chart_col2 = st.columns(2, gap="large")

        with chart_col1:
            cat_counts = df["Category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]
            fig_bar = px.bar(
                cat_counts,
                x="Category",
                y="Count",
                title="Distribution by Category",
                color="Count",
                color_continuous_scale="Blues",
                template="plotly_white",
            )
            fig_bar.update_layout(
                showlegend=False,
                coloraxis_showscale=False,
                margin=dict(t=40, b=0, l=0, r=0),
                title_font_size=14,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with chart_col2:
            pri_counts = df["Priority"].value_counts().reset_index()
            pri_counts.columns = ["Priority", "Count"]
            color_map = {"high": "#FF4B4B", "medium": "#FFA726", "low": "#66BB6A"}
            fig_donut = px.pie(
                pri_counts,
                names="Priority",
                values="Count",
                title="Distribution by Priority",
                color="Priority",
                color_discrete_map=color_map,
                hole=0.55,
                template="plotly_white",
            )
            fig_donut.update_layout(
                margin=dict(t=40, b=0, l=0, r=0),
                title_font_size=14,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        st.divider()

        # ── Results table ──────────────────────────────────────────────────────
        st.markdown("#### Results")

        def _priority_color(val: str) -> str:
            return {
                "high": "background-color:#FEE2E2; color:#DC2626; font-weight:600",
                "medium": "background-color:#FEF3C7; color:#D97706; font-weight:600",
                "low": "background-color:#DCFCE7; color:#16A34A; font-weight:600",
            }.get(val, "")

        styled = df.style.map(_priority_color, subset=["Priority"])
        st.dataframe(styled, use_container_width=True, height=320)

        # ── Download ───────────────────────────────────────────────────────────
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download CSV",
            data=csv_bytes,
            file_name="flowsure_classifications.csv",
            mime="text/csv",
        )
