import streamlit as st
import pandas as pd
import time
import os
import plotly.io as pio

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Analyst — Jivan Patkal ",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Load CSS ──────────────────────────────────────────────────────────────────
def load_css():
    css_path = os.path.join("assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ── Session state init ────────────────────────────────────────────────────────
defaults = {
    "df_raw":       None,
    "df_clean":     None,
    "clean_log":    {},
    "analyzed":     False,
    "run_complete": False,
    "output_type":  "",
    "user_prompt":  "",
    "report_text":  "",
    "report_raw":   "",
    "chat_history": [],
    "system_prompt": "",
    "chat_api_key": ""
}
# ── ✅ BULLETPROOF Session state init ─────────────────────────────────────────
defaults = {
    "df_raw": None,
    "df_clean": None,
    "clean_log": {},
    "analyzed": False,
    "run_complete": False,
    "output_type": "",
    "user_prompt": "",
    "report_text": "",
    "report_raw": "",
    "chat_history": [],
    "system_prompt": "",
    "chat_api_key": ""
}

# Force initialization BEFORE anything else runs
for key in defaults:
    if key not in st.session_state:
        st.session_state[key] = defaults[key]

# Extra safety (prevents rare Streamlit race condition)
_ = st.session_state.df_raw


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: HEADER
# ══════════════════════════════════════════════════════════════════════════════
def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Data Analyst</h1>
        <p class="creator-tag">Created by Jeevan Bhatkar</p>
        <p>Upload your data and get instant insights,
           dashboards, and AI-powered reports.</p>
    </div>
    """, unsafe_allow_html=True)

render_header()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: FILE UPLOAD + PREVIEW
# ══════════════════════════════════════════════════════════════════════════════
def load_file(uploaded_file):
    """Load CSV or Excel into DataFrame."""
    try:
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)
        else:
            st.error("❌ Unsupported format. Upload CSV or Excel.")
            return None
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        return None


def render_data_preview(df):
    """Show stats, tabbed preview, and Analyze button."""
    st.markdown("#### 👀 Dataset Preview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📋 Rows",       f"{df.shape[0]:,}")
    c2.metric("📌 Columns",    f"{df.shape[1]}")
    c3.metric("⚠️ Missing",    f"{df.isnull().sum().sum():,}")
    c4.metric("🔁 Duplicates", f"{df.duplicated().sum():,}")

    tab1, tab2, tab3 = st.tabs(["🔍 First 5 Rows", "📊 Column Info", "📈 Quick Stats"])
    with tab1:
        st.dataframe(df.head(), use_container_width=True)
    with tab2:
        info_df = pd.DataFrame({
            "Column":    df.columns,
            "Type":      df.dtypes.values,
            "Non-Null":  df.count().values,
            "Nulls":     df.isnull().sum().values,
            "Null %":    (df.isnull().mean()*100).round(1).astype(str)+"%"
        })
        st.dataframe(info_df, use_container_width=True)
    with tab3:
        num = df.select_dtypes(include="number")
        if not num.empty:
            st.dataframe(num.describe().round(2), use_container_width=True)
        else:
            st.warning("No numeric columns for statistics.")

    st.markdown("---")
    col_btn, _ = st.columns([1, 4])
    with col_btn:
        if st.button("🚀 Analyze Data", type="primary", use_container_width=True):
            st.session_state.analyzed     = True
            st.session_state.run_complete = False
            st.rerun()


def render_upload():
    st.markdown("### 📂 Upload Your Dataset")
    st.markdown("---")

    col_up, col_info = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Choose a CSV or Excel file",
            type=["csv","xlsx","xls"]
        )
    with col_info:
        st.info("**Supported:** CSV · Excel\n\n"
                "**Tip:** First row must be column headers")

    if uploaded:
        df = load_file(uploaded)
        if df is not None:
            st.session_state.df_raw = df
            st.session_state.run_complete = False
            st.success(f"✅ Loaded: **{uploaded.name}**")

    if st.session_state.df_raw is not None:
        render_data_preview(st.session_state.df_raw)

render_upload()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: ANALYSIS OPTIONS + CLEANING
# ══════════════════════════════════════════════════════════════════════════════
from modules.cleaner import clean_dataset

def render_analysis_options():
    """Render config panel. Returns user selections."""
    st.markdown("---")
    st.markdown("### ⚙️ Analysis Configuration")

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown("#### 📊 Output Type")
        output_type = st.radio(
            "What to generate:",
            ["📊 Dashboard Only",
             "📝 Report Only",
             "🚀 Dashboard + Report (Recommended)"],
            index=2
        )
        st.markdown("#### 💬 Custom Requirement")
        user_prompt = st.text_area(
            "What do you want to analyze?",
            placeholder="e.g. Monthly sales trend\n"
                        "e.g. Which product performs best?",
            height=110
        )

    with col_r:
        st.markdown("#### 🧹 Cleaning Options")
        missing_strategy = st.selectbox(
            "Handle missing values:",
            ["auto","median","drop"],
            format_func=lambda x: {
                "auto":   "Auto (Mean / Mode)",
                "median": "Median (skewed data)",
                "drop":   "Drop missing rows"
            }[x]
        )
        handle_out = st.toggle("Cap outliers (IQR method)", value=True)
        st.info("**Always runs:**\n"
                "- ✅ Standardize column names\n"
                "- ✅ Remove duplicates\n"
                "- ✅ Auto-fix data types")

    st.markdown("---")
    cb, _ = st.columns([1, 4])
    with cb:
        run = st.button("▶️ Run Analysis",
                        type="primary",
                        use_container_width=True)

    return run, output_type, missing_strategy, handle_out, user_prompt


def render_cleaning_results(df_clean, clean_log):
    """Show before/after metrics and cleaning log."""
    st.markdown("---")
    st.markdown("### 🧹 Cleaning Results")
    df_raw = st.session_state.df_raw

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Rows Before",    f"{df_raw.shape[0]:,}")
    c2.metric("Rows After",     f"{df_clean.shape[0]:,}",
              delta=f"{df_clean.shape[0]-df_raw.shape[0]:,}",
              delta_color="inverse")
    c3.metric("Missing Before", f"{df_raw.isnull().sum().sum():,}")
    c4.metric("Missing After",  f"{df_clean.isnull().sum().sum():,}",
              delta=f"{df_clean.isnull().sum().sum()-df_raw.isnull().sum().sum():,}",
              delta_color="inverse")

    with st.expander("📋 Detailed Cleaning Log"):
        for step, detail in clean_log.items():
            st.markdown(f"**{step.replace('_',' ').title()}**")
            if isinstance(detail, dict):
                for k,v in detail.items():
                    st.markdown(f"  - `{k}`: {v}")
            elif isinstance(detail, list):
                for item in detail:
                    st.markdown(f"  - {item}")
            else:
                st.markdown(f"  - {detail}")

    st.markdown("#### ✅ Cleaned Dataset Preview")
    st.dataframe(df_clean.head(10), use_container_width=True)

    csv_bytes = df_clean.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Cleaned CSV",
                       csv_bytes, "cleaned_dataset.csv", "text/csv")


if st.session_state.analyzed and st.session_state.df_raw is not None:
    run, output_type, missing_strategy, handle_out, user_prompt = \
        render_analysis_options()

    if run:
        with st.spinner("🧹 Cleaning dataset..."):
            prog = st.progress(0, text="Starting...")
            prog.progress(20, text="Fixing column names...")
            time.sleep(0.2)
            prog.progress(55, text="Handling missing values...")
            time.sleep(0.2)
            prog.progress(85, text="Fixing types & outliers...")

            df_clean, clean_log = clean_dataset(
                st.session_state.df_raw.copy(),
                missing_strategy=missing_strategy,
                handle_outliers_flag=handle_out
            )
            prog.progress(100, text="✅ Done!")
            time.sleep(0.3)
            prog.empty()

        st.session_state.df_clean     = df_clean
        st.session_state.clean_log    = clean_log
        st.session_state.output_type  = output_type
        st.session_state.user_prompt  = user_prompt
        st.session_state.run_complete = True
        # Reset chat on new run
        st.session_state.chat_history  = []
        st.session_state.system_prompt = ""

    if st.session_state.run_complete and st.session_state.df_clean is not None:
        render_cleaning_results(
            st.session_state.df_clean,
            st.session_state.clean_log
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
from modules.visualizer import (
    build_smart_dashboard, make_bar_chart, make_line_chart,
    make_pie_chart, make_correlation_heatmap,
    make_histogram, detect_column_types
)

def render_chart_controls(df):
    """Manual chart builder expander."""
    col_types = detect_column_types(df)
    num_cols  = col_types["numeric"]
    all_cols  = df.columns.tolist()

    with st.expander("🎛️ Custom Chart Builder"):
        cc1,cc2,cc3 = st.columns(3)
        ctype = cc1.selectbox("Chart type",
                              ["Bar","Line","Pie","Histogram","Heatmap"])
        x_col = cc2.selectbox("X axis / Label", all_cols)
        y_col = cc3.selectbox("Y axis / Value",
                              num_cols if num_cols else all_cols)

        if st.button("🔨 Build Chart"):
            try:
                if   ctype == "Bar":       fig = make_bar_chart(df, x_col, y_col)
                elif ctype == "Line":      fig = make_line_chart(df, x_col, [y_col])
                elif ctype == "Pie":       fig = make_pie_chart(df, x_col, y_col)
                elif ctype == "Histogram": fig = make_histogram(df, y_col)
                elif ctype == "Heatmap":   fig = make_correlation_heatmap(df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Chart error: {e}")


def render_dashboard(df):
    st.markdown("---")
    st.markdown("### 📊 Interactive Dashboard")

    with st.spinner("Generating charts..."):
        charts = build_smart_dashboard(df)

    if not charts:
        st.warning("Could not auto-generate charts. Use custom builder below.")
    else:
        st.success(f"✅ {len(charts)} charts generated automatically.")

        for i in range(0, len(charts), 2):
            ca, cb = st.columns(2, gap="medium")
            with ca:
                c = charts[i]
                st.markdown(
                    f"<small style='color:#667eea;font-weight:600'>"
                    f"{c['type'].upper()}</small>",
                    unsafe_allow_html=True
                )
                st.plotly_chart(c["fig"],
                                use_container_width=True,
                                key=f"ch_{i}")
            if i+1 < len(charts):
                with cb:
                    c2 = charts[i+1]
                    st.markdown(
                        f"<small style='color:#667eea;font-weight:600'>"
                        f"{c2['type'].upper()}</small>",
                        unsafe_allow_html=True
                    )
                    st.plotly_chart(c2["fig"],
                                    use_container_width=True,
                                    key=f"ch_{i+1}")

    render_chart_controls(df)

    # Download dashboard as HTML
    if charts:
        st.markdown("---")
        html = ["<html><body style='background:#f8f9ff;font-family:Arial'>"]
        html.append("<h1 style='text-align:center;color:#667eea'>"
                    "📊 AI Data Analyst Dashboard</h1>")
        html.append("<h4 style='text-align:center;color:gray'>"
                    "Created by Jeevan Bhatkar</h4><hr>")
        for c in charts:
            html.append(f"<h3 style='color:#764ba2'>{c['title']}</h3>")
            html.append(pio.to_html(c["fig"], full_html=False,
                                    include_plotlyjs="cdn"))
            html.append("<hr>")
        html.append("</body></html>")
        st.download_button("⬇️ Download Dashboard (HTML)",
                           "\n".join(html).encode("utf-8"),
                           "dashboard.html", "text/html")


if (st.session_state.get("run_complete")
        and st.session_state.df_clean is not None
        and "Dashboard" in st.session_state.get("output_type","")):
    render_dashboard(st.session_state.df_clean)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: AI REPORT
# ══════════════════════════════════════════════════════════════════════════════
from modules.ai_report  import generate_report, format_report_as_text
from modules.downloader import report_to_txt, report_to_pdf

def render_report_section(df, user_requirement):
    st.markdown("---")
    st.markdown("### 🤖 AI-Powered Report")

    with st.expander("🔑 API Key",
                     expanded=not bool(st.session_state.report_text)):
        api_key = st.text_input("Anthropic API Key:",
                                type="password",
                                placeholder="sk-ant-...")
        st.caption("🔒 Never stored — session only.")

    c1, c2 = st.columns([1,2])
    with c1:
        rtype = st.selectbox("Report type:",
                             ["full","summary","recommendations"],
                             format_func=lambda x: {
                                 "full":"📋 Full Report",
                                 "summary":"⚡ Quick Summary",
                                 "recommendations":"🎯 Recommendations"
                             }[x])
    with c2:
        st.info("**Full** = 9 sections  |  "
                "**Summary** = 3 sections  |  "
                "**Recommendations** = action-focused")

    cb, _ = st.columns([1,4])
    with cb:
        gen = st.button("🤖 Generate Report",
                        type="primary",
                        use_container_width=True,
                        disabled=not api_key)

    if not api_key:
        st.warning("⚠️ Enter your Claude API key to generate report.")

    if gen and api_key:
        with st.spinner("🤖 Claude is analysing your data..."):
            try:
                prog = st.progress(0, text="Building context...")
                time.sleep(0.4)
                prog.progress(30, text="Sending to Claude AI...")
                raw  = generate_report(df, api_key, user_requirement, rtype)
                prog.progress(85, text="Formatting...")
                full = format_report_as_text(raw)
                st.session_state.report_raw  = raw
                st.session_state.report_text = full
                st.session_state.chat_api_key = api_key
                prog.progress(100, text="✅ Done!")
                time.sleep(0.3)
                prog.empty()
            except Exception as e:
                st.error(f"❌ {e}")
                st.stop()

    if st.session_state.report_raw:
        st.markdown("---")
        st.markdown("#### 📄 Report")
        for i, section in enumerate(
                st.session_state.report_raw.split("## ")):
            if not section.strip():
                continue
            if i == 0:
                st.markdown(section)
                continue
            lines    = section.split("\n", 1)
            s_title  = lines[0].strip()
            s_body   = lines[1].strip() if len(lines)>1 else ""
            with st.expander(f"📌 {s_title}", expanded=(i<=2)):
                st.markdown(s_body)

        st.markdown("---")
        st.markdown("#### ⬇️ Download Report")
        d1, d2, d3 = st.columns(3)
        with d1:
            st.download_button("📄 TXT",
                               report_to_txt(st.session_state.report_text),
                               "ai_report.txt", "text/plain",
                               use_container_width=True)
        with d2:
            try:
                st.download_button("📕 PDF",
                                   report_to_pdf(st.session_state.report_text),
                                   "ai_report.pdf", "application/pdf",
                                   use_container_width=True)
            except Exception as e:
                st.warning(f"PDF error: {e}")
        with d3:
            st.download_button("📝 Markdown",
                               st.session_state.report_raw.encode("utf-8"),
                               "ai_report.md", "text/markdown",
                               use_container_width=True)


if (st.session_state.get("run_complete")
        and st.session_state.df_clean is not None
        and "Report" in st.session_state.get("output_type","")):
    render_report_section(
        st.session_state.df_clean,
        st.session_state.get("user_prompt","")
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: ASK YOUR DATA CHATBOT
# ══════════════════════════════════════════════════════════════════════════════
from modules.chatbot import (
    build_system_prompt,
    get_suggested_questions,
    chat_with_data,
    format_chat_history_as_text
)

def render_chatbot(df):
    """Full chatbot UI with history, suggested questions, and export."""
    st.markdown("---")
    st.markdown("### 💬 Ask Your Data")
    st.caption("Chat with your dataset using natural language.")

    # ── API key for chatbot ───────────────────────────────────────────────────
    with st.expander("🔑 Chatbot API Key",
                     expanded=not bool(st.session_state.chat_api_key)):
        chat_key = st.text_input(
            "Anthropic API Key for chatbot:",
            type="password",
            placeholder="sk-ant-...",
            value=st.session_state.chat_api_key,
            key="chat_key_input"
        )
        if chat_key:
            st.session_state.chat_api_key = chat_key
        st.caption("🔒 If you already generated a report, your key is pre-filled.")

    if not st.session_state.chat_api_key:
        st.warning("⚠️ Enter your API key above to start chatting.")
        return

    # ── Build system prompt once per dataset ─────────────────────────────────
    if not st.session_state.system_prompt:
        with st.spinner("Initialising AI analyst..."):
            st.session_state.system_prompt = build_system_prompt(df)

    # ── Suggested questions ───────────────────────────────────────────────────
    st.markdown("**💡 Suggested Questions:**")
    suggestions = get_suggested_questions(df)

    # Display as clickable pills in rows of 3
    for i in range(0, len(suggestions), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(suggestions):
                if col.button(suggestions[idx],
                              key=f"sug_{idx}",
                              use_container_width=True):
                    # Add to history and trigger response
                    st.session_state.chat_history.append(
                        {"role": "user", "content": suggestions[idx]}
                    )
                    with st.spinner("🤖 Thinking..."):
                        reply = chat_with_data(
                            st.session_state.chat_history,
                            st.session_state.chat_api_key,
                            st.session_state.system_prompt
                        )
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": reply}
                    )
                    st.rerun()

    st.markdown("---")

    # ── Chat history display ──────────────────────────────────────────────────
    if st.session_state.chat_history:
        st.markdown("**💬 Conversation:**")
        chat_container = st.container()

        with chat_container:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    with st.chat_message("user"):
                        st.markdown(msg["content"])
                else:
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(msg["content"])
    else:
        st.info("👆 Click a suggested question above or type below to start.")

    # ── Chat input ────────────────────────────────────────────────────────────
    user_input = st.chat_input(
        "Ask anything about your data...",
        key="chat_input"
    )

    if user_input:
        # Append user message
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input}
        )

        # Get AI response
        with st.spinner("🤖 Analysing..."):
            try:
                reply = chat_with_data(
                    st.session_state.chat_history,
                    st.session_state.chat_api_key,
                    st.session_state.system_prompt
                )
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": reply}
                )
            except Exception as e:
                st.error(f"❌ Chat error: {e}")

        st.rerun()

    # ── Chat controls ─────────────────────────────────────────────────────────
    if st.session_state.chat_history:
        st.markdown("---")
        ctrl1, ctrl2, ctrl3 = st.columns(3)

        with ctrl1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        with ctrl2:
            history_txt = format_chat_history_as_text(
                st.session_state.chat_history
            )
            st.download_button(
                "⬇️ Export Chat (TXT)",
                history_txt.encode("utf-8"),
                "chat_history.txt",
                "text/plain",
                use_container_width=True
            )

        with ctrl3:
            turns = len(st.session_state.chat_history) // 2
            st.metric("💬 Turns", turns)


# ── Render chatbot if data is ready ───────────────────────────────────────────
if (st.session_state.get("run_complete")
        and st.session_state.df_clean is not None):
    render_chatbot(st.session_state.df_clean)


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:gray;font-size:0.85rem;padding:1rem'>"
    "🤖 <b>AI Data Analyst</b> &nbsp;|&nbsp; "
    "Built by <b>Jeevan Bhatkar</b> &nbsp;|&nbsp; "
    "Python · Pandas · Plotly · Claude AI"
    "</div>",
    unsafe_allow_html=True
)
