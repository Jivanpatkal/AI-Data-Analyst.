import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZER MODULE
# Smart chart generation based on column data types
# All functions return a Plotly figure object
# ══════════════════════════════════════════════════════════════════════════════

# ── Color theme used across all charts ────────────────────────────────────────
CHART_COLORS = [
    "#667eea", "#764ba2", "#f093fb", "#4facfe",
    "#00f2fe", "#43e97b", "#fa709a", "#fee140"
]

CHART_TEMPLATE = "plotly_white"


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY: Detect column types
# ─────────────────────────────────────────────────────────────────────────────
def detect_column_types(df):
    """
    Classify all columns into:
    - numeric_cols   : int or float columns
    - categorical_cols: object/string columns with low cardinality (< 30 unique)
    - datetime_cols  : datetime columns
    - high_cardinality: text columns with too many unique values (skip for charts)

    Returns a dict of column type lists.
    """
    numeric_cols     = df.select_dtypes(include=["number"]).columns.tolist()
    datetime_cols    = df.select_dtypes(include=["datetime"]).columns.tolist()

    categorical_cols = []
    high_cardinality = []

    for col in df.select_dtypes(include=["object", "category"]).columns:
        n_unique = df[col].nunique()
        if n_unique <= 30:
            categorical_cols.append(col)
        else:
            high_cardinality.append(col)

    return {
        "numeric":      numeric_cols,
        "categorical":  categorical_cols,
        "datetime":     datetime_cols,
        "high_card":    high_cardinality
    }


# ─────────────────────────────────────────────────────────────────────────────
# CHART 1: Bar Chart
# ─────────────────────────────────────────────────────────────────────────────
def make_bar_chart(df, x_col, y_col, title=None):
    """
    Create an interactive bar chart.

    Args:
        df    : DataFrame
        x_col : categorical column for X axis
        y_col : numeric column for Y axis
        title : chart title (auto-generated if None)
    """
    # Aggregate: group by x_col, sum y_col
    plot_df = (
        df.groupby(x_col)[y_col]
        .sum()
        .reset_index()
        .sort_values(y_col, ascending=False)
        .head(15)  # Show top 15 for readability
    )

    fig = px.bar(
        plot_df,
        x=x_col,
        y=y_col,
        title=title or f"📊 {y_col.replace('_',' ').title()} by {x_col.replace('_',' ').title()}",
        color=y_col,
        color_continuous_scale=["#667eea", "#764ba2"],
        template=CHART_TEMPLATE,
        text_auto=".2s"
    )

    fig.update_traces(
        textposition="outside",
        marker_line_width=0
    )
    fig.update_layout(
        coloraxis_showscale=False,
        xaxis_tickangle=-30,
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", size=12),
        margin=dict(t=60, b=60, l=40, r=20),
        hoverlabel=dict(bgcolor="white", font_size=13)
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHART 2: Line Chart
# ─────────────────────────────────────────────────────────────────────────────
def make_line_chart(df, x_col, y_cols, title=None):
    """
    Create an interactive multi-line chart.
    Works with datetime OR categorical X axis.

    Args:
        df     : DataFrame
        x_col  : column for X axis (datetime or categorical)
        y_cols : list of numeric columns to plot as lines
        title  : chart title
    """
    # For datetime: sort by date
    if pd.api.types.is_datetime64_any_dtype(df[x_col]):
        plot_df = df.sort_values(x_col)
    else:
        # For categorical: aggregate
        plot_df = df.groupby(x_col)[y_cols].sum().reset_index()

    # Keep only top 50 points for performance
    plot_df = plot_df.head(50)

    fig = go.Figure()

    for i, col in enumerate(y_cols):
        color = CHART_COLORS[i % len(CHART_COLORS)]
        fig.add_trace(go.Scatter(
            x=plot_df[x_col],
            y=plot_df[col],
            mode="lines+markers",
            name=col.replace("_", " ").title(),
            line=dict(color=color, width=2.5),
            marker=dict(size=6),
            hovertemplate=f"<b>{col}</b>: %{{y:,.2f}}<extra></extra>"
        ))

    fig.update_layout(
        title=title or f"📈 Trend Analysis",
        template=CHART_TEMPLATE,
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", size=12),
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=60, b=80, l=40, r=20),
        hovermode="x unified"
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHART 3: Pie / Donut Chart
# ─────────────────────────────────────────────────────────────────────────────
def make_pie_chart(df, label_col, value_col, title=None):
    """
    Create an interactive donut chart showing distribution.

    Args:
        df        : DataFrame
        label_col : categorical column (slice labels)
        value_col : numeric column (slice sizes)
        title     : chart title
    """
    plot_df = (
        df.groupby(label_col)[value_col]
        .sum()
        .reset_index()
        .sort_values(value_col, ascending=False)
        .head(10)   # Max 10 slices
    )

    fig = px.pie(
        plot_df,
        names=label_col,
        values=value_col,
        title=title or f"🥧 {value_col.replace('_',' ').title()} Distribution",
        hole=0.4,   # Donut style
        color_discrete_sequence=CHART_COLORS,
        template=CHART_TEMPLATE
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Value: %{value:,.0f}<br>Share: %{percent}<extra></extra>"
    )
    fig.update_layout(
        font=dict(family="Arial", size=12),
        margin=dict(t=60, b=20, l=20, r=20),
        legend=dict(orientation="v", x=1.05)
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHART 4: Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────
def make_correlation_heatmap(df, title=None):
    """
    Create a correlation heatmap for all numeric columns.
    Only works if DataFrame has 2+ numeric columns.

    Returns None if insufficient numeric columns.
    """
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        return None

    # Cap at 12 columns for readability
    if numeric_df.shape[1] > 12:
        numeric_df = numeric_df.iloc[:, :12]

    corr = numeric_df.corr().round(2)

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale=[
            [0.0,  "#fa709a"],
            [0.5,  "#ffffff"],
            [1.0,  "#667eea"]
        ],
        zmid=0,
        text=corr.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=11),
        hoverongaps=False,
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z}<extra></extra>"
    ))

    fig.update_layout(
        title=title or "🔥 Correlation Heatmap",
        template=CHART_TEMPLATE,
        font=dict(family="Arial", size=11),
        margin=dict(t=60, b=60, l=100, r=40),
        xaxis_tickangle=-30,
        height=450
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHART 5: Distribution Histogram
# ─────────────────────────────────────────────────────────────────────────────
def make_histogram(df, col, title=None):
    """
    Show distribution of a numeric column with KDE overlay.
    """
    fig = px.histogram(
        df,
        x=col,
        nbins=30,
        title=title or f"📉 Distribution of {col.replace('_',' ').title()}",
        color_discrete_sequence=["#667eea"],
        template=CHART_TEMPLATE,
        marginal="box",     # Show box plot on top
        opacity=0.8
    )

    fig.update_layout(
        font=dict(family="Arial", size=12),
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, b=40, l=40, r=20),
        showlegend=False
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SMART DASHBOARD: Auto-select best charts from data
# ─────────────────────────────────────────────────────────────────────────────
def build_smart_dashboard(df):
    """
    Automatically select the best charts based on column types.

    Logic:
    - Has numeric + categorical  → Bar chart
    - Has datetime + numeric     → Line chart
    - Has categorical + numeric  → Pie chart
    - Has 2+ numeric cols        → Correlation heatmap
    - Has any numeric col        → Histogram

    Returns a list of dicts:
        [{"title": str, "fig": plotly_figure, "type": str}]
    """
    col_types = detect_column_types(df)
    charts    = []

    num_cols  = col_types["numeric"]
    cat_cols  = col_types["categorical"]
    dt_cols   = col_types["datetime"]

    # ── Bar chart ─────────────────────────────────────────────────────────────
    if cat_cols and num_cols:
        x = cat_cols[0]
        y = num_cols[0]
        try:
            fig = make_bar_chart(df, x, y)
            charts.append({"title": f"Bar: {y} by {x}", "fig": fig, "type": "bar"})
        except Exception:
            pass

    # ── Line chart ────────────────────────────────────────────────────────────
    if dt_cols and num_cols:
        x = dt_cols[0]
        ys = num_cols[:3]   # Up to 3 lines
        try:
            fig = make_line_chart(df, x, ys)
            charts.append({"title": "Line: Trend over Time", "fig": fig, "type": "line"})
        except Exception:
            pass
    elif cat_cols and len(num_cols) >= 2:
        # Fallback: use categorical X if no datetime
        x = cat_cols[0]
        ys = num_cols[:2]
        try:
            fig = make_line_chart(df, x, ys)
            charts.append({"title": f"Line: {x} Trend", "fig": fig, "type": "line"})
        except Exception:
            pass

    # ── Pie chart ─────────────────────────────────────────────────────────────
    if cat_cols and num_cols:
        label = cat_cols[0]
        val   = num_cols[0]
        try:
            fig = make_pie_chart(df, label, val)
            charts.append({"title": f"Pie: {val} Split", "fig": fig, "type": "pie"})
        except Exception:
            pass

    # ── Correlation heatmap ───────────────────────────────────────────────────
    if len(num_cols) >= 2:
        try:
            fig = make_correlation_heatmap(df)
            if fig:
                charts.append({"title": "Correlation Heatmap", "fig": fig, "type": "heatmap"})
        except Exception:
            pass

    # ── Histogram for top numeric column ──────────────────────────────────────
    if num_cols:
        try:
            fig = make_histogram(df, num_cols[0])
            charts.append({"title": f"Distribution: {num_cols[0]}", "fig": fig, "type": "histogram"})
        except Exception:
            pass

    return charts