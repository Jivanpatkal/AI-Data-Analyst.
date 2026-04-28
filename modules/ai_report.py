import anthropic
import pandas as pd
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# AI REPORT MODULE
# Uses Claude API to generate intelligent business reports
# based on dataset statistics and user requirements
# ══════════════════════════════════════════════════════════════════════════════


def build_data_summary(df):
    """
    Build a compact text summary of the dataset
    to pass as context to the AI model.

    Includes:
    - Shape, column names, types
    - Numeric stats (mean, min, max, std)
    - Categorical value counts (top 5)
    - Missing values
    - Correlation highlights
    """
    lines = []

    # ── Basic info ────────────────────────────────────────────────────────────
    lines.append(f"DATASET OVERVIEW:")
    lines.append(f"- Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]}")
    lines.append(f"- Column names: {', '.join(df.columns.tolist())}")
    lines.append("")

    # ── Numeric summary ───────────────────────────────────────────────────────
    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty:
        lines.append("NUMERIC COLUMN STATISTICS:")
        for col in numeric_df.columns[:10]:   # Cap at 10 cols
            s = numeric_df[col]
            lines.append(
                f"- {col}: mean={s.mean():.2f}, "
                f"min={s.min():.2f}, max={s.max():.2f}, "
                f"std={s.std():.2f}, "
                f"missing={s.isnull().sum()}"
            )
        lines.append("")

    # ── Categorical summary ───────────────────────────────────────────────────
    cat_df = df.select_dtypes(include=["object", "category"])
    if not cat_df.empty:
        lines.append("CATEGORICAL COLUMN SUMMARY:")
        for col in cat_df.columns[:8]:   # Cap at 8 cols
            top = df[col].value_counts().head(5)
            top_str = ", ".join([f"{k}({v})" for k, v in top.items()])
            lines.append(f"- {col}: {df[col].nunique()} unique | Top: {top_str}")
        lines.append("")

    # ── Correlation highlights ────────────────────────────────────────────────
    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr().abs()
        # Get upper triangle pairs
        upper = corr.where(
            np.triu(np.ones(corr.shape), k=1).astype(bool)
        )
        # Top 5 strongest correlations
        top_corr = (
            upper.stack()
            .reset_index()
            .rename(columns={"level_0": "col_a", "level_1": "col_b", 0: "corr"})
            .sort_values("corr", ascending=False)
            .head(5)
        )
        if not top_corr.empty:
            lines.append("TOP CORRELATIONS:")
            for _, row in top_corr.iterrows():
                lines.append(
                    f"- {row['col_a']} ↔ {row['col_b']}: "
                    f"{row['corr']:.2f}"
                )
            lines.append("")

    # ── Missing values ────────────────────────────────────────────────────────
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        lines.append("MISSING VALUES:")
        for col, count in missing.items():
            pct = (count / len(df) * 100)
            lines.append(f"- {col}: {count} missing ({pct:.1f}%)")
        lines.append("")

    return "\n".join(lines)


def build_prompt(data_summary, user_requirement, output_sections):
    """
    Build the complete prompt to send to Claude.

    Args:
        data_summary      : text summary from build_data_summary()
        user_requirement  : custom text entered by user (can be empty)
        output_sections   : list of sections to include in report

    Returns:
        system_prompt, user_prompt (both strings)
    """
    system_prompt = """You are an expert Data Analyst and Business Intelligence consultant 
with 15+ years of experience. Your job is to analyze datasets and generate 
professional, actionable business reports.

Your reports must:
- Be written in clear, professional English
- Cite specific numbers and percentages from the data
- Give concrete, actionable business recommendations
- Use structured formatting with clear section headers
- Be insightful, not just descriptive
- Highlight risks and opportunities
- Be suitable for a C-level executive audience

Always base your analysis strictly on the data provided. 
Never fabricate numbers not present in the data summary."""

    req_section = ""
    if user_requirement.strip():
        req_section = f"""
USER REQUIREMENT:
The user specifically wants: "{user_requirement}"
Make sure this requirement is addressed prominently in the report.
"""

    sections_str = "\n".join([f"- {s}" for s in output_sections])

    user_prompt = f"""
Please analyze the following dataset and generate a comprehensive business report.

{req_section}

{data_summary}

Generate a professional report with EXACTLY these sections:
{sections_str}

Format each section with:
## [Section Name]
[Content with specific data references]

Be specific, data-driven, and actionable. 
Reference actual numbers from the dataset statistics above.
"""

    return system_prompt, user_prompt


def generate_report(
    df,
    api_key,
    user_requirement="",
    report_type="full"
):
    """
    Main report generation function.
    Calls Claude API and returns the report text.

    Args:
        df               : cleaned DataFrame
        api_key          : Anthropic API key
        user_requirement : custom user prompt (optional)
        report_type      : 'full', 'summary', or 'recommendations'

    Returns:
        report_text (str) or raises Exception
    """

    # ── Build data context ────────────────────────────────────────────────────
    data_summary = build_data_summary(df)

    # ── Select report sections based on type ──────────────────────────────────
    if report_type == "summary":
        sections = [
            "Executive Summary (3-4 sentences)",
            "Key Metrics Overview",
            "Top 3 Insights"
        ]
    elif report_type == "recommendations":
        sections = [
            "Executive Summary",
            "Critical Findings",
            "Business Recommendations (minimum 5 actionable items)",
            "Risk Factors",
            "Next Steps"
        ]
    else:  # full report
        sections = [
            "Executive Summary",
            "Dataset Overview & Data Quality",
            "Key Performance Metrics",
            "Trend Analysis & Patterns",
            "Correlation Insights",
            "Business Insights (minimum 5 specific insights)",
            "Risks & Opportunities",
            "Actionable Recommendations (minimum 5)",
            "Conclusion"
        ]

    # ── Build prompts ─────────────────────────────────────────────────────────
    system_prompt, user_prompt = build_prompt(
        data_summary, user_requirement, sections
    )

    # ── Call Claude API ───────────────────────────────────────────────────────
    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2500,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_prompt}
        ]
    )

    return message.content[0].text


def format_report_as_text(report_text, df_name="Dataset", user_name="Jeevan Bhatkar"):
    """
    Wrap the AI report with a professional header/footer
    for the downloadable text file version.
    """
    from datetime import datetime
    now = datetime.now().strftime("%d %B %Y, %I:%M %p")

    header = f"""
╔══════════════════════════════════════════════════════════════════╗
║              AI DATA ANALYST — BUSINESS REPORT                  ║
║                  Created by {user_name:<36}║
╚══════════════════════════════════════════════════════════════════╝

Dataset : {df_name}
Generated: {now}
Powered by: Claude AI (Anthropic) + Python + Pandas
{'─' * 66}

"""
    footer = f"""

{'─' * 66}
Report generated by AI Data Analyst
Built by {user_name} | Powered by Claude API
{'═' * 66}
"""
    return header + report_text + footer 