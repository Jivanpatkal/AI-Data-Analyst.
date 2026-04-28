import anthropic
import pandas as pd
import numpy as np
from modules.ai_report import build_data_summary

# ══════════════════════════════════════════════════════════════════════════════
# CHATBOT MODULE
# Conversational AI that answers questions about the uploaded dataset
# Maintains chat history for multi-turn conversations
# ══════════════════════════════════════════════════════════════════════════════


def build_system_prompt(df):
    """
    Build a rich system prompt giving Claude full context
    about the current dataset. Called once per session.

    Args:
        df : cleaned DataFrame

    Returns:
        system_prompt (str)
    """
    data_summary = build_data_summary(df)

    # Build sample rows as readable text
    sample = df.head(5).to_string(index=False)

    system_prompt = f"""You are an expert Data Analyst AI assistant embedded in a 
data analysis application. You have been given access to a dataset that the user 
has uploaded and cleaned.

Your role is to:
1. Answer questions about the dataset clearly and accurately
2. Perform calculations and analysis based on the data statistics provided
3. Give business insights and recommendations when asked
4. Explain patterns, trends, and anomalies
5. Suggest next steps or additional analyses
6. Help the user understand their data better

IMPORTANT RULES:
- Only reference numbers and facts that appear in the data summary below
- Never fabricate data points not present in the summary
- If you cannot answer from the data provided, say so clearly
- Keep answers concise but complete (3-6 sentences for simple questions)
- Use bullet points for lists and comparisons
- Always be helpful, professional, and encouraging

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATASET CONTEXT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{data_summary}

SAMPLE DATA (first 5 rows):
{sample}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You are ready to answer questions about this specific dataset.
Start every first response by briefly confirming what dataset you're analyzing."""

    return system_prompt


def get_suggested_questions(df):
    """
    Auto-generate smart suggested questions based on
    the columns and data types in the dataset.

    Returns a list of up to 6 question strings.
    """
    questions = []
    cols      = df.columns.tolist()
    num_cols  = df.select_dtypes(include="number").columns.tolist()
    cat_cols  = df.select_dtypes(include=["object","category"]).columns.tolist()
    dt_cols   = df.select_dtypes(include="datetime").columns.tolist()

    # Always include
    questions.append("What are the top 3 insights from this dataset?")
    questions.append("Summarize this dataset in 5 bullet points.")

    # Numeric-based questions
    if num_cols:
        col = num_cols[0]
        questions.append(f"What is the average and range of {col}?")
        if len(num_cols) >= 2:
            questions.append(
                f"Is there a relationship between {num_cols[0]} and {num_cols[1]}?"
            )

    # Categorical-based questions
    if cat_cols:
        col = cat_cols[0]
        questions.append(f"Which {col} appears most frequently?")

    # Datetime-based questions
    if dt_cols:
        questions.append("What trends do you see over time in this data?")

    # Correlation
    if len(num_cols) >= 2:
        questions.append("Which columns are most strongly correlated?")

    # Business
    questions.append("What business recommendations do you have based on this data?")
    questions.append("Are there any anomalies or outliers I should be aware of?")
    questions.append("What additional data would improve this analysis?")

    return questions[:6]   # Return max 6


def chat_with_data(messages, api_key, system_prompt):
    """
    Send conversation history to Claude and get a response.

    Args:
        messages      : list of {"role": "user"/"assistant", "content": str}
        api_key       : Anthropic API key
        system_prompt : system context built from dataset

    Returns:
        response_text (str)
    """
    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=system_prompt,
        messages=messages
    )

    return response.content[0].text


def format_chat_history_as_text(chat_history, dataset_name="Dataset"):
    """
    Export the full chat history as a readable text file.

    Args:
        chat_history : list of {"role", "content"} dicts
        dataset_name : name of the dataset analyzed

    Returns:
        formatted string
    """
    from datetime import datetime
    now = datetime.now().strftime("%d %B %Y, %I:%M %p")

    lines = [
        "╔══════════════════════════════════════════════════════════╗",
        "║         ASK YOUR DATA — CHAT HISTORY EXPORT             ║",
        "║              AI Data Analyst by Jeevan Bhatkar          ║",
        "╚══════════════════════════════════════════════════════════╝",
        f"\nDataset  : {dataset_name}",
        f"Exported : {now}",
        "─" * 60,
        ""
    ]

    for i, msg in enumerate(chat_history):
        role  = "🧑 You" if msg["role"] == "user" else "🤖 AI Analyst"
        lines.append(f"{role}:")
        lines.append(msg["content"])
        lines.append("")
        if i < len(chat_history) - 1:
            lines.append("─" * 40)
            lines.append("")

    lines.append("─" * 60)
    lines.append("Powered by Claude AI | Built by Jeevan Bhatkar")

    return "\n".join(lines)