"""
Twitter Sentiment (Group 9)
Course: ID-2221, Fall 2025

Dashboard: Twitter sentiment vs. age & time.
Run `python visualization.py` to launch the dashboard on http://localhost:8050.
"""

from __future__ import annotations

import os
from functools import lru_cache
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pymongo
from pymongo.errors import PyMongoError
from dash import Dash, Input, Output, dcc, html

DEFAULT_URI = (
    "mongodb+srv://saad:mongoPass@dataprojectid2221.2uplah7.mongodb.net/"
    "?retryWrites=true&w=majority&appName=DataProjectID2221"
)
MONGO_URI = os.getenv("MONGO_URI", DEFAULT_URI)
DB_NAME = os.getenv("MONGO_DB", "Our_Database")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION", "processed_sentiment_data")


@lru_cache(maxsize=1)
def get_collection() -> pymongo.collection.Collection:
    client = pymongo.MongoClient(MONGO_URI)
    return client[DB_NAME][COLLECTION_NAME]


def fetch_dataframe() -> pd.DataFrame:
    projection = {
        "_id": 0,
        "textID": 1,
        "sentiment": 1,                
        "Time of Tweet": 1,
        "Age of User": 1,
        "processed_timestamp": 1,
        "compound": 1,                
        "p_positive": 1,              
        "p_negative": 1,               
    }
    try:
        records = list(get_collection().find({}, projection))
    except PyMongoError as exc:
        print(f"[visualization] MongoDB query failed: {exc}")
        return pd.DataFrame(columns=projection.keys())

    if not records:
        return pd.DataFrame(columns=projection.keys())

    df = pd.DataFrame(records)
    df = df.dropna(subset=["sentiment", "Time of Tweet", "Age of User"])
    df["sentiment"] = df["sentiment"].astype(str).str.lower()
    df["time_period"] = df["Time of Tweet"].astype(str).str.strip().str.lower()

    age_mapping = {
        "0-20": "0-30",
        "21-30": "0-30",
        "31-45": "31-60",
        "46-60": "31-60",
        "60-70": "61-100",
        "70-100": "61-100",
    }
    df["age_group"] = df["Age of User"].map(age_mapping).fillna(df["Age of User"])

    base_order = [
        "late night", "early morning", "morning",
        "noon", "afternoon", "evening", "night",
    ]
    dynamic_values = [val for val in df["time_period"].unique() if val not in base_order]
    ordering = base_order + sorted(dynamic_values)
    df["time_period"] = df["time_period"].astype(
        pd.CategoricalDtype(categories=ordering, ordered=True)
    )

    if "processed_timestamp" in df.columns:
        df["processed_timestamp"] = pd.to_datetime(df["processed_timestamp"], errors="coerce")

    def _compute_intensity(row):
        comp = row.get("compound", None)
        if pd.notnull(comp):
            try:
                val = float(comp)
                return max(-1.0, min(1.0, val))
            except Exception:
                pass

        ppos, pneg = row.get("p_positive", None), row.get("p_negative", None)
        if pd.notnull(ppos) and pd.notnull(pneg):
            try:
                val = float(ppos) - float(pneg)
                return max(-1.0, min(1.0, val))
            except Exception:
                pass

        s = (row.get("sentiment") or "").lower()
        return {"positive": 1.0, "neutral": 0.0, "negative": -1.0}.get(s, 0.0)

    df["sentiment_intensity"] = df.apply(_compute_intensity, axis=1)

    return df

def overall_sentiment(df: pd.DataFrame) -> go.Figure:
    counts = df["sentiment"].value_counts().rename_axis("sentiment").reset_index(name="count")
    if counts.empty:
        return empty_figure("No sentiment data found")

    fig = px.pie(
        counts,
        names="sentiment",
        values="count",
        color="sentiment",
        color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"},
        hole=0.45,
    )
    fig.update_traces(textinfo="label+percent", pull=[0.05, 0.05, 0.05])
    fig.update_layout(
        title="Overall Sentiment Mix",
        margin=dict(t=70, l=30, r=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=-0.05, xanchor="center", x=0.5),
    )
    return fig


def sentiment_by_time(df: pd.DataFrame) -> go.Figure:
    grouped = (
        df.groupby(["time_period", "sentiment"])
          .size()
          .reset_index(name="count")
          .dropna(subset=["time_period"])
    )
    if grouped.empty:
        return empty_figure("No time-of-day sentiment data")

    if isinstance(grouped["time_period"].dtype, pd.CategoricalDtype):
        ordering = [c for c in grouped["time_period"].cat.categories if c in grouped["time_period"].unique()]
    else:
        ordering = sorted(grouped["time_period"].unique())
    fig = px.bar(
        grouped,
        x="time_period",
        y="count",
        color="sentiment",
        barmode="group",
        color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"},
        category_orders={"time_period": ordering},
    )
    fig.update_layout(
        title="Sentiment by Time of Day",
        xaxis_title="Time of Tweet",
        yaxis_title="Tweet Count",
        legend_title="",
        margin=dict(t=70, l=50, r=30, b=50),
    )
    return fig


def sentiment_by_age(df: pd.DataFrame) -> go.Figure:
    grouped = df.groupby(["age_group", "sentiment"]).size().reset_index(name="count")
    if grouped.empty:
        return empty_figure("No age sentiment data")

    totals = grouped.groupby("age_group")["count"].transform("sum")
    grouped["share"] = grouped["count"] / totals

    fig = px.bar(
        grouped,
        x="age_group",
        y="share",
        color="sentiment",
        text=grouped["share"].apply(lambda x: f"{x:.0%}"),
        color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"},
        category_orders={"age_group": sorted(grouped["age_group"].unique())},
    )
    fig.update_traces(textposition="inside")
    fig.update_layout(
        title="Sentiment Share by Age Group",
        xaxis_title="Age Group",
        yaxis_title="Share of Sentiment",
        yaxis_tickformat="%",
        legend_title="",
        margin=dict(t=70, l=60, r=30, b=50),
    )
    return fig


def age_time_intensity_heatmap(df: pd.DataFrame) -> go.Figure:
    use_df = df.dropna(subset=["time_period", "age_group"])
    if use_df.empty:
        return empty_figure("No age/time sentiment data")

    agg = (
        use_df.groupby(["age_group", "time_period"])
              .agg(mean_intensity=("sentiment_intensity", "mean"),
                   n=("sentiment_intensity", "size"))
              .reset_index()
    )

    fig = px.scatter(
        agg,
        x="time_period",
        y="age_group",
        size="n",
        color="mean_intensity",
        color_continuous_scale="RdYlGn",
        size_max=50,
        title="Age × Time — Sentiment Intensity Bubble Map",
        labels={"n": "Samples", "mean_intensity": "Avg Intensity"},
    )

    fig.update_layout(
        template="plotly_white",
        height=420,
        coloraxis_colorbar=dict(len=0.75, thickness=15),
        margin=dict(t=80, l=70, r=40, b=60),
    )
    return fig

def sentiment_trend_by_age(df: pd.DataFrame) -> go.Figure:
    """
    折线图：展示不同年龄组在各时间段的平均情感强度变化趋势
    """
    use_df = df.dropna(subset=["time_period", "age_group"])
    if use_df.empty:
        return empty_figure("No age/time sentiment trend data")

    agg = (
        use_df.groupby(["time_period", "age_group"])
              .agg(mean_intensity=("sentiment_intensity", "mean"))
              .reset_index()
    )

    # 时间排序（与 dashboard 一致）
    if isinstance(use_df["time_period"].dtype, pd.CategoricalDtype):
        ordering = [c for c in use_df["time_period"].cat.categories if c in agg["time_period"].unique()]
    else:
        ordering = sorted(agg["time_period"].unique())

    fig = px.line(
        agg,
        x="time_period",
        y="mean_intensity",
        color="age_group",
        markers=True,
        category_orders={"time_period": ordering},
        color_discrete_sequence=px.colors.qualitative.Set2,
        title="Sentiment Intensity over Time by Age Group"
    )

    fig.update_traces(line=dict(width=3), marker=dict(size=8))
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Time of Tweet",
        yaxis_title="Average Sentiment Intensity",
        yaxis=dict(range=[-1, 1], zeroline=True, zerolinecolor="gray"),
        legend_title="Age Group",
        margin=dict(t=80, l=70, r=40, b=60),
        height=420,
    )
    return fig


def throughput_figure(df: pd.DataFrame, window_hours: int = 24) -> go.Figure:
    if "processed_timestamp" not in df.columns:
        return empty_figure("No ingestion timestamps")

    time_df = df.dropna(subset=["processed_timestamp"]).copy()
    if time_df.empty:
        return empty_figure("No ingestion timestamps")

    time_df["processed_timestamp"] = pd.to_datetime(
        time_df["processed_timestamp"], errors="coerce"
    ).dt.tz_localize(None)

    # 以 5 分钟为粒度
    time_df["minute"] = time_df["processed_timestamp"].dt.floor("5min")

    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(hours=window_hours - 1)

    grouped = (
        time_df.groupby(["minute", "sentiment"])
               .size()
               .reset_index(name="count")
    )
    if grouped.empty:
        return empty_figure("No throughput data")

    full_index = pd.date_range(max(grouped["minute"].min(), start),
                               grouped["minute"].max(), freq="5min")
    wide = (
        grouped.pivot(index="minute", columns="sentiment", values="count")
               .reindex(full_index)
               .fillna(0)
    )
    long = (
        wide.reset_index()
            .melt(id_vars="index", var_name="sentiment", value_name="count")
            .rename(columns={"index": "hour"})
    )

    fig = px.line(
        long,
        x="hour",
        y="count",
        color="sentiment",
        markers=True,
        color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"},
    )
    fig.update_layout(
        title="Pipeline Throughput (Every 5 Minutes)",
        xaxis_title="Processing Timestamp",
        yaxis_title="Tweets",
        margin=dict(t=70, l=60, r=30, b=50),
        yaxis=dict(rangemode="tozero"),
    )
    return fig

def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=message,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        annotations=[
            dict(
                text=message,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16, color="#888"),
            )
        ],
    )
    return fig

app = Dash(__name__)
app.title = "Twitter Sentiment Intelligence"

app.layout = html.Div(
    [
        dcc.Store(id="sentiment-store"),
        dcc.Interval(id="refresh", interval=5 * 60 * 1000, n_intervals=0),
        html.Div(
            [
                html.H1("Twitter Sentiment Intelligence"),
                html.P(
                    "Multi-view dashboard tracking sentiment fluctuations by time of day and age group.",
                    style={"color": "#555", "maxWidth": "760px"},
                ),
            ],
            style={"padding": "22px 30px"},
        ),
        html.Div(
            [
                html.Div(dcc.Graph(id="overall"), className="panel"),
                html.Div(dcc.Graph(id="age"), className="panel"),
            ],
            className="grid grid-2",
        ),
        html.Div(
            [html.Div(dcc.Graph(id="time"), className="panel")],
            className="grid grid-1",
        ),
        html.Div(
            [
                html.Div(dcc.Graph(id="bubble"), className="panel"),
                html.Div(dcc.Graph(id="trend"), className="panel"),
            ],
            className="grid grid-2",
        ),
        html.Div(
            [
                html.Div(dcc.Graph(id="throughput"), className="panel"),
            ],
            className="grid grid-2",
        ),


    ],
    style={"fontFamily": "Helvetica, Arial, sans-serif", "backgroundColor": "#f6f8fb"},
)

@app.callback(Output("sentiment-store", "data"), Input("refresh", "n_intervals"))
def refresh_data(_):
    df = fetch_dataframe()
    if "processed_timestamp" in df.columns:
        df["processed_timestamp"] = (
            pd.to_datetime(df["processed_timestamp"], errors="coerce")
              .dt.tz_localize(None)
              .dt.strftime("%Y-%m-%d %H:%M:%S")
        )
    df = df.where(pd.notnull(df), None)
    return df.to_dict("records")


@app.callback(
    Output("overall", "figure"),
    Output("age", "figure"),
    Output("time", "figure"),
    # Output("age-time", "figure"),
    Output("bubble", "figure"),      # 新增 bubble 图
    Output("trend", "figure"),       # 新增 trend 折线图
    Output("throughput", "figure"),
    Input("sentiment-store", "data"),
)
def update_figures(data):
    if not data:
        empty = empty_figure("No sentiment records retrieved")
        return empty, empty, empty, empty, empty, empty, empty

    df = pd.DataFrame(data)
    return (
        overall_sentiment(df),
        sentiment_by_age(df),
        sentiment_by_time(df),
        age_time_intensity_heatmap(df),
        sentiment_trend_by_age(df),
        throughput_figure(df),
    )

app.clientside_callback(
    """
    function(_) {
        if (document.getElementById('dash-custom-css')) {
            return null;
        }
        const style = document.createElement('style');
        style.id = 'dash-custom-css';
        style.innerHTML = `
            .grid {
                display: grid;
                gap: 20px;
                padding: 0 30px 30px;
                box-sizing: border-box;
            }
            .grid-1 { grid-template-columns: 1fr; }
            .grid-2 { grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); }
            .panel {
                background: #ffffff;
                border-radius: 18px;
                box-shadow: 0 12px 22px rgba(51, 83, 145, 0.12);
                padding: 18px;
            }
        `;
        document.head.appendChild(style);
        return null;
    }
    """,
    Output("overall", "id"),
    Input("refresh", "n_intervals"),
)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.getenv("PORT", 8050)))
