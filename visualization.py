"""
Twitter Sentiment (Group 9)
Course: ID-2221, Fall 2025

Dashboard: Twitter sentiment vs. age & time.
Run `python visualization.py` to launch the dashboard on http://localhost:8050.
"""

from __future__ import annotations

import os
from functools import lru_cache

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pymongo
from pymongo.errors import PyMongoError
from dash import Dash, Input, Output, dcc, html
from datetime import datetime, timedelta

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
    df["sentiment"] = df["sentiment"].str.lower()
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
        "late night",
        "early morning",
        "morning",
        "noon",
        "afternoon",
        "evening",
        "night",
    ]
    dynamic_values = [val for val in df["time_period"].unique() if val not in base_order]
    ordering = base_order + sorted(dynamic_values)
    df["time_period"] = df["time_period"].astype(
        pd.CategoricalDtype(categories=ordering, ordered=True)
    )

    if "processed_timestamp" in df.columns:
        df["processed_timestamp"] = pd.to_datetime(df["processed_timestamp"], errors="coerce")

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
        color_discrete_map={
            "positive": "#2ecc71",
            "negative": "#e74c3c",
            "neutral": "#95a5a6",
        },
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
        df.groupby(["time_period", "sentiment"]).size()
          .reset_index(name="count")
          .dropna(subset=["time_period"])
    )
    if grouped.empty:
        return empty_figure("No time-of-day sentiment data")

    if isinstance(grouped["time_period"].dtype, pd.CategoricalDtype):
        ordering = [c for c in grouped["time_period"].cat.categories
                    if c in grouped["time_period"].unique()]
    else:
        ordering = sorted(grouped["time_period"].unique())
    fig = px.bar(
        grouped,
        x="time_period",
        y="count",
        color="sentiment",
        barmode="group",
        color_discrete_map={
            "positive": "#2ecc71",
            "negative": "#e74c3c",
            "neutral": "#95a5a6",
        },
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
        color_discrete_map={
            "positive": "#2ecc71",
            "negative": "#e74c3c",
            "neutral": "#95a5a6",
        },
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


def age_time_heatmap(df: pd.DataFrame) -> go.Figure:
    pivot = df.groupby(["age_group", "time_period", "sentiment"]).size().reset_index(name="count")
    if pivot.empty:
        return empty_figure("No age/time sentiment data")

    pivot = pivot.pivot_table(
        index="age_group",
        columns=["time_period", "sentiment"],
        values="count",
        fill_value=0,
    )
    pivot = pivot.reindex(sorted(pivot.index), axis=0)
    pivot = pivot.sort_index(axis=1, level=[0, 1])

    labels = [f"{str(time)}\n{sent}" for time, sent in pivot.columns]

    heatmap = go.Heatmap(
        z=pivot.values,
        x=labels,
        y=pivot.index,
        colorscale="RdYlGn",
        colorbar=dict(title="Tweet Count"),
    )
    fig = go.Figure(heatmap)
    fig.update_layout(
        title="Age vs Time Sentiment Intensity",
        xaxis_title="Time / Sentiment",
        yaxis_title="Age Group",
        margin=dict(t=90, l=70, r=40, b=90),
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

    # time_df["hour"] = time_df["processed_timestamp"].dt.floor("h")
    time_df["minute"] = time_df["processed_timestamp"].dt.floor("5min") 


    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(hours=window_hours-1)
    # full_index = pd.date_range(start, now, freq="h")
    
    # grouped = (
    #     time_df.groupby(["hour", "sentiment"])
    #            .size()
    #            .reset_index(name="count")
    # )
    # wide = (
    #     grouped.pivot(index="hour", columns="sentiment", values="count")
    #            .reindex(full_index)
    #            .fillna(0)
    # )
    grouped = (
    time_df.groupby(["minute", "sentiment"])
           .size()
           .reset_index(name="count")
    )
    full_index = pd.date_range(grouped["minute"].min(), grouped["minute"].max(), freq="5min")
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
        color_discrete_map={
            "positive": "#2ecc71",
            "negative": "#e74c3c",
            "neutral": "#95a5a6",
        },
    )
    fig.update_layout(
        # title=f"Pipeline Throughput (Hourly, last {window_hours}h)",
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
                html.Div(dcc.Graph(id="age-time"), className="panel"),
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
    Output("age-time", "figure"),
    Output("throughput", "figure"),
    Input("sentiment-store", "data"),
)
def update_figures(data):
    if not data:
        empty = empty_figure("No sentiment records retrieved")
        return empty, empty, empty, empty, empty

    df = pd.DataFrame(data)
    return (
        overall_sentiment(df),
        sentiment_by_age(df),
        sentiment_by_time(df),
        age_time_heatmap(df),
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
