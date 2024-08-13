import pandas as pd
import numpy as np
import json
from pandas import json_normalize
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import loguru as logger

train_path = "data/train_v2.csv"
test_path = "data/test_v2.csv"
nrows = 200000


def load_df(csv_path, nrows=None):
    """
    This function is used to load large csv files with json columns into pandas dataframes
    """
    JSON_COLUMNS = [
        # "customDimensions", #the column has single quotes and hence not picked up by JSON Loader
        "device",
        "geoNetwork",
        # "hits", #the column has single quotes and hence not picked up by JSON Loader
        "totals",
        "trafficSource",
    ]
    df = pd.read_csv(
        csv_path,
        converters={column: json.loads for column in JSON_COLUMNS},
        dtype={"fullVisitorId": "str"},
        nrows=nrows,
    )

    for column in JSON_COLUMNS:
        columns_as_df = json_normalize(df[column])
        columns_as_df.columns = [
            f"{column}_{subcolumn}" for subcolumn in columns_as_df.columns
        ]
        df = df.drop(column, axis=1).merge(
            columns_as_df, right_index=True, left_index=True
        )

    print(f"Loaded {csv_path}, shape: {df.shape}")
    return df


def read_data(path, nrows):
    """
    In this section of code, we are doing the following steps as part of preprocessing.
    1. filling the boolean features missing value with 0
    2. from POSIX timestamp to datetime
    3. from integer date to datetime date
    """
    df = load_df(csv_path=path, nrows=nrows)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["visitStartTime"] = df["visitStartTime"].apply(datetime.fromtimestamp)

    df["totals_newVisits"] = df["totals_newVisits"].fillna(0)
    df["totals_bounces"] = df["totals_bounces"].fillna(0)
    df["trafficSource_isTrueDirect"] = df["trafficSource_isTrueDirect"].fillna(0)
    df["device_isMobile"] = df["device_isMobile"].map({True: 1, False: 0})

    # defining the target variable using log1p transformation
    df["target"] = np.log1p(df["totals_transactionRevenue"].fillna(0).astype("float"))

    logger.debug("Reading data from {path} completed, data shape = {df.shape}")
    return df


def plotly_bargraphs(df, col_name, color):
    trace = go.Figure(
        go.Bar(x=df[col_name], y=df.index, marker=dict(color=color), orientation="h")
    )
    return trace
