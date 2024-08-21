import pandas as pd
import numpy as np
import json
from pandas import json_normalize
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import loguru as logger
import os

train_path = "data/train_v2.csv"
test_path = "data/test_v2.csv"
nrows = 200000

st.set_page_config(layout="wide")

feature_list = [
    "fullVisitorId",
    "date",
    "visitNumber",  # exponential distribution
    "VisitsStartTime",
    "channelGrouping",
    "socialEngagementType",
    "device_browser",
    "device_operatingSystem",
    "device_isMobile",
    "device_deviceCategory",
    "geoNetwork_continent",
    "geoNetwork_subcontinent",
    "geoNetwork_country",
    "geoNetwork_region",
    "geoNetwork_city",
    "geoNetwork_metro",
    "geoNetwork_networkDomain",
    "geoNetwork_sessionQualityDim",
    "geoNetwork_timeOnSite",
    "geoNetwork_transactions",
    "trafficSource_campaign",
    "trafficSource_soure",
    "trafficSource_medium",
    "trafficSource_keyword",
    "trafficSource_referralPath",
    "trafficSource_isTrueDirect",
    "trafficSource_adwordsClickInfo.page",
    "trafficSource_adwordsClickInfo.slot",
    "trafficSource_adwordsClickInfo.adNetworkType",
    "trafficSource_isVideoAd",
]


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

    print(f"Reading data from {path} completed, data shape = {df.shape}")
    return df


def plotly_bargraphs(df, col_name, color):
    trace = go.Figure(
        go.Bar(x=df[col_name], y=df.index, marker=dict(color=color), orientation="h")
    )
    return trace


# upload_file = st.file_uploader("Upload a file containing Google Customer Revenue Data")
data_file_path = os.path.join("/Users/divye/Desktop/gcrp_data/data", "train_v2.csv")


@st.cache_data
def read_df(path):
    df = read_data(data_file_path, 200000)  # this takes time to load and read the data
    return df


def home():
    st.header("Welcome to strealit web app Home Page")
    st.write("Begin exploring the data using the menu on the left")


@st.cache_data
def data_scanner(df):
    st.header("Header of Dataframe")
    st.write(df.head())


@st.cache_data
def data_statistics(trigger, df, selected_options):
    st.header("Statistics of Dataframe")
    if trigger or st.session_state.disable_opt:
        st.write(df[selected_options].describe())
        st.session_state.disable_opt = True


st.title("Google Customer Revenue Prediction Data")
st.text(
    "This is a streamlit web app to allow exploration of Google Customer Revenue Data sourced from Kaggle"
)

# Sidebar setup
st.sidebar.title("Sidebar")
# Sidebar navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio(
    "Select what you want to display:",
    ["Home", "Data Scanner", "Data Statistics", "Data Exploration"],
)
# df = read_df(data_file_path)

refresh = st.sidebar.button("Refresh", on_click=st.legacy_caching.caching.clear_cache())
if refresh:
    df = read_df(data_file_path)

# Navigation options
if options == "Home":
    home()
elif options == "Data Scanner":
    data_scanner(df)
elif options == "Data Statistics":
    try:
        _ = st.session_state.disable_opt
    except AttributeError:
        st.session_state.disable_opt = False

    with st.form(key="describe_selctions"):
        selected_options = st.multiselect(
            "Select one or more options (maximum 6 selections are allowed):",
            df.select_dtypes(include=["float64", "int64"]).columns.tolist(),
            default=["target"],
            max_selections=6,
            # disabled=st.session_state.disable_opt,
            key="describe_options",
        )
        apply = st.form_submit_button("Apply")
    data_statistics(trigger=apply, df=df, selected_options=selected_options)
elif options == "Data Exploration":
    pass
