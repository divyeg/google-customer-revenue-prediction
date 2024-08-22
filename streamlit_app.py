import pandas as pd
import numpy as np
import json
import os
from pandas import json_normalize
from datetime import datetime, timedelta
import random

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as py

import streamlit as st

import warnings

warnings.simplefilter("ignore")

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
numeric_columns = [
    "totals_visits",
    "totals_hits",
    "totals_pageviews",
    "totals_bounces",
    "totals_newVisits",
    "totals_sessionQualityDim",
    "totals_timeOnSite",
    "totals_transactions",
    "totals_transactionRevenue",
    "totals_totalTransactionRevenue",
]

aggregations = {
    "fullVisitorId": "count",
    "totals_totalTransactionRevenue": "count",
    "totals_transactionRevenue": "sum",
}
agg_keys = [i for i in aggregations.keys()]
col_names = ["Visitors", "VisitorsWithRevenue", "TotalRevenue"]


# Helper Functions
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
    df[numeric_columns] = df[numeric_columns].astype("float64")

    print(f"Reading data from {path} completed, data shape = {df.shape}")
    return df


def create_trace(df, col_name, category_name, color):
    if len(category_name) == 0:
        legend_val = False
    else:
        legend_val = True
    trace = go.Figure(
        go.Bar(
            x=df[col_name][::-1],
            y=df.index[::-1],
            marker=dict(color=color),
            orientation="h",
            name=category_name,
            showlegend=legend_val,
        )
    )
    return trace


def create_plotly_data(df, groupby_col):
    df[groupby_cols] = df[groupby_cols].astype("object")
    df[agg_keys] = df[agg_keys].astype("float64")
    temp_data = df.groupby(groupby_col).agg(aggregations)
    temp_data.index = np.where(
        temp_data.totals_transactionRevenue == 0, "(Others)", temp_data.index
    )
    temp_data = temp_data.groupby(temp_data.index).agg(
        {
            "fullVisitorId": "sum",
            "totals_totalTransactionRevenue": "sum",
            "totals_transactionRevenue": "sum",
        }
    )
    temp_data.columns = col_names
    temp_data["Visitors"] = np.where(
        temp_data["Visitors"] == 0, 1, temp_data["Visitors"]
    )
    temp_data["VisitorsWithRevenue"] = np.where(
        temp_data["VisitorsWithRevenue"] == 0, 1, temp_data["VisitorsWithRevenue"]
    )
    temp_data["revenuePerVisit"] = np.round(
        temp_data["TotalRevenue"] / temp_data["Visitors"], 2
    )
    temp_data["spendPerRevenueVisit"] = np.round(
        temp_data["TotalRevenue"] / temp_data["VisitorsWithRevenue"], 2
    )
    temp_data.sort_values(by="Visitors", ascending=False, inplace=True)
    return temp_data


def create_bargraphs(df, groupby_col):
    plot_data = create_plotly_data(df, groupby_col)
    red = random.randint(0, 250)
    blue = random.randint(0, 250)
    green = random.randint(0, 250)
    plot1 = create_trace(
        plot_data, "Visitors", groupby_col, color=f"rgba({red}, {blue}, {green}, 0.6)"
    )
    plot2 = create_trace(
        plot_data, "VisitorsWithRevenue", "", color=f"rgba({red}, {blue}, {green}, 0.6)"
    )
    plot3 = create_trace(
        plot_data, "revenuePerVisit", "", color=f"rgba({red}, {blue}, {green}, 0.6)"
    )
    plot4 = create_trace(
        plot_data,
        "spendPerRevenueVisit",
        "",
        color=f"rgba({red}, {blue}, {green}, 0.6)",
    )
    return plot1, plot2, plot3, plot4


# Streamlit Functions
def home():
    st.header("Welcome to strealit web app Home Page")
    st.write("Begin exploring the data using the menu on the left")


@st.cache_data()
def data_scanner(df):
    st.header("Header of Dataframe")
    st.write(df.head())


@st.cache_data()
def data_statistics(trigger, df, selected_options):
    st.header("Statistics of Dataframe")
    if trigger or st.session_state.disable_opt1:
        st.session_state.disable_opt1 = True
        st.write(df[selected_options].describe())


@st.cache_data()
def create_interactive_plots(trigger, df, groupby_cols):
    st.header("Categorical Bar Plots")
    if trigger or st.session_state.disable_opt2:
        st.session_state.disable_opt2 = True
        fig = go.Figure()

        fig = make_subplots(
            rows=len(groupby_cols),
            cols=4,
            shared_xaxes=True,
            shared_yaxes=False,
            vertical_spacing=0.25 / len(groupby_cols),
            horizontal_spacing=0.10,
            subplot_titles=[
                "Visits",
                "Visits With Revenue",
                "Revenue per Visit",
                "Spend per Revenue Visit",
            ],
        )

        for i, groupby_col in enumerate(groupby_cols):
            plot1, plot2, plot3, plot4 = create_bargraphs(df, groupby_col)
            fig.add_trace(plot1.data[0], row=i + 1, col=1)
            fig.add_trace(plot2.data[0], row=i + 1, col=2)
            fig.add_trace(plot3.data[0], row=i + 1, col=3)
            fig.add_trace(plot4.data[0], row=i + 1, col=4)
            # fig.update_layout(
            #     yaxis={
            #         "tickmode": "array",
            #         "tickvals": plot1["data"][0]["y"].tolist(),
            #         "ticktext": [
            #             txt[:15] + "<br>" + txt[15:] if len(txt) > 15 else txt
            #             for txt in plot1["data"][0]["y"].tolist()
            #         ],
            #     }
            # )
            fig.update_yaxes(showticklabels=False, row=i + 1, col=2)
            fig.update_yaxes(showticklabels=False, row=i + 1, col=3)
            fig.update_yaxes(showticklabels=False, row=i + 1, col=4)

        # Updating layout
        fig.update_layout(
            yaxis={"dtick": 1},
            margin={"t": 50, "b": 50},
            height=400 * len(groupby_cols),
            width=1500,
            paper_bgcolor="rgb(233,233,233)",
            title="Visits and Revenue Spend across categorical features sorted by descending order of Visits",
        )
        fig.update_annotations(font_size=12)
        st.plotly_chart(fig)


# Streamlit App Code begins
st.title("Google Customer Revenue Prediction Data")
st.text(
    "This is a streamlit web app to allow exploration of Google Customer Revenue Data sourced from Kaggle"
)


@st.cache_resource()
def read_df(path):
    df = read_data(path, 200000)  # this takes time to load and read the data
    return df


if "df" not in st.session_state:
    st.session_state["df"] = None

data_file_path = os.path.join("/Users/divye/Desktop/gcrp_data/data", "train_v2.csv")
st.session_state["df"] = read_df(data_file_path)

# Sidebar setup
st.sidebar.title("Sidebar")
# Sidebar navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio(
    "Select what you want to display:",
    ["Home", "Data Scanner", "Data Statistics", "Data Exploration"],
)
refresh = st.sidebar.button("Refresh", on_click=st.cache_data.clear())

# Refresh button is not required in form since we are catching all the results
# with st.sidebar.form(key="page_options"):
#     # Sidebar setup
#     st.sidebar.title("Sidebar")
#     # Sidebar navigation
#     st.sidebar.title("Navigation")
#     options = st.sidebar.radio(
#         "Select what you want to display:",
#         ["Home", "Data Scanner", "Data Statistics", "Data Exploration"],
#     )
#     refresh = st.form_submit_button("Refresh", on_click=st.cache_data.clear())

if refresh:
    st.session_state["df"] = read_df(data_file_path)

if st.session_state["df"] is not None:
    if options == "Home":
        home()
    elif options == "Data Scanner":
        data_scanner(st.session_state["df"])
    elif options == "Data Statistics":
        try:
            _ = st.session_state.disable_opt1
        except AttributeError:
            st.session_state.disable_opt1 = False

        with st.form(key="describe_selctions"):
            selected_options = st.multiselect(
                "Select one or more options (maximum 6 selections are allowed):",
                st.session_state["df"]
                .select_dtypes(include=["float64", "int64"])
                .columns.tolist(),
                default=["target"],
                max_selections=6,
                # disabled=st.session_state.disable_opt,
                key="describe_options",
            )
            apply = st.form_submit_button("Apply")
        data_statistics(
            trigger=apply, df=st.session_state["df"], selected_options=selected_options
        )
    elif options == "Data Exploration":
        try:
            _ = st.session_state.disable_opt2
        except AttributeError:
            st.session_state.disable_opt2 = False

        with st.form(key="graph_selctions"):
            groupby_cols = st.multiselect(
                "Select one or more options (maximum 6 selections are allowed):",
                st.session_state["df"]
                .select_dtypes(include=["object"])
                .columns.tolist(),
                default=["channelGrouping", "device_browser"],
                max_selections=7,
                key="graph_options",
            )
            apply = st.form_submit_button("Apply")
        create_interactive_plots(
            trigger=apply, df=st.session_state["df"], groupby_cols=groupby_cols
        )
