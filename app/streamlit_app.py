from pathlib import Path
import json

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "best_airbnb_price_model.joblib"
METADATA_PATH = ROOT_DIR / "app" / "dashboard_metadata.json"
ROOM_TYPE_SUMMARY_PATH = ROOT_DIR / "reports" / "room_type_summary.csv"
NEIGHBORHOOD_SUMMARY_PATH = ROOT_DIR / "reports" / "neighborhood_summary.csv"
NEIGHBORHOOD_COORDINATES_PATH = ROOT_DIR / "reports" / "neighborhood_coordinates.csv"
MODEL_RESULTS_PATH = ROOT_DIR / "reports" / "model_results.csv"
FEATURE_IMPORTANCE_PATH = ROOT_DIR / "reports" / "feature_importance.csv"

PRIMARY = "#3B82F6"
TEAL = "#14B8A6"
FEATURE = "#8B5CF6"

st.set_page_config(
    page_title="Los Angeles Airbnb Market",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp {
            background: #F5F7FB;
            color: #172033;
        }

        [data-testid="stHeader"] {
            background: rgba(245, 247, 251, 0.92);
        }

        [data-testid="stMainBlockContainer"] {
            max-width: 1280px;
            padding-top: 4.5rem;
            padding-bottom: 4rem;
        }

        .block-container {
            padding-top: 4.5rem !important;
        }

        [data-testid="stSidebar"] {
            background: #FFFFFF;
            border-right: 1px solid #E5EAF1;
            color: #1E293B;
        }

        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.7rem;
        }

        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
            color: #334155 !important;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] summary p {
            color: #FFFFFF !important;
            font-weight: 700;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] summary svg {
            color: #FFFFFF !important;
            fill: #FFFFFF !important;
            stroke: #FFFFFF !important;
        }

        [data-testid="stSidebar"]
        [data-testid="stExpander"]
        details:not([open]) > summary {
            background: #0E1117 !important;
            border-radius: 11px !important;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] * {
            color: #FFFFFF !important;
        }

        [data-testid="stSidebar"] input {
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
            caret-color: #FFFFFF !important;
        }

        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
            color: #64748B !important;
        }

        [data-testid="stSidebarHeader"],
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="stExpandSidebarButton"] {
            opacity: 1 !important;
            visibility: visible !important;
            transition: none !important;
        }

        [data-testid="stSidebarCollapseButton"] button,
        [data-testid="stSidebarCollapsedControl"] button,
        button[data-testid="stExpandSidebarButton"],
        [data-testid="stExpandSidebarButton"] button {
            width: 2.25rem !important;
            min-width: 2.25rem !important;
            height: 2.25rem !important;
            background: #2563EB !important;
            border: 1px solid #1D4ED8 !important;
            border-radius: 10px !important;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.28) !important;
            color: #FFFFFF !important;
            opacity: 1 !important;
            visibility: visible !important;
            transition: none !important;
        }

        [data-testid="stSidebarCollapseButton"] button:hover,
        [data-testid="stSidebarCollapsedControl"] button:hover,
        button[data-testid="stExpandSidebarButton"]:hover,
        [data-testid="stExpandSidebarButton"] button:hover {
            background: #1D4ED8 !important;
            border-color: #1E40AF !important;
        }

        [data-testid="stSidebarCollapseButton"] button svg,
        [data-testid="stSidebarCollapsedControl"] button svg,
        button[data-testid="stExpandSidebarButton"] svg,
        [data-testid="stExpandSidebarButton"] button svg {
            color: #FFFFFF !important;
            fill: #FFFFFF !important;
            stroke: #FFFFFF !important;
        }

        [data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"],
        button[data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"] {
            color: #FFFFFF !important;
        }

        [data-testid="stSidebarCollapsedControl"],
        [data-testid="stExpandSidebarButton"] {
            opacity: 1 !important;
            visibility: visible !important;
            z-index: 1000 !important;
        }

        [data-testid="stMetric"] {
            min-height: 142px;
            padding: 1.15rem 1.2rem;
            background: #FFFFFF;
            border: 1px solid #E3E9F2;
            border-radius: 14px;
            box-shadow: 0 5px 18px rgba(23, 50, 77, 0.05);
        }

        [data-testid="stMetricLabel"] {
            color: #607086;
            font-size: 0.86rem;
            font-weight: 600;
        }

        [data-testid="stMetricValue"] {
            color: #17324D;
            font-weight: 750;
            line-height: 1.15;
            overflow-wrap: anywhere;
            white-space: normal;
        }

        [data-testid="stMetricValue"] > div {
            font-size: clamp(1.35rem, 2.1vw, 2rem);
            line-height: 1.15;
            overflow-wrap: anywhere;
            white-space: normal;
        }

        [data-testid="stMetricDelta"] {
            color: #334E70 !important;
            font-size: 0.9rem !important;
            font-weight: 750 !important;
            opacity: 1 !important;
        }

        [data-testid="stMetricDelta"] p,
        [data-testid="stMetricDelta"] svg {
            color: #334E70 !important;
            fill: #334E70 !important;
            opacity: 1 !important;
        }

        [data-testid="stExpander"] {
            background: #F8FAFD;
            border: 1px solid #E5EAF1;
            border-radius: 12px;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            background: #FFFFFF;
            border-color: #E3E9F2;
            border-radius: 16px;
            box-shadow: 0 5px 18px rgba(23, 50, 77, 0.04);
        }

        h1, h2, h3 {
            color: #17324D;
            letter-spacing: -0.025em;
        }

        h2 {
            margin-top: 0.2rem;
        }

        .eyebrow {
            margin-bottom: 0.55rem;
            color: #2563EB;
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .hero-title {
            margin: 0;
            color: #17324D;
            font-size: clamp(2rem, 4vw, 3.2rem);
            font-weight: 800;
            letter-spacing: -0.045em;
            line-height: 1.05;
        }

        .hero-copy {
            max-width: 760px;
            margin: 0.8rem 0 1.8rem;
            color: #607086;
            font-size: 1.05rem;
            line-height: 1.65;
        }

        .sidebar-brand {
            padding: 0.4rem 0 0.65rem;
        }

        .sidebar-brand-title {
            color: #17324D !important;
            font-size: 1.25rem;
            font-weight: 800;
            letter-spacing: -0.025em;
        }

        .sidebar-brand-copy {
            margin-top: 0.25rem;
            color: #64748B !important;
            font-size: 0.86rem;
            line-height: 1.45;
        }

        .section-kicker {
            margin: 0;
            color: #2563EB;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }

        .chart-title {
            margin: 0;
            color: #17324D;
            font-size: 1.06rem;
            font-weight: 750;
        }

        .chart-copy {
            margin: 0.25rem 0 0.8rem;
            color: #718096;
            font-size: 0.88rem;
        }

        #MainMenu, footer {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r") as file:
        return json.load(file)


@st.cache_data
def load_market_data():
    room_type_summary = pd.read_csv(ROOM_TYPE_SUMMARY_PATH, index_col=0)
    neighborhood_summary = pd.read_csv(NEIGHBORHOOD_SUMMARY_PATH, index_col=0)
    neighborhood_coordinates = pd.read_csv(
        NEIGHBORHOOD_COORDINATES_PATH,
        index_col=0,
    )
    return room_type_summary, neighborhood_summary, neighborhood_coordinates


@st.cache_data
def load_model_reports():
    model_results = pd.read_csv(MODEL_RESULTS_PATH)
    feature_importance = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    return model_results, feature_importance


def horizontal_bar_chart(
    data,
    category,
    value,
    color,
    value_title,
    axis_format,
    text_format,
):
    """Return a clean, full-width horizontal bar chart."""
    chart_data = data[[category, value]].copy().sort_values(value, ascending=False)
    chart_max = float(chart_data[value].max())

    shared_encoding = {
        "y": alt.Y(
            f"{category}:N",
            sort=alt.SortField(field=value, order="descending"),
            title=None,
            axis=alt.Axis(
                domain=False,
                labelColor="#385C8A",
                labelFontSize=12,
                labelLimit=290,
                labelPadding=10,
                ticks=False,
            ),
        ),
        "x": alt.X(
            f"{value}:Q",
            title=None,
            scale=alt.Scale(domain=[0, chart_max * 1.18]),
            axis=alt.Axis(
                domain=False,
                format=axis_format,
                grid=True,
                gridColor="#E2EAF5",
                labelColor="#5F79A5",
                labelFontSize=11,
                tickCount=5,
                ticks=False,
            ),
        ),
    }

    bars = (
        alt.Chart(chart_data)
        .mark_bar(color=color, cornerRadiusEnd=7, size=28)
        .encode(
            **shared_encoding,
            tooltip=[
                alt.Tooltip(f"{category}:N", title="Category"),
                alt.Tooltip(f"{value}:Q", title=value_title, format=text_format),
            ],
        )
    )

    labels = (
        alt.Chart(chart_data)
        .mark_text(
            align="left",
            baseline="middle",
            color="#2563EB",
            dx=8,
            fontSize=12,
            fontWeight=700,
        )
        .encode(
            **shared_encoding,
            text=alt.Text(f"{value}:Q", format=text_format),
        )
    )

    return (
        (bars + labels)
        .properties(
            height=max(190, len(chart_data) * 40),
            background="#FFFFFF",
        )
        .configure_view(strokeWidth=0)
    )


model = load_model()
metadata = load_metadata()
room_type_summary, neighborhood_summary, neighborhood_coordinates = load_market_data()
model_results, feature_importance = load_model_reports()
defaults = metadata["defaults"]
best_result = model_results.sort_values("log_rmse").iloc[0]


with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">Listing Details</div>
            <div class="sidebar-brand-copy">
                Enter the details of a stay to estimate its nightly price and
                compare it with the Los Angeles market.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Listing essentials", expanded=True, icon="🏠"):
        neighbourhood_cleansed = st.selectbox(
            "Neighborhood",
            metadata["neighborhood_options"],
            index=metadata["neighborhood_options"].index(
                defaults["neighbourhood_cleansed"]
            ),
        )
        latitude = float(
            neighborhood_coordinates.loc[neighbourhood_cleansed, "latitude"]
        )
        longitude = float(
            neighborhood_coordinates.loc[neighbourhood_cleansed, "longitude"]
        )
        property_type = st.selectbox(
            "Detailed property type",
            metadata["property_type_options"],
            index=metadata["property_type_options"].index(defaults["property_type"]),
        )
        room_type = st.selectbox(
            "Booking type",
            metadata["room_type_options"],
            index=metadata["room_type_options"].index(defaults["room_type"]),
        )
        host_is_superhost = st.checkbox(
            "Superhost",
            value=bool(defaults["host_is_superhost"]),
        )
        accommodates = st.number_input(
            "Guests",
            min_value=1,
            max_value=20,
            value=int(defaults["accommodates"]),
        )
        bedrooms = st.number_input(
            "Bedrooms",
            min_value=0.0,
            max_value=20.0,
            value=float(defaults["bedrooms"]),
            step=1.0,
        )
        beds = st.number_input(
            "Beds",
            min_value=0.0,
            max_value=30.0,
            value=float(defaults["beds"]),
            step=1.0,
        )
        bathrooms = st.number_input(
            "Bathrooms",
            min_value=0.0,
            max_value=20.0,
            value=float(defaults["bathrooms"]),
            step=0.5,
        )
        amenities_count = st.number_input(
            "Amenities",
            min_value=0,
            max_value=200,
            value=int(defaults["amenities_count"]),
        )

    with st.expander("Stay rules & availability", icon="📅"):
        minimum_nights = st.number_input(
            "Minimum nights",
            min_value=1,
            max_value=365,
            value=int(defaults["minimum_nights"]),
        )
        maximum_nights = st.number_input(
            "Maximum nights",
            min_value=1,
            max_value=1125,
            value=int(defaults["maximum_nights"]),
        )
        availability_365 = st.number_input(
            "Available days per year",
            min_value=0,
            max_value=365,
            value=int(defaults["availability_365"]),
        )
        number_of_reviews = st.number_input(
            "Number of reviews",
            min_value=0,
            value=int(defaults["number_of_reviews"]),
        )

    with st.expander("Location & guest ratings", icon="📍"):
        st.caption(
            "Location is set automatically from the selected neighborhood."
        )
        review_scores_rating = st.number_input(
            "Overall rating",
            min_value=0.0,
            max_value=5.0,
            value=float(defaults["review_scores_rating"]),
            step=0.1,
        )
        review_scores_cleanliness = st.number_input(
            "Cleanliness rating",
            min_value=0.0,
            max_value=5.0,
            value=float(defaults["review_scores_cleanliness"]),
            step=0.1,
        )
        review_scores_location = st.number_input(
            "Location rating",
            min_value=0.0,
            max_value=5.0,
            value=float(defaults["review_scores_location"]),
            step=0.1,
        )
        review_scores_value = st.number_input(
            "Value rating",
            min_value=0.0,
            max_value=5.0,
            value=float(defaults["review_scores_value"]),
            step=0.1,
        )

    st.caption("The results update automatically when you change a detail.")


input_data = pd.DataFrame(
    [
        {
            "host_is_superhost": int(host_is_superhost),
            "neighbourhood_cleansed": neighbourhood_cleansed,
            "latitude": latitude,
            "longitude": longitude,
            "property_type": property_type,
            "room_type": room_type,
            "accommodates": accommodates,
            "bathrooms": bathrooms,
            "bedrooms": bedrooms,
            "beds": beds,
            "minimum_nights": minimum_nights,
            "maximum_nights": maximum_nights,
            "availability_365": availability_365,
            "number_of_reviews": number_of_reviews,
            "review_scores_rating": review_scores_rating,
            "review_scores_cleanliness": review_scores_cleanliness,
            "review_scores_location": review_scores_location,
            "review_scores_value": review_scores_value,
            "amenities_count": amenities_count,
        }
    ]
)

log_prediction = model.predict(input_data)[0]
predicted_price = float(np.expm1(log_prediction))
neighborhood_average = float(
    neighborhood_summary.loc[neighbourhood_cleansed, "avg_price"]
)
room_type_average = float(room_type_summary.loc[room_type, "avg_price"])
neighborhood_delta = (predicted_price / neighborhood_average - 1) * 100


st.markdown('<div class="eyebrow">Los Angeles market</div>', unsafe_allow_html=True)
st.markdown(
    '<h1 class="hero-title">Explore stays. Understand prices.</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <p class="hero-copy">
        Compare estimated nightly prices across Los Angeles. Guests can get a
        better idea of what a stay may cost, while hosts can use the same market
        view to help set a competitive rate.
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="section-kicker">Listing estimate</p>', unsafe_allow_html=True)
st.subheader("Estimated nightly price")

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

metric_col1.metric(
    "Estimated price",
    f"${predicted_price:,.0f}",
    f"{neighborhood_delta:+.0f}% vs. neighborhood",
    delta_color="off",
    border=False,
)
metric_col2.metric(
    "Neighborhood average",
    f"${neighborhood_average:,.0f}",
    neighbourhood_cleansed,
    delta_color="off",
    border=False,
)
metric_col3.metric(
    "Room type average",
    f"${room_type_average:,.0f}",
    room_type,
    delta_color="off",
    border=False,
)
metric_col4.metric(
    "Typical model error",
    f"±${best_result['dollar_mae']:,.0f}",
    "Mean absolute error",
    delta_color="off",
    border=False,
)

st.caption(
    "Use this estimate as a guide. Actual prices can change based on travel dates, "
    "local events, and the demand of the listing."
)

st.divider()

st.markdown('<p class="section-kicker">Market insights</p>', unsafe_allow_html=True)
st.header("Compare prices across Los Angeles")
st.caption(
    f"These comparisons use {int(room_type_summary['num_listings'].sum()):,} listings. "
    "Neighborhoods need at least 30 listings to appear in the ranking."
)

room_type_chart_data = (
    room_type_summary.reset_index()
    .rename(columns={room_type_summary.index.name or "index": "room_type"})
)

with st.container(border=True):
    st.markdown(
        """
        <p class="chart-title">Average nightly price by room type</p>
        <p class="chart-copy">See the typical price for each type of stay.</p>
        """,
        unsafe_allow_html=True,
    )
    room_type_chart = horizontal_bar_chart(
        room_type_chart_data,
        category="room_type",
        value="avg_price",
        color=PRIMARY,
        value_title="Average price",
        axis_format="$,.0f",
        text_format="$,.0f",
    )
    st.altair_chart(room_type_chart, width="stretch", theme=None)

top_neighborhoods = (
    neighborhood_summary[neighborhood_summary["num_listings"] >= 30]
    .sort_values("avg_price", ascending=False)
    .head(10)
    .reset_index()
    .rename(
        columns={
            neighborhood_summary.index.name or "index": "neighborhood",
        }
    )
)

with st.container(border=True):
    st.markdown(
        """
        <p class="chart-title">Top neighborhoods by average nightly price</p>
        <p class="chart-copy">See which neighborhoods have the highest average prices.</p>
        """,
        unsafe_allow_html=True,
    )
    neighborhood_chart = horizontal_bar_chart(
        top_neighborhoods,
        category="neighborhood",
        value="avg_price",
        color=TEAL,
        value_title="Average price",
        axis_format="$,.0f",
        text_format="$,.0f",
    )
    st.altair_chart(neighborhood_chart, width="stretch", theme=None)

st.divider()

st.markdown('<p class="section-kicker">Model quality</p>', unsafe_allow_html=True)
st.header("How accurate is the estimate?")

performance_col1, performance_col2, performance_col3 = st.columns([1.4, 1, 1])
performance_col1.metric("Best model", str(best_result["model"]), border=False)
performance_col2.metric(
    "Mean absolute error",
    f"${best_result['dollar_mae']:,.2f}",
    border=False,
)
performance_col3.metric(
    "R² on log price",
    f"{best_result['r2_log_price']:.3f}",
    border=False,
)

st.caption(
    "The model's error shows the average difference between predicted and actual prices. "
    "The R² score shows how well the model explains price variation."

)

feature_name_map = {
    "minimum_nights": "Minimum nights",
    "accommodates": "Guest capacity",
    "longitude": "Longitude",
    "latitude": "Latitude",
    "room_type": "Room type",
    "bathrooms": "Bathrooms",
    "bedrooms": "Bedrooms",
    "property_type": "Property type",
    "availability_365": "Annual availability",
    "neighbourhood_cleansed": "Neighborhood",
}

top_features = (
    feature_importance.sort_values("importance_mean", ascending=False).head(10).copy()
)
top_features["feature"] = (
    top_features["feature"].map(feature_name_map).fillna(top_features["feature"])
)

with st.container(border=True):
    st.markdown(
        """
        <p class="chart-title">Top feature importances</p>
        <p class="chart-copy">
            These are the listing details the model pays the most attention to.
            A longer bar means the detail has more influence on the estimate.
        </p>
        """,
        unsafe_allow_html=True,
    )
    feature_chart = horizontal_bar_chart(
        top_features,
        category="feature",
        value="importance_mean",
        color=FEATURE,
        value_title="Mean importance",
        axis_format=".2f",
        text_format=".3f",
    )
    st.altair_chart(feature_chart, width="stretch", theme=None)
