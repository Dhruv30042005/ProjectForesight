import streamlit as st
import pandas as pd
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FORESIGHT | Demand & Inventory Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "processed" / "weekly_demand.csv"
FORECAST_PATH = BASE_DIR / "data" / "processed" / "future_demand_forecast.csv"
RISK_PATH = BASE_DIR / "reports" / "risk_scoring.csv"

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)
forecast_df = pd.read_csv(FORECAST_PATH)
risk_df = pd.read_csv(RISK_PATH)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       MAIN APP
       ======================================================== */

    [data-testid="stAppViewContainer"] {
        background-color: #f5f7fb;
    }

    [data-testid="stMain"] {
        background-color: #f5f7fb;
    }

    /* ========================================================
       SIDEBAR
       ======================================================== */

    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
    }

    /* Sidebar headings and normal text */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: #ffffff !important;
    }

    /* Sidebar selectbox container */
    [data-testid="stSidebar"] [data-baseweb="select"] {
        width: 100%;
    }

    /* White selectbox */
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border: 1px solid #d1d5db !important;
        border-radius: 10px !important;
        min-height: 42px !important;
    }

    /* Selected value */
    [data-testid="stSidebar"] [data-baseweb="select"] > div > div {
        color: #111827 !important;
        background-color: #ffffff !important;
    }

    /* Selected text */
    [data-testid="stSidebar"] [data-baseweb="select"] span {
        color: #111827 !important;
    }

    /* Input text */
    [data-testid="stSidebar"] [data-baseweb="select"] input {
        color: #111827 !important;
        caret-color: #111827 !important;
    }

    /* Dropdown arrow */
    [data-testid="stSidebar"] [data-baseweb="select"] svg {
        fill: #111827 !important;
        color: #111827 !important;
    }

    /* Dropdown menu */
    [data-baseweb="menu"] {
        background-color: #ffffff !important;
        border-radius: 10px !important;
    }

    /* Dropdown options */
    [data-baseweb="menu"] *,
    [role="option"] {
        color: #111827 !important;
        background-color: #ffffff !important;
    }

    /* Hover option */
    [role="option"]:hover {
        background-color: #e5e7eb !important;
        color: #111827 !important;
    }

    /* ========================================================
       SIDEBAR BRAND
       ======================================================== */

    .sidebar-brand {
        padding: 10px 0 28px 0;
    }

    .sidebar-brand h1 {
        font-size: 30px;
        font-weight: 800;
        margin: 0;
        color: #ffffff !important;
    }

    .sidebar-brand p {
        margin-top: 5px;
        color: #94a3b8 !important;
        font-size: 14px;
    }

    .sidebar-description {
        margin-top: 35px;
        padding: 15px 0;
        border-top: 1px solid rgba(255,255,255,0.08);
    }

    .sidebar-description p {
        color: #64748b !important;
        font-size: 13px;
        line-height: 1.5;
    }

    /* ========================================================
       HERO
       ======================================================== */

    .hero {
        padding: 34px 38px;
        border-radius: 24px;
        background: linear-gradient(
            135deg,
            #0f172a 0%,
            #172554 45%,
            #1d4ed8 100%
        );
        color: white;
        margin-bottom: 26px;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.18);
    }

    .hero h1 {
        font-size: 48px;
        font-weight: 800;
        margin: 0 0 8px 0;
        color: white !important;
    }

    .hero h2 {
        font-size: 24px;
        font-weight: 700;
        margin: 0 0 12px 0;
        color: white !important;
    }

    .hero p {
        font-size: 17px;
        margin: 0;
        color: #dbeafe !important;
    }

    /* ========================================================
       KPI CARDS
       ======================================================== */

    [data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 20px 20px 16px 20px;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.07);
    }

    [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-size: 13px !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricValue"] {
        color: #111827 !important;
        font-weight: 800 !important;
    }

    /* ========================================================
       HEADINGS
       ======================================================== */

    h1, h2, h3 {
        color: #111827 !important;
    }

    /* ========================================================
       DATAFRAME
       ======================================================== */

    [data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid #e5e7eb;
    }

    /* ========================================================
       EXPANDER
       ======================================================== */

    [data-testid="stExpander"] {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
    }

    /* ========================================================
       DIVIDER
       ======================================================== */

    hr {
        border-color: #e5e7eb !important;
    }

    /* ========================================================
       CAPTION
       ======================================================== */

    [data-testid="stCaptionContainer"] {
        color: #64748b !important;
    }

    /* ========================================================
       GENERAL SPACING
       ======================================================== */

    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            <h1>📦 FORESIGHT</h1>
            <p>Demand & Inventory Intelligence</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("## Dashboard Filters")

    stores = ["All"] + sorted(
        df["Store ID"].astype(str).unique().tolist()
    )

    products = ["All"] + sorted(
        df["Product ID"].astype(str).unique().tolist()
    )

    categories = ["All"] + sorted(
        df["Category"].astype(str).unique().tolist()
    )

    selected_store = st.selectbox(
        "Store",
        stores,
        index=0
    )

    selected_product = st.selectbox(
        "Product",
        products,
        index=0
    )

    selected_category = st.selectbox(
        "Category",
        categories,
        index=0
    )

    st.markdown(
        """
        <div class="sidebar-description">
            <p>
                Use the filters to inspect demand, forecasts
                and inventory risks for a specific scope.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# FILTER DATA
# ============================================================

filtered = df.copy()
filtered_risk = risk_df.copy()
filtered_forecast = forecast_df.copy()

if selected_store != "All":

    filtered = filtered[
        filtered["Store ID"].astype(str) == selected_store
    ]

    filtered_risk = filtered_risk[
        filtered_risk["Store ID"].astype(str) == selected_store
    ]

    filtered_forecast = filtered_forecast[
        filtered_forecast["Store ID"].astype(str) == selected_store
    ]


if selected_product != "All":

    filtered = filtered[
        filtered["Product ID"].astype(str) == selected_product
    ]

    filtered_risk = filtered_risk[
        filtered_risk["Product ID"].astype(str) == selected_product
    ]

    filtered_forecast = filtered_forecast[
        filtered_forecast["Product ID"].astype(str) == selected_product
    ]


if selected_category != "All":

    filtered = filtered[
        filtered["Category"].astype(str) == selected_category
    ]

    filtered_risk = filtered_risk[
        filtered_risk["Category"].astype(str) == selected_category
    ]

    filtered_forecast = filtered_forecast[
        filtered_forecast["Category"].astype(str) == selected_category
    ]

# ============================================================
# KPI VALUES
# ============================================================

products_count = filtered["Product ID"].nunique()

stores_count = filtered["Store ID"].nunique()

units_sold = pd.to_numeric(
    filtered["Units_Sold"],
    errors="coerce"
).fillna(0).sum()

stockout_risks = (
    filtered_risk["Stockout_Risk_Level"]
    .astype(str)
    .str.lower()
    .eq("high")
    .sum()
)

estimated_impact = pd.to_numeric(
    filtered_risk["Estimated_Impact_INR"],
    errors="coerce"
).fillna(0).sum()

# ============================================================
# HERO
# ============================================================

st.markdown(
    '<div class="hero">'
    '<h1>📦 FORESIGHT</h1>'
    '<h2>Demand & Inventory Intelligence Platform</h2>'
    '<p>AI-powered demand forecasting, inventory risk detection '
    'and actionable planning recommendations</p>'
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# KPI CARDS
# ============================================================

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric(
        "PRODUCTS",
        f"{products_count:,}"
    )

with c2:
    st.metric(
        "STORES",
        f"{stores_count:,}"
    )

with c3:
    st.metric(
        "UNITS SOLD",
        f"{units_sold:,.0f}"
    )

with c4:
    st.metric(
        "STOCKOUT RISKS",
        f"{stockout_risks:,}"
    )

with c5:
    st.metric(
        "ESTIMATED IMPACT",
        f"₹{estimated_impact:,.0f}"
    )

st.divider()

# ============================================================
# DEMAND INTELLIGENCE
# ============================================================

st.header("📈 Demand Intelligence")

st.caption(
    "Historical weekly demand for the selected scope."
)

weekly = (
    filtered
    .groupby("Week", as_index=False)["Units_Sold"]
    .sum()
    .sort_values("Week")
)

if not weekly.empty:

    chart = weekly.set_index("Week")

    st.area_chart(
        chart["Units_Sold"],
        height=420
    )

else:

    st.info(
        "No demand data available for the selected filters."
    )

# ============================================================
# PRODUCT DEMAND
# ============================================================

st.header("📊 Product Demand")

product_demand = (
    filtered
    .groupby("Product ID", as_index=False)["Units_Sold"]
    .sum()
    .sort_values(
        "Units_Sold",
        ascending=False
    )
    .head(10)
)

if not product_demand.empty:

    st.bar_chart(
        product_demand.set_index("Product ID")["Units_Sold"],
        height=400
    )

else:

    st.info(
        "No product demand data available."
    )

# ============================================================
# FUTURE FORECAST
# ============================================================

st.header("🔮 Future Demand Forecast")

if not filtered_forecast.empty:

    forecast_display = filtered_forecast.copy()

    if "Forecast_Units" in forecast_display.columns:

        forecast_display["Forecast_Units"] = pd.to_numeric(
            forecast_display["Forecast_Units"],
            errors="coerce"
        )

    forecast_chart = (
        forecast_display
        .groupby(
            "Week",
            as_index=False
        )["Forecast_Units"]
        .sum()
        .sort_values("Week")
    )

    if not forecast_chart.empty:

        st.line_chart(
            forecast_chart.set_index("Week")["Forecast_Units"],
            height=420
        )

    else:

        st.info(
            "No future forecast available."
        )

else:

    st.info(
        "No forecast data available for the selected filters."
    )

# ============================================================
# INVENTORY RISK OVERVIEW
# ============================================================

st.header("⚠️ Inventory Risk Overview")

r1, r2, r3 = st.columns(3)

stockout_count = (
    filtered_risk["Stockout_Risk_Level"]
    .astype(str)
    .str.lower()
    .eq("high")
    .sum()
)

overstock_count = (
    filtered_risk["Overstock_Risk_Level"]
    .astype(str)
    .str.lower()
    .eq("high")
    .sum()
)

impact_total = pd.to_numeric(
    filtered_risk["Estimated_Impact_INR"],
    errors="coerce"
).fillna(0).sum()

with r1:

    st.metric(
        "Stockout Risk Items",
        f"{stockout_count:,}"
    )

with r2:

    st.metric(
        "Overstock Risk Items",
        f"{overstock_count:,}"
    )

with r3:

    st.metric(
        "Estimated Inventory Impact",
        f"₹{impact_total:,.0f}"
    )

# ============================================================
# DECISION DISTRIBUTION
# ============================================================

st.subheader("Inventory Decision Distribution")

if not filtered_risk.empty:

    decision_counts = (
        filtered_risk["Decision_Quadrant"]
        .value_counts()
    )

    st.bar_chart(
        decision_counts,
        height=350
    )

else:

    st.info(
        "No inventory decision data available."
    )

# ============================================================
# ACTION RECOMMENDATIONS
# ============================================================

st.header("🎯 Action Recommendations")

action_columns = [
    "Store ID",
    "Product ID",
    "Category",
    "Region",
    "Stockout_Risk_Level",
    "Overstock_Risk_Level",
    "Decision_Quadrant",
    "Recommended_Action",
    "Estimated_Impact_INR"
]

available_columns = [
    col
    for col in action_columns
    if col in filtered_risk.columns
]

if available_columns and not filtered_risk.empty:

    st.dataframe(
        filtered_risk[available_columns],
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No action recommendations available for the selected filters."
    )

# ============================================================
# DATASET SUMMARY
# ============================================================

st.header("🧭 Inventory Decision View")

with st.expander(
    "📋 Dataset Summary",
    expanded=True
):

    s1, s2, s3, s4 = st.columns(4)

    with s1:

        st.metric(
            "Weekly Records",
            f"{len(filtered):,}"
        )

    with s2:

        st.metric(
            "Unique Stores",
            f"{filtered['Store ID'].nunique():,}"
        )

    with s3:

        st.metric(
            "Unique Products",
            f"{filtered['Product ID'].nunique():,}"
        )

    with s4:

        if not filtered.empty:

            st.metric(
                "Date Range",
                f"{filtered['Week'].min()} → {filtered['Week'].max()}"
            )

        else:

            st.metric(
                "Date Range",
                "N/A"
            )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "FORESIGHT — Demand & Inventory Intelligence | Decision-support dashboard"
)