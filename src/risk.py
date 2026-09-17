from pathlib import Path
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

WEEKLY_DATA_PATH = (
    BASE_DIR / "data" / "processed" / "weekly_demand.csv"
)

FORECAST_PATH = (
    BASE_DIR / "data" / "processed" / "future_demand_forecast.csv"
)

REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = REPORT_DIR / "risk_scoring.csv"


# ============================================================
# PROJECT ASSUMPTIONS
# ============================================================

# The source dataset does not contain lead_time_days.
# FORESIGHT requires lead-time-aware stockout analysis.
# Therefore, a transparent project assumption is used.
ASSUMED_LEAD_TIME_DAYS = 14

ASSUMED_LEAD_TIME_WEEKS = (
    ASSUMED_LEAD_TIME_DAYS // 7
)

# Risk thresholds
HIGH_RISK_THRESHOLD = 70
MEDIUM_RISK_THRESHOLD = 40


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def risk_level(score):
    if score >= HIGH_RISK_THRESHOLD:
        return "High"
    elif score >= MEDIUM_RISK_THRESHOLD:
        return "Medium"
    return "Low"


def safe_percentage(value):
    return min(max(float(value), 0.0), 100.0)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not WEEKLY_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Weekly dataset not found: {WEEKLY_DATA_PATH}"
        )

    if not FORECAST_PATH.exists():
        raise FileNotFoundError(
            f"Forecast dataset not found: {FORECAST_PATH}"
        )

    weekly = pd.read_csv(WEEKLY_DATA_PATH)
    forecast = pd.read_csv(FORECAST_PATH)

    weekly["Week"] = pd.to_datetime(
        weekly["Week"]
    )

    forecast["Week"] = pd.to_datetime(
        forecast["Week"]
    )

    return weekly, forecast


# ============================================================
# VALIDATION
# ============================================================

def validate_data(weekly, forecast):

    required_weekly = [
        "Week",
        "Store ID",
        "Product ID",
        "Units_Sold",
        "Units_Ordered",
        "Avg_Inventory",
        "Avg_Price"
    ]

    required_forecast = [
        "Week",
        "Store ID",
        "Product ID",
        "Forecast_Units"
    ]

    missing_weekly = [
        col for col in required_weekly
        if col not in weekly.columns
    ]

    missing_forecast = [
        col for col in required_forecast
        if col not in forecast.columns
    ]

    if missing_weekly:
        raise ValueError(
            f"Missing weekly columns: {missing_weekly}"
        )

    if missing_forecast:
        raise ValueError(
            f"Missing forecast columns: {missing_forecast}"
        )

    if weekly[
        ["Store ID", "Product ID"]
    ].isnull().any().any():

        raise ValueError(
            "Missing Store ID or Product ID in weekly dataset."
        )

    if forecast[
        ["Store ID", "Product ID"]
    ].isnull().any().any():

        raise ValueError(
            "Missing Store ID or Product ID in forecast dataset."
        )

    if (weekly["Avg_Inventory"] < 0).any():
        raise ValueError(
            "Negative inventory values found."
        )

    if (forecast["Forecast_Units"] < 0).any():
        raise ValueError(
            "Negative forecast demand found."
        )

    if (weekly["Units_Sold"] < 0).any():
        raise ValueError(
            "Negative sales values found."
        )


# ============================================================
# CURRENT INVENTORY
# ============================================================

def get_current_inventory(weekly):

    latest_week = weekly["Week"].max()

    latest = weekly[
        weekly["Week"] == latest_week
    ].copy()

    # One record per Store + Product
    current = (
        latest
        .groupby(
            ["Store ID", "Product ID"],
            as_index=False
        )
        .agg(
            Category=("Category", "first"),
            Region=("Region", "first"),
            Current_Inventory=(
                "Avg_Inventory",
                "mean"
            ),
            Units_Ordered=(
                "Units_Ordered",
                "sum"
            ),
            Current_Price=(
                "Avg_Price",
                "mean"
            )
        )
    )

    current["Current_Inventory"] = (
        current["Current_Inventory"]
        .clip(lower=0)
    )

    current["Units_Ordered"] = (
        current["Units_Ordered"]
        .clip(lower=0)
    )

    current["Current_Price"] = (
        current["Current_Price"]
        .clip(lower=0)
    )

    return current, latest_week


# ============================================================
# FORWARD DEMAND
# ============================================================

def calculate_forward_demand(forecast):

    forecast = forecast.copy()

    forecast = forecast.sort_values(
        ["Store ID", "Product ID", "Week"]
    )

    # Total demand expected over the complete
    # 8-week forecast horizon.
    forward = (
        forecast
        .groupby(
            ["Store ID", "Product ID"],
            as_index=False
        )
        .agg(
            Forward_8_Week_Demand=(
                "Forecast_Units",
                "sum"
            )
        )
    )

    # Lead-time demand:
    # first 2 forecast weeks because assumed lead time = 14 days.
    first_weeks = (
        forecast
        .sort_values(
            ["Store ID", "Product ID", "Week"]
        )
        .groupby(
            ["Store ID", "Product ID"]
        )
        .head(ASSUMED_LEAD_TIME_WEEKS)
    )

    lead_time = (
        first_weeks
        .groupby(
            ["Store ID", "Product ID"],
            as_index=False
        )
        .agg(
            Lead_Time_Demand=(
                "Forecast_Units",
                "sum"
            )
        )
    )

    forward = forward.merge(
        lead_time,
        on=["Store ID", "Product ID"],
        how="left"
    )

    forward["Forward_8_Week_Demand"] = (
        forward["Forward_8_Week_Demand"]
        .clip(lower=0)
    )

    forward["Lead_Time_Demand"] = (
        forward["Lead_Time_Demand"]
        .clip(lower=0)
    )

    return forward


# ============================================================
# STOCKOUT RISK
# ============================================================

def calculate_stockout_risk(df):

    # Available inventory includes current inventory
    # plus units already ordered.
    df["Available_Supply"] = (
        df["Current_Inventory"]
        + df["Units_Ordered"]
    )

    df["Stockout_Gap_Units"] = (
        df["Lead_Time_Demand"]
        - df["Available_Supply"]
    )

    df["Stockout_Gap_Units"] = (
        df["Stockout_Gap_Units"]
        .clip(lower=0)
    )

    # Risk is based on the percentage of lead-time
    # demand that cannot be covered.
    df["stockout_risk_score"] = np.where(
        df["Lead_Time_Demand"] > 0,
        (
            df["Stockout_Gap_Units"]
            / df["Lead_Time_Demand"]
        ) * 100,
        0
    )

    df["stockout_risk_score"] = (
        df["stockout_risk_score"]
        .apply(safe_percentage)
        .round(2)
    )

    df["Stockout_Risk_Level"] = (
        df["stockout_risk_score"]
        .apply(risk_level)
    )

    return df


# ============================================================
# OVERSTOCK RISK
# ============================================================

def calculate_overstock_risk(df):

    df["Excess_Inventory_Units"] = (
        df["Current_Inventory"]
        - df["Forward_8_Week_Demand"]
    )

    df["Excess_Inventory_Units"] = (
        df["Excess_Inventory_Units"]
        .clip(lower=0)
    )

    df["overstock_risk_score"] = np.where(
        df["Current_Inventory"] > 0,
        (
            df["Excess_Inventory_Units"]
            / df["Current_Inventory"]
        ) * 100,
        0
    )

    df["overstock_risk_score"] = (
        df["overstock_risk_score"]
        .apply(safe_percentage)
        .round(2)
    )

    df["Overstock_Risk_Level"] = (
        df["overstock_risk_score"]
        .apply(risk_level)
    )

    return df


# ============================================================
# RUPEE IMPACT
# ============================================================

def calculate_impact(df):

    # Estimated lost-sales value if lead-time demand
    # cannot be covered.
    df["Estimated_Stockout_Impact_INR"] = (
        df["Stockout_Gap_Units"]
        * df["Current_Price"]
    )

    # Estimated value of inventory above the complete
    # 8-week forecast demand.
    df["Estimated_Overstock_Impact_INR"] = (
        df["Excess_Inventory_Units"]
        * df["Current_Price"]
    )

    # Combined estimated exposure.
    df["Estimated_Impact_INR"] = (
        df["Estimated_Stockout_Impact_INR"]
        + df["Estimated_Overstock_Impact_INR"]
    )

    df[
        [
            "Estimated_Stockout_Impact_INR",
            "Estimated_Overstock_Impact_INR",
            "Estimated_Impact_INR"
        ]
    ] = df[
        [
            "Estimated_Stockout_Impact_INR",
            "Estimated_Overstock_Impact_INR",
            "Estimated_Impact_INR"
        ]
    ].round(2)

    return df


# ============================================================
# DECISION QUADRANTS
# ============================================================

def assign_decision(row):

    stockout_high = (
        row["stockout_risk_score"]
        >= HIGH_RISK_THRESHOLD
    )

    overstock_high = (
        row["overstock_risk_score"]
        >= HIGH_RISK_THRESHOLD
    )

    if stockout_high and not overstock_high:
        return "Reorder Now"

    if overstock_high and not stockout_high:
        return "Markdown / Clear"

    if stockout_high and overstock_high:
        return "Watch / Volatile"

    return "Healthy"


def assign_action(row):

    decision = row["Decision_Quadrant"]

    if decision == "Reorder Now":
        return (
            "Prioritize replenishment; "
            "available supply may not cover lead-time demand."
        )

    if decision == "Markdown / Clear":
        return (
            "Review excess inventory and consider "
            "markdown or clearance action."
        )

    if decision == "Watch / Volatile":
        return (
            "Monitor closely; both stockout and "
            "overstock signals require review."
        )

    return (
        "Maintain current inventory position "
        "and continue monitoring."
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

def create_risk_table(weekly, forecast):

    current, latest_week = get_current_inventory(
        weekly
    )

    forward = calculate_forward_demand(
        forecast
    )

    df = current.merge(
        forward,
        on=["Store ID", "Product ID"],
        how="inner"
    )

    df = calculate_stockout_risk(df)

    df = calculate_overstock_risk(df)

    df = calculate_impact(df)

    df["Decision_Quadrant"] = (
        df.apply(
            assign_decision,
            axis=1
        )
    )

    df["Recommended_Action"] = (
        df.apply(
            assign_action,
            axis=1
        )
    )

    df["Analysis_Week"] = latest_week

    # Final validation: one row per Store + Product
    duplicates = df.duplicated(
        subset=["Store ID", "Product ID"]
    ).sum()

    if duplicates > 0:
        raise ValueError(
            f"Final output contains {duplicates} "
            "duplicate Store + Product records."
        )

    return df


# ============================================================
# VALIDATE FINAL OUTPUT
# ============================================================

def validate_output(df):

    if df.empty:
        raise ValueError(
            "Risk scoring output is empty."
        )

    if df[
        ["Store ID", "Product ID"]
    ].isnull().any().any():

        raise ValueError(
            "Final output contains missing identifiers."
        )

    numeric_columns = [
        "Current_Inventory",
        "Units_Ordered",
        "Forward_8_Week_Demand",
        "Lead_Time_Demand",
        "Stockout_Gap_Units",
        "Excess_Inventory_Units",
        "stockout_risk_score",
        "overstock_risk_score"
    ]

    for col in numeric_columns:

        if (df[col] < 0).any():
            raise ValueError(
                f"Negative values found in {col}."
            )

    valid_quadrants = {
        "Reorder Now",
        "Markdown / Clear",
        "Watch / Volatile",
        "Healthy"
    }

    invalid_quadrants = set(
        df["Decision_Quadrant"].unique()
    ) - valid_quadrants

    if invalid_quadrants:
        raise ValueError(
            f"Invalid decision quadrants: "
            f"{invalid_quadrants}"
        )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(df):

    print("\n" + "=" * 70)
    print("FORESIGHT - RISK SCORING SUMMARY")
    print("=" * 70)

    print(
        f"\nTotal Store + Product combinations: "
        f"{len(df):,}"
    )

    print(
        f"High Stockout Risk: "
        f"{(df['Stockout_Risk_Level'] == 'High').sum():,}"
    )

    print(
        f"Medium Stockout Risk: "
        f"{(df['Stockout_Risk_Level'] == 'Medium').sum():,}"
    )

    print(
        f"Low Stockout Risk: "
        f"{(df['Stockout_Risk_Level'] == 'Low').sum():,}"
    )

    print(
        f"\nHigh Overstock Risk: "
        f"{(df['Overstock_Risk_Level'] == 'High').sum():,}"
    )

    print(
        f"Medium Overstock Risk: "
        f"{(df['Overstock_Risk_Level'] == 'Medium').sum():,}"
    )

    print(
        f"Low Overstock Risk: "
        f"{(df['Overstock_Risk_Level'] == 'Low').sum():,}"
    )

    print("\nDecision Quadrants:")

    for quadrant in [
        "Reorder Now",
        "Markdown / Clear",
        "Watch / Volatile",
        "Healthy"
    ]:
        count = (
            df["Decision_Quadrant"] == quadrant
        ).sum()

        print(
            f"  {quadrant}: {count:,}"
        )

    print(
        f"\nTotal Estimated Stockout Impact: "
        f"₹{df['Estimated_Stockout_Impact_INR'].sum():,.2f}"
    )

    print(
        f"Total Estimated Overstock Value: "
        f"₹{df['Estimated_Overstock_Impact_INR'].sum():,.2f}"
    )

    print(
        f"Total Estimated Impact: "
        f"₹{df['Estimated_Impact_INR'].sum():,.2f}"
    )

    print(
        f"\nAssumed Lead Time: "
        f"{ASSUMED_LEAD_TIME_DAYS} days "
        f"({ASSUMED_LEAD_TIME_WEEKS} weeks)"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FORESIGHT - INVENTORY RISK SCORING")
    print("=" * 70)

    print(
        f"\nWeekly dataset: {WEEKLY_DATA_PATH}"
    )

    print(
        f"Forecast dataset: {FORECAST_PATH}"
    )

    weekly, forecast = load_data()

    print(
        f"\nWeekly rows: {len(weekly):,}"
    )

    print(
        f"Forecast rows: {len(forecast):,}"
    )

    validate_data(
        weekly,
        forecast
    )

    print(
        "\nData validation: PASSED"
    )

    risk_df = create_risk_table(
        weekly,
        forecast
    )

    validate_output(
        risk_df
    )

    risk_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"\nRisk scoring saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print_summary(
        risk_df
    )

    print("\n" + "=" * 70)
    print("RISK SCORING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()