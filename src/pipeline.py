from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "retail_store_inventory.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "weekly_demand.csv"


REQUIRED_COLUMNS = [
    "Date",
    "Store ID",
    "Product ID",
    "Category",
    "Region",
    "Inventory Level",
    "Units Sold",
    "Units Ordered",
    "Price",
    "Discount",
    "Weather Condition",
    "Holiday/Promotion",
    "Competitor Pricing",
    "Seasonality",
]


def load_data():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)

    print(f"Raw rows: {len(df):,}")
    print(f"Raw columns: {len(df.columns)}")

    return df


def validate_columns(df):
    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


def clean_data(df):
    df = df.copy()

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df = df.dropna(subset=["Date"])

    numeric_columns = [
        "Inventory Level",
        "Units Sold",
        "Units Ordered",
        "Price",
        "Discount",
        "Competitor Pricing",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=numeric_columns
    )

    df = df.drop_duplicates(
        subset=["Date", "Store ID", "Product ID"]
    )

    df = df.sort_values(
        ["Store ID", "Product ID", "Date"]
    ).reset_index(drop=True)

    return df


def create_weekly_dataset(df):
    df = df.copy()

    df["Week"] = (
        df["Date"]
        .dt.to_period("W")
        .apply(lambda period: period.start_time)
    )

    weekly = (
        df.groupby(
            [
                "Week",
                "Store ID",
                "Product ID",
                "Category",
                "Region",
            ],
            as_index=False
        )
        .agg(
            Units_Sold=("Units Sold", "sum"),
            Units_Ordered=("Units Ordered", "sum"),
            Avg_Inventory=("Inventory Level", "mean"),
            Avg_Price=("Price", "mean"),
            Avg_Discount=("Discount", "mean"),
            Avg_Competitor_Price=(
                "Competitor Pricing",
                "mean"
            ),
            Promotion_Days=(
                "Holiday/Promotion",
                "sum"
            ),
        )
    )

    return weekly


def save_dataset(weekly):
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    weekly.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"Processed dataset saved to:\n{OUTPUT_PATH}"
    )


def main():
    print("=" * 70)
    print("FORESIGHT - DATA PIPELINE")
    print("=" * 70)

    df = load_data()

    validate_columns(df)

    df = clean_data(df)

    print(f"Cleaned rows: {len(df):,}")

    weekly = create_weekly_dataset(df)

    print(
        f"Weekly rows: {len(weekly):,}"
    )

    print("\nWeekly dataset columns:")
    for column in weekly.columns:
        print(f"- {column}")

    save_dataset(weekly)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()