import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path("data/processed/weekly_demand.csv")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Week"] = pd.to_datetime(df["Week"])
    return df


def main():
    print("=" * 70)
    print("FORESIGHT - EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    df = load_data()

    print("\n1. DATASET OVERVIEW")
    print("-" * 40)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Date range: {df['Week'].min().date()} to {df['Week'].max().date()}")
    print(f"Stores: {df['Store ID'].nunique()}")
    print(f"Products: {df['Product ID'].nunique()}")
    print(f"Categories: {df['Category'].nunique()}")
    print(f"Regions: {df['Region'].nunique()}")

    print("\n2. MISSING VALUES")
    print("-" * 40)
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.any() else "No missing values found.")

    print("\n3. BASIC DEMAND STATISTICS")
    print("-" * 40)
    print(df["Units_Sold"].describe())

    print("\n4. DEMAND BY PRODUCT")
    print("-" * 40)
    product_demand = (
        df.groupby("Product ID")["Units_Sold"]
        .sum()
        .sort_values(ascending=False)
    )
    print(product_demand.to_string())

    print("\n5. DEMAND BY CATEGORY")
    print("-" * 40)
    category_demand = (
        df.groupby("Category")["Units_Sold"]
        .sum()
        .sort_values(ascending=False)
    )
    print(category_demand.to_string())

    print("\n6. DEMAND BY STORE")
    print("-" * 40)
    store_demand = (
        df.groupby("Store ID")["Units_Sold"]
        .sum()
        .sort_values(ascending=False)
    )
    print(store_demand.to_string())

    print("\n7. INVENTORY ANALYSIS")
    print("-" * 40)
    print(df["Avg_Inventory"].describe())

    inventory_summary = df[
        ["Units_Sold", "Units_Ordered", "Avg_Inventory"]
    ].corr()

    print("\nInventory / Demand Correlation:")
    print(inventory_summary)

    print("\n8. PROMOTION ANALYSIS")
    print("-" * 40)

    promotion_summary = (
        df.groupby("Promotion_Days")["Units_Sold"]
        .agg(["mean", "sum", "count"])
    )

    print(promotion_summary)

    print("\n9. SEASONALITY ANALYSIS")
    print("-" * 40)

    df["Month"] = df["Week"].dt.month
    df["Quarter"] = df["Week"].dt.quarter

    monthly_demand = (
        df.groupby("Month")["Units_Sold"]
        .mean()
        .sort_index()
    )

    quarterly_demand = (
        df.groupby("Quarter")["Units_Sold"]
        .mean()
        .sort_index()
    )

    print("Average demand by month:")
    print(monthly_demand)

    print("\nAverage demand by quarter:")
    print(quarterly_demand)

    print("\n10. PRICE AND DISCOUNT ANALYSIS")
    print("-" * 40)

    print("Price statistics:")
    print(df["Avg_Price"].describe())

    print("\nDiscount statistics:")
    print(df["Avg_Discount"].describe())

    print("\n11. OUTLIER CHECK")
    print("-" * 40)

    q1 = df["Units_Sold"].quantile(0.25)
    q3 = df["Units_Sold"].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = df[
        (df["Units_Sold"] < lower_bound)
        | (df["Units_Sold"] > upper_bound)
    ]

    print(f"Q1: {q1:.2f}")
    print(f"Q3: {q3:.2f}")
    print(f"IQR: {iqr:.2f}")
    print(f"Lower bound: {lower_bound:.2f}")
    print(f"Upper bound: {upper_bound:.2f}")
    print(f"Demand outliers: {len(outliers):,}")

    print("\n12. WEEKLY DEMAND TREND")
    print("-" * 40)

    weekly_total = (
        df.groupby("Week")["Units_Sold"]
        .sum()
        .sort_index()
    )

    plt.figure(figsize=(12, 6))
    plt.plot(weekly_total.index, weekly_total.values)
    plt.title("Weekly Total Demand")
    plt.xlabel("Week")
    plt.ylabel("Units Sold")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "weekly_demand_trend.png", dpi=150)
    plt.close()

    print("Saved: reports/weekly_demand_trend.png")

    print("\n13. CATEGORY DEMAND CHART")
    print("-" * 40)

    plt.figure(figsize=(10, 6))
    category_demand.plot(kind="bar")
    plt.title("Total Demand by Category")
    plt.xlabel("Category")
    plt.ylabel("Units Sold")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "category_demand.png", dpi=150)
    plt.close()

    print("Saved: reports/category_demand.png")

    print("\n14. TOP PRODUCTS")
    print("-" * 40)

    top_products = product_demand.head(10)

    plt.figure(figsize=(10, 6))
    top_products.sort_values().plot(kind="barh")
    plt.title("Top 10 Products by Demand")
    plt.xlabel("Units Sold")
    plt.ylabel("Product ID")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "top_products.png", dpi=150)
    plt.close()

    print("Saved: reports/top_products.png")

    print("\n15. EDA SUMMARY")
    print("-" * 40)

    summary = {
        "total_rows": len(df),
        "total_stores": df["Store ID"].nunique(),
        "total_products": df["Product ID"].nunique(),
        "total_categories": df["Category"].nunique(),
        "total_regions": df["Region"].nunique(),
        "total_units_sold": int(df["Units_Sold"].sum()),
        "average_weekly_demand": round(df["Units_Sold"].mean(), 2),
        "average_inventory": round(df["Avg_Inventory"].mean(), 2),
        "demand_outliers": len(outliers),
    }

    for key, value in summary.items():
        print(f"{key}: {value}")

    summary_df = pd.DataFrame(
        list(summary.items()),
        columns=["Metric", "Value"]
    )

    summary_df.to_csv(
        REPORT_DIR / "eda_summary.csv",
        index=False
    )

    print("\nSaved: reports/eda_summary.csv")

    print("\n" + "=" * 70)
    print("EDA COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()