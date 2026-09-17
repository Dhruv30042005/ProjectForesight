from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "retail_store_inventory.csv"
REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 70)
    print("FORESIGHT - DATA INSPECTION & QUALITY AUDIT")
    print("=" * 70)

    print(f"\nDataset path:\n{DATA_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    print("\n1. DATASET DIMENSIONS")
    print("-" * 40)
    print(f"Rows    : {df.shape[0]:,}")
    print(f"Columns : {df.shape[1]}")

    print("\n2. COLUMN NAMES")
    print("-" * 40)
    for column in df.columns:
        print(f"- {column}")

    print("\n3. DATA TYPES")
    print("-" * 40)
    print(df.dtypes.to_string())

    print("\n4. MISSING VALUES")
    print("-" * 40)
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("No missing values found.")
    else:
        print(missing.to_string())

    print("\n5. DUPLICATE ROWS")
    print("-" * 40)
    print(f"Duplicate rows: {df.duplicated().sum():,}")

    print("\n6. UNIQUE VALUES")
    print("-" * 40)

    for column in ["Store ID", "Product ID", "Category", "Region"]:
        if column in df.columns:
            print(f"{column}: {df[column].nunique()}")

    print("\n7. DATE RANGE")
    print("-" * 40)

    if "Date" in df.columns:
        dates = pd.to_datetime(df["Date"], errors="coerce")
        print(f"Minimum date: {dates.min()}")
        print(f"Maximum date: {dates.max()}")
        print(f"Invalid dates: {dates.isna().sum():,}")

    print("\n8. NUMERICAL SUMMARY")
    print("-" * 40)
    print(df.describe().to_string())

    print("\n9. NEGATIVE VALUES CHECK")
    print("-" * 40)

    numeric_columns = [
        "Inventory Level",
        "Units Sold",
        "Units Ordered",
        "Demand Forecast",
        "Price",
        "Discount",
        "Competitor Pricing",
    ]

    for column in numeric_columns:
        if column in df.columns:
            count = (df[column] < 0).sum()
            print(f"{column}: {count:,} negative values")

    print("\n10. ZERO / NON-POSITIVE PRICE CHECK")
    print("-" * 40)

    if "Price" in df.columns:
        print(f"Price <= 0: {(df['Price'] <= 0).sum():,}")

    print("\n11. DATE-STORE-PRODUCT DUPLICATES")
    print("-" * 40)

    required = ["Date", "Store ID", "Product ID"]

    if all(column in df.columns for column in required):
        combination_duplicates = df.duplicated(
            subset=required
        ).sum()

        print(
            f"Duplicate Date + Store + Product records: "
            f"{combination_duplicates:,}"
        )

    print("\n12. RECORDS PER STORE")
    print("-" * 40)

    if "Store ID" in df.columns:
        print(df["Store ID"].value_counts().sort_index().to_string())

    print("\n13. RECORDS PER PRODUCT")
    print("-" * 40)

    if "Product ID" in df.columns:
        print(df["Product ID"].value_counts().sort_index().to_string())

    print("\n14. DEMAND FORECAST LEAKAGE CHECK")
    print("-" * 40)

    if "Demand Forecast" in df.columns:
        print(
            "Demand Forecast column FOUND."
        )
        print(
            "IMPORTANT: This column will NOT be used as a "
            "model feature."
        )
        print(
            "It will be treated as a pre-existing forecast field "
            "and investigated separately."
        )

    print("\n15. FORESIGHT FIELD MAPPING")
    print("-" * 40)

    mapping = {
        "Date": "Date",
        "SKU / Product": "Product ID",
        "Demand / Sales": "Units Sold",
        "Inventory": "Inventory Level",
        "Units Ordered": "Units Ordered",
        "Price": "Price",
        "Promotion": "Promotion",
        "Seasonality": "Seasonality",
        "Store": "Store ID",
        "Category": "Category",
        "Region": "Region",
        "Lead Time": "NOT AVAILABLE",
        "Reorder Point": "NOT AVAILABLE",
    }

    for requirement, column in mapping.items():
        status = "AVAILABLE" if column in df.columns else column
        print(f"{requirement:20} -> {status}")

    print("\n" + "=" * 70)
    print("DATA INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()