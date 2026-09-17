from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "weekly_demand.csv"
REPORT_DIR = BASE_DIR / "reports"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


TARGET = "Units_Sold"
GROUP_COLS = ["Store ID", "Product ID"]

FORECAST_HORIZON = 8
SEASONAL_LAG = 4

LAG_WINDOWS = [1, 2, 4, 8, 13, 26, 52]
ROLLING_WINDOWS = [4, 8, 13]


def wape(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    denominator = np.sum(np.abs(actual))

    if denominator == 0:
        return np.nan

    return np.sum(np.abs(actual - predicted)) / denominator * 100


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if "Week" not in df.columns:
        raise ValueError("Required column 'Week' not found.")

    if TARGET not in df.columns:
        raise ValueError(
            f"Required target column '{TARGET}' not found."
        )

    df["Week"] = pd.to_datetime(df["Week"])

    df = df.sort_values(
        GROUP_COLS + ["Week"]
    ).reset_index(drop=True)

    return df


def create_features(df):
    df = df.copy()

    df = df.sort_values(
        GROUP_COLS + ["Week"]
    ).reset_index(drop=True)

    grouped_target = df.groupby(GROUP_COLS)[TARGET]

    for lag in LAG_WINDOWS:
        df[f"lag_{lag}"] = grouped_target.shift(lag)

    for window in ROLLING_WINDOWS:
        df[f"rolling_mean_{window}"] = (
            df.groupby(GROUP_COLS)[TARGET]
            .transform(
                lambda x: x.shift(1).rolling(
                    window,
                    min_periods=window
                ).mean()
            )
        )

        df[f"rolling_std_{window}"] = (
            df.groupby(GROUP_COLS)[TARGET]
            .transform(
                lambda x: x.shift(1).rolling(
                    window,
                    min_periods=window
                ).std()
            )
        )

    df["week_number"] = df["Week"].dt.isocalendar().week.astype(int)
    df["month"] = df["Week"].dt.month

    df["sin_week"] = np.sin(
        2 * np.pi * df["week_number"] / 52
    )

    df["cos_week"] = np.cos(
        2 * np.pi * df["week_number"] / 52
    )

    df["sin_month"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["cos_month"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    return df


def prepare_model_data(df):
    df = df.copy()

    categorical_cols = [
        "Store ID",
        "Product ID",
        "Category",
        "Region"
    ]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype("category")
                .cat.codes
            )

    feature_cols = [
        "Store ID",
        "Product ID",
        "Category",
        "Region",
        "Units_Ordered",
        "Avg_Inventory",
        "Avg_Price",
        "Avg_Discount",
        "Avg_Competitor_Price",
        "Promotion_Days",
        "week_number",
        "month",
        "sin_week",
        "cos_week",
        "sin_month",
        "cos_month",
    ]

    for lag in LAG_WINDOWS:
        feature_cols.append(f"lag_{lag}")

    for window in ROLLING_WINDOWS:
        feature_cols.append(f"rolling_mean_{window}")
        feature_cols.append(f"rolling_std_{window}")

    feature_cols = [
        col for col in feature_cols
        if col in df.columns
    ]

    return df, feature_cols


def train_model(X, y):
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=14,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    return model


def rolling_origin_backtest(df):
    print("\n" + "=" * 70)
    print("ROLLING-ORIGIN FORECAST BACKTEST")
    print("=" * 70)

    feature_df = create_features(df)

    feature_df, feature_cols = prepare_model_data(
        feature_df
    )

    valid_df = feature_df.dropna(
        subset=feature_cols + [TARGET]
    ).copy()

    unique_weeks = sorted(
        valid_df["Week"].unique()
    )

    if len(unique_weeks) < 60:
        raise ValueError(
            "Not enough weekly history for backtesting."
        )

    test_size = 8

    fold_end_positions = [
        len(unique_weeks) - test_size * 3,
        len(unique_weeks) - test_size * 2,
        len(unique_weeks) - test_size
    ]

    all_results = []

    for fold_number, end_position in enumerate(
        fold_end_positions,
        start=1
    ):
        train_end = unique_weeks[
            end_position - 1
        ]

        test_weeks = unique_weeks[
            end_position:
            end_position + test_size
        ]

        train_data = valid_df[
            valid_df["Week"] <= train_end
        ].copy()

        test_data = valid_df[
            valid_df["Week"].isin(test_weeks)
        ].copy()

        if train_data.empty or test_data.empty:
            continue

        X_train = train_data[feature_cols]
        y_train = train_data[TARGET]

        X_test = test_data[feature_cols]
        y_test = test_data[TARGET]

        model = train_model(
            X_train,
            y_train
        )

        predictions = model.predict(X_test)

        test_data["ML_Prediction"] = np.maximum(
            predictions,
            0
        )

        baseline = (
            test_data
            .groupby(GROUP_COLS)[TARGET]
            .shift(SEASONAL_LAG)
        )

        test_data["Baseline_Prediction"] = baseline

        test_data = test_data.dropna(
            subset=["Baseline_Prediction"]
        )

        ml_wape = wape(
            test_data[TARGET],
            test_data["ML_Prediction"]
        )

        baseline_wape = wape(
            test_data[TARGET],
            test_data["Baseline_Prediction"]
        )

        ml_mae = mean_absolute_error(
            test_data[TARGET],
            test_data["ML_Prediction"]
        )

        baseline_mae = mean_absolute_error(
            test_data[TARGET],
            test_data["Baseline_Prediction"]
        )

        print(
            f"\nFold {fold_number}"
        )

        print(
            f"Train through: "
            f"{pd.Timestamp(train_end).date()}"
        )

        print(
            f"Test period: "
            f"{pd.Timestamp(test_weeks[0]).date()} "
            f"to "
            f"{pd.Timestamp(test_weeks[-1]).date()}"
        )

        print(
            f"ML WAPE: {ml_wape:.2f}%"
        )

        print(
            f"Baseline WAPE: {baseline_wape:.2f}%"
        )

        print(
            f"ML MAE: {ml_mae:.2f}"
        )

        print(
            f"Baseline MAE: {baseline_mae:.2f}"
        )

        test_data["Fold"] = fold_number

        all_results.append(
            test_data[
                GROUP_COLS
                + [
                    "Week",
                    TARGET,
                    "ML_Prediction",
                    "Baseline_Prediction",
                    "Fold"
                ]
            ]
        )

    if not all_results:
        raise ValueError(
            "Backtesting produced no results."
        )

    results = pd.concat(
        all_results,
        ignore_index=True
    )

    results_path = (
        REPORT_DIR /
        "forecast_backtest_results.csv"
    )

    results.to_csv(
        results_path,
        index=False
    )

    average_ml_wape = wape(
        results[TARGET],
        results["ML_Prediction"]
    )

    average_baseline_wape = wape(
        results[TARGET],
        results["Baseline_Prediction"]
    )

    average_ml_mae = mean_absolute_error(
        results[TARGET],
        results["ML_Prediction"]
    )

    print("\n" + "-" * 70)
    print("BACKTEST SUMMARY")
    print("-" * 70)

    print(
        f"Average ML WAPE: "
        f"{average_ml_wape:.2f}%"
    )

    print(
        f"Average Baseline WAPE: "
        f"{average_baseline_wape:.2f}%"
    )

    print(
        f"Average ML MAE: "
        f"{average_ml_mae:.2f}"
    )

    print(
        f"\nSaved: {results_path}"
    )

    return (
        feature_df,
        feature_cols,
        results
    )


def train_final_model(feature_df, feature_cols):
    model_data = feature_df.dropna(
        subset=feature_cols + [TARGET]
    ).copy()

    X = model_data[feature_cols]
    y = model_data[TARGET]

    print("\n" + "=" * 70)
    print("TRAINING FINAL MODEL")
    print("=" * 70)

    print(
        f"Training rows: {len(model_data):,}"
    )

    print(
        f"Features: {len(feature_cols)}"
    )

    model = train_model(X, y)

    model_path = (
        PROCESSED_DIR /
        "foresight_forecast_model.joblib"
    )

    joblib.dump(
        {
            "model": model,
            "features": feature_cols,
            "group_cols": GROUP_COLS,
            "target": TARGET
        },
        model_path
    )

    print(
        f"Model saved: {model_path}"
    )

    return model


def generate_future_forecast(
    original_df,
    model,
    feature_cols
):
    print("\n" + "=" * 70)
    print("GENERATING 8-WEEK FUTURE FORECAST")
    print("=" * 70)

    history = original_df.copy()

    history["Week"] = pd.to_datetime(
        history["Week"]
    )

    history = history.sort_values(
        GROUP_COLS + ["Week"]
    ).reset_index(drop=True)

    latest_week = history["Week"].max()

    forecasts = []

    groups = (
        history[GROUP_COLS]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    for _, group in groups.iterrows():

        store_id = group["Store ID"]
        product_id = group["Product ID"]

        series = history[
            (history["Store ID"] == store_id)
            & (history["Product ID"] == product_id)
        ].copy()

        if len(series) < 55:
            continue

        series = series.sort_values(
            "Week"
        ).reset_index(drop=True)

        static_values = {}

        for col in [
            "Category",
            "Region"
        ]:
            if col in series.columns:
                static_values[col] = (
                    series[col].iloc[-1]
                )

        for step in range(1, FORECAST_HORIZON + 1):

            future_week = (
                latest_week +
                pd.Timedelta(
                    weeks=step
                )
            )

            temp = series.copy()

            future_row = {
                "Week": future_week,
                "Store ID": store_id,
                "Product ID": product_id,
                TARGET: np.nan
            }

            for col, value in static_values.items():
                future_row[col] = value

            numeric_defaults = [
                "Units_Ordered",
                "Avg_Inventory",
                "Avg_Price",
                "Avg_Discount",
                "Avg_Competitor_Price",
                "Promotion_Days"
            ]

            for col in numeric_defaults:
                if col in series.columns:
                    future_row[col] = (
                        series[col]
                        .tail(8)
                        .mean()
                    )

            temp = pd.concat(
                [
                    temp,
                    pd.DataFrame([future_row])
                ],
                ignore_index=True
            )

            temp = temp.sort_values(
                "Week"
            ).reset_index(drop=True)

            temp = create_features(
                temp
            )

            prediction_row = temp.tail(1).copy()

            prediction_row, _ = (
                prepare_model_data(
                    prediction_row
                )
            )

            missing_features = [
                col
                for col in feature_cols
                if col not in prediction_row.columns
            ]

            if missing_features:
                for col in missing_features:
                    prediction_row[col] = 0

            X_future = prediction_row[
                feature_cols
            ]

            X_future = X_future.fillna(0)

            prediction = model.predict(
                X_future
            )[0]

            prediction = max(
                float(prediction),
                0
            )

            future_row[TARGET] = prediction

            forecasts.append(
                {
                    "Week": future_week,
                    "Store ID": store_id,
                    "Product ID": product_id,
                    "Category": static_values.get(
                        "Category",
                        "Unknown"
                    ),
                    "Region": static_values.get(
                        "Region",
                        "Unknown"
                    ),
                    "Forecast_Units": round(
                        prediction,
                        2
                    )
                }
            )

            series = pd.concat(
                [
                    series,
                    pd.DataFrame(
                        [future_row]
                    )
                ],
                ignore_index=True
            )

    forecast_df = pd.DataFrame(
        forecasts
    )

    if forecast_df.empty:
        raise ValueError(
            "Future forecast generation failed."
        )

    forecast_path = (
        PROCESSED_DIR /
        "future_demand_forecast.csv"
    )

    forecast_df.to_csv(
        forecast_path,
        index=False
    )

    print(
        f"Forecast rows: "
        f"{len(forecast_df):,}"
    )

    print(
        f"Forecast weeks: "
        f"{forecast_df['Week'].min().date()} "
        f"to "
        f"{forecast_df['Week'].max().date()}"
    )

    print(
        f"Saved: {forecast_path}"
    )

    return forecast_df


def save_forecast_summary(
    forecast_df,
    backtest_results
):
    summary = {
        "Model": "Random Forest Regressor",
        "Forecast Horizon Weeks": FORECAST_HORIZON,
        "Seasonal Baseline Lag Weeks": SEASONAL_LAG,
        "Backtest WAPE": round(
            wape(
                backtest_results[TARGET],
                backtest_results["ML_Prediction"]
            ),
            2
        ),
        "Baseline WAPE": round(
            wape(
                backtest_results[TARGET],
                backtest_results["Baseline_Prediction"]
            ),
            2
        ),
        "Forecast Rows": len(forecast_df),
        "Unique Stores": forecast_df[
            "Store ID"
        ].nunique(),
        "Unique Products": forecast_df[
            "Product ID"
        ].nunique()
    }

    summary_df = pd.DataFrame(
        [summary]
    )

    summary_path = (
        REPORT_DIR /
        "forecast_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print(
        f"Saved: {summary_path}"
    )


def main():

    print("=" * 70)
    print("FORESIGHT - DEMAND FORECASTING")
    print("=" * 70)

    print(
        f"\nDataset: {DATA_PATH}"
    )

    df = load_data()

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Unique stores: "
        f"{df['Store ID'].nunique()}"
    )

    print(
        f"Unique products: "
        f"{df['Product ID'].nunique()}"
    )

    print(
        f"Date range: "
        f"{df['Week'].min().date()} "
        f"to "
        f"{df['Week'].max().date()}"
    )

    if "Demand Forecast" in df.columns:
        print(
            "\nLeakage protection:"
        )
        print(
            "Demand Forecast column detected "
            "- NOT used as model feature."
        )

    feature_df, feature_cols, backtest_results = (
        rolling_origin_backtest(df)
    )

    model = train_final_model(
        feature_df,
        feature_cols
    )

    forecast_df = generate_future_forecast(
        df,
        model,
        feature_cols
    )

    save_forecast_summary(
        forecast_df,
        backtest_results
    )

    print("\n" + "=" * 70)
    print("FORECASTING COMPLETE")
    print("=" * 70)

    print(
        "\nGenerated files:"
    )

    print(
        "1. reports/forecast_backtest_results.csv"
    )

    print(
        "2. reports/forecast_summary.csv"
    )

    print(
        "3. data/processed/future_demand_forecast.csv"
    )

    print(
        "4. data/processed/"
        "foresight_forecast_model.joblib"
    )


if __name__ == "__main__":
    main()