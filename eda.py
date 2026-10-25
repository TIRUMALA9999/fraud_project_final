"""Exploratory Data Analysis for FraudGuard."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_PATH = Path("data/transactions.csv")
REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    # 1. LOAD DATASET

    df = pd.read_csv(DATA_PATH)

    print("\n========== DATASET INFORMATION ==========")

    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    print("\nColumn names:")
    print(df.columns.tolist())

    # 2. DATA QUALITY CHECKS

    print("\n========== DATA QUALITY ==========")

    missing_values = int(df.isnull().sum().sum())

    duplicate_transactions = int(
        df["transaction_id"].duplicated().sum()
    )

    invalid_amounts = int(
        (df["amount_usd"] <= 0).sum()
    )

    invalid_geo_velocity = int(
        (df["geo_velocity_kmh"] < 0).sum()
    )

    invalid_coordinates = int(
        (
            ~df["latitude"].between(-90, 90)
            | ~df["longitude"].between(-180, 180)
        ).sum()
    )

    print("Missing values:", missing_values)
    print("Duplicate transaction IDs:", duplicate_transactions)
    print("Invalid amounts:", invalid_amounts)
    print("Negative geo velocities:", invalid_geo_velocity)
    print("Invalid coordinates:", invalid_coordinates)

    # Parse timestamp and check ordering.
    df["event_timestamp_utc"] = pd.to_datetime(
        df["event_timestamp_utc"],
        utc=True,
        errors="coerce",
    )

    invalid_timestamps = int(
        df["event_timestamp_utc"].isna().sum()
    )

    print("Invalid timestamps:", invalid_timestamps)

    if (
        missing_values > 0
        or duplicate_transactions > 0
        or invalid_amounts > 0
        or invalid_geo_velocity > 0
        or invalid_coordinates > 0
        or invalid_timestamps > 0
    ):
        raise ValueError(
            "Data validation failed. Check the dataset."
        )

    if not df["event_timestamp_utc"].is_monotonic_increasing:
        raise ValueError(
            "Transactions are not chronologically ordered."
        )

    if set(df["is_fraud"].unique()) != {0, 1}:
        raise ValueError(
            "Fraud labels must contain only 0 and 1."
        )

    print("\nData quality checks passed!")

    # 3. CLASS DISTRIBUTION

    print("\n========== FRAUD DISTRIBUTION ==========")

    counts = (
        df["is_fraud"]
        .value_counts()
        .reindex([0, 1], fill_value=0)
    )

    print(counts)

    fraud_percentage = df["is_fraud"].mean() * 100

    print(f"\nFraud percentage: {fraud_percentage:.2f}%")

    # 4. TRAINING-PERIOD EDA

    # Use only the first 70% for exploring fraud patterns.
    # The remaining records will be reserved for later
    # model validation and testing.

    train_end = int(len(df) * 0.70)

    train_df = df.iloc[:train_end].copy()

    print("\n========== TRAINING-PERIOD STATISTICS ==========")

    numerical_columns = [
        "amount_usd",
        "sender_balance_before_usd",
        "prior_txn_count_1h",
        "account_age_days",
        "geo_velocity_kmh",
    ]

    print(
        train_df.groupby("is_fraud")[
            numerical_columns
        ].median().round(2)
    )

    print("\n========== BEHAVIORAL SIGNALS ==========")

    signals = [
        "is_new_receiver",
        "is_new_device",
        "is_vpn_or_proxy",
        "country_mismatch",
    ]

    signal_rates = (
        train_df.groupby("is_fraud")[signals]
        .mean()
        .multiply(100)
        .round(2)
    )

    print("\nPercentage of transactions with each signal:")
    print(signal_rates)

    # 5. CHART: FRAUD CLASS DISTRIBUTION

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar(
        ["Legitimate", "Fraud"],
        [counts.loc[0], counts.loc[1]],
        color=["#2a9d8f", "#e76f51"],
    )

    ax.set_title("Transaction Class Distribution")
    ax.set_ylabel("Number of Transactions")

    for index, value in enumerate(counts):
        ax.text(
            index,
            value,
            f"{value:,}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()

    fig.savefig(
        REPORT_DIR / "fraud_distribution.png",
        dpi=150,
    )

    plt.close(fig)

    # 6. CHART: TRANSACTION AMOUNT DISTRIBUTION

    fig, ax = plt.subplots(figsize=(8, 5))

    for fraud_label, label, color in [
        (0, "Legitimate", "#2a9d8f"),
        (1, "Fraud", "#e76f51"),
    ]:

        amounts = train_df.loc[
            train_df["is_fraud"] == fraud_label,
            "amount_usd",
        ]

        ax.hist(
            np.log10(amounts + 1),
            bins=40,
            alpha=0.55,
            density=True,
            label=label,
            color=color,
        )

    ax.set_title("Transaction Amount by Fraud Class")
    ax.set_xlabel("log10(Transaction Amount + 1)")
    ax.set_ylabel("Density")
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        REPORT_DIR / "transaction_amount.png",
        dpi=150,
    )

    plt.close(fig)

    # 7. CHART: FRAUD RATE FOR BEHAVIORAL SIGNALS

    fraud_rates = {}

    for signal in signals:

        flagged_transactions = train_df.loc[
            train_df[signal] == 1
        ]

        if len(flagged_transactions) > 0:

            fraud_rates[signal] = (
                flagged_transactions["is_fraud"].mean()
                * 100
            )

        else:
            fraud_rates[signal] = 0.0

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.barh(
        list(fraud_rates.keys()),
        list(fraud_rates.values()),
        color="#457b9d",
    )

    ax.set_title(
        "Fraud Rate When Behavioral Signal Is Present"
    )

    ax.set_xlabel("Fraud Rate (%)")

    fig.tight_layout()

    fig.savefig(
        REPORT_DIR / "behavioral_signals.png",
        dpi=150,
    )

    plt.close(fig)

    print("\n========== CHARTS SAVED ==========")

    print("reports/fraud_distribution.png")
    print("reports/transaction_amount.png")
    print("reports/behavioral_signals.png")

    print("\nEDA completed successfully!")


if __name__ == "__main__":
    main()