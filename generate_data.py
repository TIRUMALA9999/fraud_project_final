"""Generate synthetic payment transactions with fraud patterns."""

from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from math import radians, sin, cos, asin, sqrt
from pathlib import Path

import numpy as np
import pandas as pd


# Dataset configuration
TOTAL_TRANSACTIONS = 100_000
FRAUD_TRANSACTIONS = 2_000
TOTAL_CUSTOMERS = 5_000
RANDOM_SEED = 42

OUTPUT_PATH = Path("data/transactions.csv")

START_DATE = datetime(
    2026, 1, 1, tzinfo=timezone.utc
)

# Sample geographic locations
CITIES = [
    (40.7128, -74.0060),   # New York
    (32.7767, -96.7970),   # Dallas
    (41.8781, -87.6298),   # Chicago
    (34.0522, -118.2437),  # Los Angeles
    (25.7617, -80.1918),   # Miami
    (47.6062, -122.3321),  # Seattle
]


def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance in kilometers between two coordinates."""

    earth_radius_km = 6371.0

    lat1, lon1, lat2, lon2 = map(
        radians,
        [lat1, lon1, lat2, lon2],
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1)
        * cos(lat2)
        * sin(dlon / 2) ** 2
    )

    return 2 * earth_radius_km * asin(sqrt(a))


def generate_dataset():

    rng = np.random.default_rng(RANDOM_SEED)

    # Select exactly 2,000 transactions as synthetic fraud cases.
    fraud_positions = set(
        rng.choice(
            TOTAL_TRANSACTIONS,
            size=FRAUD_TRANSACTIONS,
            replace=False,
        ).tolist()
    )

    customer_ids = [
        f"C{i:05d}"
        for i in range(1, TOTAL_CUSTOMERS + 1)
    ]

    # Initialize each customer's profile.
    customer_home = {}
    favorite_receiver = {}
    starting_account_age = {}
    customer_balances = {}

    for customer in customer_ids:

        customer_home[customer] = CITIES[
            int(rng.integers(len(CITIES)))
        ]

        favorite_receiver[customer] = (
            f"R{int(rng.integers(1000, 9999)):05d}"
        )

        starting_account_age[customer] = int(
            rng.integers(5, 2000)
        )

        customer_balances[customer] = float(
            rng.uniform(3000, 50000)
        )

    # Historical data used to calculate behavioral features.
    last_transaction = {}
    recent_transaction_times = defaultdict(deque)
    previous_receivers = defaultdict(set)

    # Helps us simulate customers making repeated transactions.
    recently_active_customers = deque(maxlen=5)

    rows = []

    for i in range(TOTAL_TRANSACTIONS):

        transaction_time = START_DATE + timedelta(
            minutes=1.3 * i
        )

        is_fraud = int(i in fraud_positions)

        # Different synthetic fraud scenarios.
        if is_fraud:
            fraud_pattern = str(
                rng.choice(
                    [
                        "impossible_travel",
                        "account_takeover",
                        "transaction_burst",
                        "subtle_fraud",
                    ],
                    p=[0.30, 0.30, 0.20, 0.20],
                )
            )
        else:
            fraud_pattern = "legitimate"

        # Some fraud cases target recently active customers.
        if (
            fraud_pattern in [
                "impossible_travel",
                "transaction_burst",
            ]
            and recently_active_customers
        ):
            customer = str(
                rng.choice(
                    list(recently_active_customers)
                )
            )
        else:
            customer = str(
                rng.choice(customer_ids)
            )

        # Retrieve customer balance.
        balance_before = customer_balances[customer]

        # Add simulated funds if an account is nearly depleted.
        if balance_before < 1000:
            balance_before += float(
                rng.uniform(5000, 15000)
            )
            customer_balances[customer] = balance_before

        # Generate transaction amount.
        if fraud_pattern == "subtle_fraud":
            amount = rng.lognormal(
                mean=np.log(350),
                sigma=0.9,
            )

        elif is_fraud:
            amount = rng.lognormal(
                mean=np.log(1800),
                sigma=1.0,
            )

        else:
            amount = rng.lognormal(
                mean=np.log(250),
                sigma=1.0,
            )

        amount = round(
            float(
                max(
                    1.0,
                    min(amount, balance_before * 0.85),
                )
            ),
            2,
        )

        # Transaction type.
        if is_fraud:
            transaction_type = str(
                rng.choice(
                    ["PAYMENT", "TRANSFER", "CASH_OUT"],
                    p=[0.15, 0.55, 0.30],
                )
            )
        else:
            transaction_type = str(
                rng.choice(
                    ["PAYMENT", "TRANSFER", "CASH_OUT"],
                    p=[0.65, 0.30, 0.05],
                )
            )

        # Choose recipient.
        use_favorite_receiver = (
            rng.random() < (0.20 if is_fraud else 0.65)
        )

        if use_favorite_receiver:
            receiver = favorite_receiver[customer]
        else:
            receiver = (
                f"R{int(rng.integers(10000, 99999)):05d}"
            )

        # Was the recipient previously used by this customer?
        is_new_receiver = int(
            receiver not in previous_receivers[customer]
        )

        # Calculate previous transaction count in the last hour.
        history = recent_transaction_times[customer]

        while (
            history
            and (
                transaction_time - history[0]
            ).total_seconds() > 3600
        ):
            history.popleft()

        prior_txn_count_1h = len(history)

        # Account age is based on the customer's history.
        account_age_days = (
            starting_account_age[customer]
            + (transaction_time - START_DATE).days
        )

        # Generate device and network signals.
        if fraud_pattern == "account_takeover":
            new_device_probability = 0.80
            vpn_probability = 0.70
            mismatch_probability = 0.65

        elif fraud_pattern == "impossible_travel":
            new_device_probability = 0.35
            vpn_probability = 0.55
            mismatch_probability = 0.60

        elif fraud_pattern == "transaction_burst":
            new_device_probability = 0.25
            vpn_probability = 0.20
            mismatch_probability = 0.15

        elif fraud_pattern == "subtle_fraud":
            new_device_probability = 0.08
            vpn_probability = 0.08
            mismatch_probability = 0.05

        else:
            new_device_probability = 0.04
            vpn_probability = 0.03
            mismatch_probability = 0.01

        is_new_device = int(
            rng.random() < new_device_probability
        )

        is_vpn_or_proxy = int(
            rng.random() < vpn_probability
        )

        country_mismatch = int(
            rng.random() < mismatch_probability
        )

        # Generate the transaction location.
        previous = last_transaction.get(customer)

        if (
            fraud_pattern == "impossible_travel"
            and previous is not None
        ):
            # Choose a location far from the last transaction.
            farthest_city = max(
                CITIES,
                key=lambda city: haversine(
                    previous[1],
                    previous[2],
                    city[0],
                    city[1],
                ),
            )

            latitude, longitude = farthest_city

        else:
            # Most customers transact near their previous location.
            if previous is not None:
                latitude = previous[1]
                longitude = previous[2]
            else:
                latitude, longitude = customer_home[customer]

            # Occasionally simulate legitimate travel or relocation.
            travel_probability = (
                0.09 if is_fraud else 0.03
            )

            if rng.random() < travel_probability:
                latitude, longitude = CITIES[
                    int(rng.integers(len(CITIES)))
                ]

        # Small location variation.
        latitude += float(rng.normal(0, 0.02))
        longitude += float(rng.normal(0, 0.02))

        # Calculate geographic velocity from past events only.
        if previous is None:
            geo_velocity_kmh = 0.0

        else:
            previous_time, previous_lat, previous_lon = previous

            distance_km = haversine(
                previous_lat,
                previous_lon,
                latitude,
                longitude,
            )

            hours_elapsed = max(
                (
                    transaction_time - previous_time
                ).total_seconds() / 3600,
                1 / 60,
            )

            geo_velocity_kmh = round(
                min(distance_km / hours_elapsed, 10000),
                2,
            )

        rows.append({
            "transaction_id": f"TXN{i + 1:06d}",
            "customer_id": customer,
            "receiver_id": receiver,
            "event_timestamp_utc": (
                transaction_time.isoformat().replace(
                    "+00:00", "Z"
                )
            ),
            "transaction_type": transaction_type,
            "amount_usd": amount,
            "sender_balance_before_usd": round(
                balance_before, 2
            ),
            "prior_txn_count_1h": prior_txn_count_1h,
            "is_new_receiver": is_new_receiver,
            "account_age_days": account_age_days,
            "is_new_device": is_new_device,
            "is_vpn_or_proxy": is_vpn_or_proxy,
            "country_mismatch": country_mismatch,
            "latitude": round(latitude, 5),
            "longitude": round(longitude, 5),
            "geo_velocity_kmh": geo_velocity_kmh,
            "is_fraud": is_fraud,
        })

        # Update customer history AFTER calculating the features.
        history.append(transaction_time)

        previous_receivers[customer].add(receiver)

        last_transaction[customer] = (
            transaction_time,
            latitude,
            longitude,
        )

        customer_balances[customer] = (
            balance_before - amount
        )

        recently_active_customers.append(customer)

    # Create final DataFrame.
    df = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(OUTPUT_PATH, index=False)

    print("\nDataset created successfully!")
    print("File:", OUTPUT_PATH)
    print("Shape:", df.shape)

    print("\nFraud distribution:")
    print(df["is_fraud"].value_counts())

    print("\nFirst five transactions:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    generate_dataset()