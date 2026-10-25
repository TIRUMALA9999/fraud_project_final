"""FraudGuard FastAPI backend."""

import os

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from functools import lru_cache

import numpy as np
import shap
from fastapi import HTTPException

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    JSON,
    String,
    create_engine,
    select,
)

from sqlalchemy.orm import declarative_base, sessionmaker


# ----------------------------------
# 1. LOAD TRAINED MODEL
# ----------------------------------

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "best_model.joblib"
)

saved_model = joblib.load(MODEL_PATH)

model = saved_model["model"]
features = saved_model["features"]
threshold = float(saved_model["threshold"])

model_name = saved_model["model_name"]


# ----------------------------------
# 2. DATABASE CONNECTION
# ----------------------------------

# SQLite for local testing.
# Docker Compose will override this with PostgreSQL.
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./fraudguard.db",
)

connection_args = (
    {"check_same_thread": False}
    if DATABASE_URL.startswith("sqlite")
    else {}
)

engine = create_engine(
    DATABASE_URL,
    connect_args=connection_args,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()


# ----------------------------------
# 3. DATABASE TABLE
# ----------------------------------

class TransactionRecord(Base):

    __tablename__ = "transactions"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    customer_id = Column(String(50), nullable=False)
    receiver_id = Column(String(50), nullable=False)

    input_data = Column(JSON, nullable=False)

    fraud_score = Column(Float, nullable=False)

    decision = Column(String(20), nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


Base.metadata.create_all(bind=engine)


# ----------------------------------
# 4. INPUT VALIDATION
# ----------------------------------

class TransactionInput(BaseModel):

    customer_id: str
    receiver_id: str

    transaction_type: Literal[
        "PAYMENT",
        "TRANSFER",
        "CASH_OUT",
    ]

    amount_usd: float = Field(gt=0)

    sender_balance_before_usd: float = Field(ge=0)

    prior_txn_count_1h: int = Field(ge=0)

    is_new_receiver: int = Field(ge=0, le=1)

    account_age_days: int = Field(ge=0)

    is_new_device: int = Field(ge=0, le=1)

    is_vpn_or_proxy: int = Field(ge=0, le=1)

    country_mismatch: int = Field(ge=0, le=1)

    geo_velocity_kmh: float = Field(ge=0)


# ----------------------------------
# 5. FASTAPI APPLICATION
# ----------------------------------

app = FastAPI(
    title="FraudGuard",
    description="Local ML-powered payment fraud detection demo",
    version="1.0",
)


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": model_name,
    }


# ----------------------------------
# 6. PREDICT FRAUD
# ----------------------------------

@app.post("/predict")
def predict(transaction: TransactionInput):

    # Convert incoming transaction into a dictionary.
    transaction_data = transaction.model_dump()

    # Use the same features as model training.
    model_input = pd.DataFrame([
        {
            feature: transaction_data[feature]
            for feature in features
        }
    ])

    # Predict fraud probability.
    fraud_score = float(
        model.predict_proba(model_input)[0][1]
    )

    # Simulated payment decision.
    if fraud_score >= threshold:
        decision = "REJECTED"
    else:
        decision = "APPROVED"

    # Store transaction in database.
    with SessionLocal() as session:

        record = TransactionRecord(
            customer_id=transaction.customer_id,
            receiver_id=transaction.receiver_id,
            input_data=transaction_data,
            fraud_score=fraud_score,
            decision=decision,
        )

        session.add(record)
        session.commit()
        session.refresh(record)

        transaction_id = record.id

    return {
        "transaction_id": transaction_id,
        "fraud_score": round(fraud_score, 4),
        "threshold": round(threshold, 4),
        "decision": decision,
    }


# ----------------------------------
# 7. RETRIEVE RECENT TRANSACTIONS
# ----------------------------------

@app.get("/transactions")
def get_transactions():

    with SessionLocal() as session:

        records = session.scalars(
            select(TransactionRecord)
            .order_by(TransactionRecord.id.desc())
            .limit(20)
        ).all()

        return [
            {
                "id": record.id,
                "customer_id": record.customer_id,
                "receiver_id": record.receiver_id,
                "fraud_score": round(
                    record.fraud_score, 4
                ),
                "decision": record.decision,
                "created_at": record.created_at,
            }
            for record in records
        ]

@lru_cache(maxsize=1)
def get_shap_explainer():
    # Last step of our selected pipeline is Random Forest
    classifier = model.steps[-1][1]
    return shap.TreeExplainer(classifier)


@app.get("/transactions/{transaction_id}/explanation")
def explain_transaction(transaction_id: int):

    # 1. Retrieve the transaction from PostgreSQL
    with SessionLocal() as session:
        transaction = session.get(
            TransactionRecord,
            transaction_id
        )

        if transaction is None:
            raise HTTPException(
                status_code=404,
                detail="Transaction not found"
            )

        transaction_data = transaction.input_data

        # 2. Reconstruct the original model input
        input_df = pd.DataFrame(
            [{feature: transaction_data[feature] for feature in features}],
            columns=features
        )

        fraud_score = float(transaction.fraud_score)
        decision = transaction.decision

    # 3. Apply the same preprocessing used during training
    transformed = model[:-1].transform(input_df)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed = np.asarray(transformed)

    try:
        transformed_names = list(
            model[:-1].get_feature_names_out()
        )
    except Exception:
        transformed_names = [
            f"feature_{i + 1}"
            for i in range(transformed.shape[1])
        ]

    # 4. Calculate SHAP values for the fraud class
    explainer = get_shap_explainer()
    shap_result = explainer.shap_values(transformed)

    if isinstance(shap_result, list):
        shap_values = np.asarray(shap_result[1])[0]
    else:
        shap_array = np.asarray(shap_result)

        if shap_array.ndim == 3:
            shap_values = shap_array[0, :, 1]
        else:
            shap_values = shap_array[0]

    # 5. Combine one-hot encoded features back into
    #    the original business feature names
    feature_impacts = {feature: 0.0 for feature in features}

    for encoded_name, impact in zip(transformed_names, shap_values):
        clean_name = str(encoded_name).split("__")[-1]

        original_feature = next(
            (
                feature
                for feature in features
                if clean_name == feature
                or clean_name.startswith(feature + "_")
            ),
            clean_name
        )

        feature_impacts[original_feature] = (
            feature_impacts.get(original_feature, 0.0)
            + float(impact)
        )

    # 6. Sort features by absolute contribution
    contributions = [
        {
            "feature": feature,
            "shap_value": round(impact, 6),
            "direction": (
                "Increases Fraud Risk"
                if impact > 0
                else "Decreases Fraud Risk"
                if impact < 0
                else "No Impact"
            )
        }
        for feature, impact in feature_impacts.items()
    ]

    contributions.sort(
        key=lambda item: abs(item["shap_value"]),
        reverse=True
    )

    # 7. Return explanation to Streamlit
    return {
        "transaction_id": transaction_id,
        "fraud_score": fraud_score,
        "decision": decision,
        "contributions": contributions
    }