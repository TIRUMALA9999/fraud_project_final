"""FraudGuard Streamlit frontend."""

import os

import pandas as pd
import requests
import streamlit as st


# ----------------------------------
# 1. CONFIGURATION
# ----------------------------------

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

st.set_page_config(
    page_title="FraudGuard",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ FraudGuard")
st.subheader("AI-Powered Payment Fraud Detection")

st.caption(
    "Submit a sample payment to receive a fraud-risk score "
    "and a simulated APPROVED or REJECTED decision."
)


# ----------------------------------
# 2. PAYMENT FORM
# ----------------------------------

st.header("New Transaction")

with st.form("transaction_form"):

    left, right = st.columns(2)

    with left:

        customer_id = st.text_input(
            "Customer ID",
            value="C00001",
        )

        transaction_type = st.selectbox(
            "Transaction Type",
            ["PAYMENT", "TRANSFER", "CASH_OUT"],
        )

        amount_usd = st.number_input(
            "Transaction Amount ($)",
            min_value=1.0,
            value=100.0,
        )

        sender_balance = st.number_input(
            "Available Balance ($)",
            min_value=0.0,
            value=5000.0,
        )

        account_age = st.number_input(
            "Account Age (Days)",
            min_value=0,
            value=365,
        )

        prior_transactions = st.number_input(
            "Transactions in Previous Hour",
            min_value=0,
            value=0,
        )

    with right:

        receiver_id = st.text_input(
            "Receiver ID",
            value="R10001",
        )

        geo_velocity = st.number_input(
            "Geographic Velocity (km/h)",
            min_value=0.0,
            value=0.0,
        )

        new_receiver = st.checkbox(
            "New Receiver"
        )

        new_device = st.checkbox(
            "New Device"
        )

        vpn_proxy = st.checkbox(
            "VPN / Proxy Detected"
        )

        country_mismatch = st.checkbox(
            "Country Mismatch"
        )

    submitted = st.form_submit_button(
        "Check Transaction",
        type="primary",
    )


# ----------------------------------
# 3. SEND TRANSACTION TO FASTAPI
# ----------------------------------

if submitted:

    transaction = {
        "customer_id": customer_id,
        "receiver_id": receiver_id,
        "transaction_type": transaction_type,
        "amount_usd": amount_usd,
        "sender_balance_before_usd": sender_balance,
        "prior_txn_count_1h": prior_transactions,
        "is_new_receiver": int(new_receiver),
        "account_age_days": account_age,
        "is_new_device": int(new_device),
        "is_vpn_or_proxy": int(vpn_proxy),
        "country_mismatch": int(country_mismatch),
        "geo_velocity_kmh": geo_velocity,
    }

    try:

        response = requests.post(
            f"{API_URL}/predict",
            json=transaction,
            timeout=15,
        )

        response.raise_for_status()

        result = response.json()

        st.divider()
        st.header("Fraud Detection Result")

        if result["decision"] == "REJECTED":

            st.error(
                "🚨 TRANSACTION REJECTED — High Fraud Risk"
            )

        else:

            st.success(
                "✅ TRANSACTION APPROVED"
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Fraud Risk Score",
                f"{result['fraud_score']:.4f}",
            )

        with col2:
            st.metric(
                "Decision Threshold",
                f"{result['threshold']:.4f}",
            )

        with col3:
            st.metric(
                "Transaction ID",
                result["transaction_id"],
            )

    except requests.exceptions.RequestException as error:

        st.error(
            f"Unable to connect to the backend: {error}"
        )


# -------------------------------------------------------
# TRANSACTION HISTORY
# -------------------------------------------------------

st.divider()
st.header("📋 Transaction History")

if st.button("🔄 Refresh Transactions"):
    st.rerun()

try:
    response = requests.get(
        f"{API_URL}/transactions",
        timeout=15
    )
    response.raise_for_status()

    transactions = response.json()

except requests.exceptions.RequestException as error:
    st.error(f"Unable to load transactions: {error}")
    transactions = []


if transactions:

    # Display transaction counts
    total_transactions = len(transactions)

    approved_count = sum(
        tx["decision"] == "APPROVED"
        for tx in transactions
    )

    rejected_count = sum(
        tx["decision"] == "REJECTED"
        for tx in transactions
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Recent Transactions", total_transactions)
    col2.metric("Approved", approved_count)
    col3.metric("Rejected", rejected_count)

    # Display transactions in a table
    transaction_df = pd.DataFrame(transactions)

    st.dataframe(
        transaction_df,
        use_container_width=True,
        hide_index=True
    )

    # -------------------------------------------------------
    # SHAP EXPLAINABILITY
    # -------------------------------------------------------

    st.divider()
    st.header("🔍 Transaction Explainability")

    st.write(
        "Select a transaction to understand "
        "which features influenced its fraud prediction."
    )

    # Create transaction selection options
    transaction_options = {
        f"Transaction #{tx['id']} | "
        f"{tx['customer_id']} | "
        f"{tx['decision']}": tx["id"]
        for tx in transactions
    }

    selected_transaction = st.selectbox(
        "Select Transaction",
        options=list(transaction_options.keys())
    )

    if selected_transaction:

        transaction_id = transaction_options[
            selected_transaction
        ]

        try:
            with st.spinner("Calculating SHAP explanation..."):

                explanation_response = requests.get(
                    f"{API_URL}/transactions/"
                    f"{transaction_id}/explanation",
                    timeout=60
                )

                explanation_response.raise_for_status()
                explanation = explanation_response.json()

            # Display prediction information
            st.subheader(
                f"Explanation for Transaction #{transaction_id}"
            )

            col1, col2 = st.columns(2)

            fraud_score = explanation["fraud_score"]
            decision = explanation["decision"]

            col1.metric(
                "Fraud Risk Score",
                f"{fraud_score * 100:.2f}%"
            )

            col2.metric(
                "Decision",
                decision
            )

            if decision == "REJECTED":
                st.error("🚨 High Fraud Risk")
            else:
                st.success("✅ Transaction Approved")

            # SHAP contributions
            shap_df = pd.DataFrame(
                explanation["contributions"]
            )

            st.subheader("SHAP Feature Contributions")

            st.bar_chart(
                shap_df,
                x="feature",
                y="shap_value",
                horizontal=True,
                sort=False
            )

            st.write(
                "🔴 Positive SHAP values increase fraud risk."
            )

            st.write(
                "🟢 Negative SHAP values decrease fraud risk."
            )

            # Display exact values
            st.subheader("Detailed SHAP Values")

            st.dataframe(
                shap_df,
                use_container_width=True,
                hide_index=True
            )

        except requests.exceptions.RequestException as error:
            st.error(
                f"Unable to calculate SHAP explanation: {error}"
            )

else:
    st.info("No transactions available yet.")