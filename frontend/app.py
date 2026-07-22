

import streamlit as st
import pandas as pd
import requests

# ---------------------------------------------------
# Backend URL
# ---------------------------------------------------

BACKEND_URL = "http://backend:7860"

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="SuperKart Sales Forecasting",
    page_icon="🛒",
    layout="wide"
)

# ---------------------------------------------------
# Header
# ---------------------------------------------------

st.title("🛒 SuperKart Sales Forecasting Dashboard")

st.write(
    "Predict the sales revenue of a product at a SuperKart outlet "
    "using the trained Random Forest Regression model."
)

# =====================================================
# ONLINE PREDICTION
# =====================================================

st.header("📈 Online Prediction")

left, right = st.columns(2)

# ---------------- Product Details ----------------

with left:

    st.subheader("Product Details")

    product_id = st.text_input(
        "Product ID",
        value="FDX07"
    )

    product_weight = st.number_input(
        "Product Weight",
        min_value=0.0,
        value=19.2,
        step=0.1
    )

    product_sugar_content = st.selectbox(
        "Product Sugar Content",
        [
            "Low Sugar",
            "Regular",
            "No Sugar"
        ]
    )

    product_allocated_area = st.number_input(
        "Allocated Display Area",
        min_value=0.0,
        value=0.075,
        step=0.001,
        format="%.3f"
    )

    product_type = st.selectbox(
        "Product Type",
        [
            "Baking Goods",
            "Breads",
            "Breakfast",
            "Canned",
            "Dairy",
            "Frozen Foods",
            "Fruits and Vegetables",
            "Hard Drinks",
            "Health and Hygiene",
            "Household",
            "Meat",
            "Others",
            "Seafood",
            "Snack Foods",
            "Soft Drinks",
            "Starchy Foods"
        ]
    )

    product_mrp = st.number_input(
        "Product MRP",
        min_value=0.0,
        value=249.8,
        step=0.5
    )

# ---------------- Store Details ----------------

with right:

    st.subheader("Store Details")

    store_id = st.text_input(
        "Store ID",
        value="OUT049"
    )

    store_size = st.selectbox(
        "Store Size",
        [
            "Low",
            "Medium",
            "High"
        ]
    )

    store_city = st.selectbox(
        "Store Location City Type",
        [
            "Tier 1",
            "Tier 2",
            "Tier 3"
        ]
    )

    store_type = st.selectbox(
        "Store Type",
        [
            "Departmental Store",
            "Food Mart",
            "Supermarket Type1",
            "Supermarket Type2"
        ]
    )

    store_age = st.number_input(
        "Store Age",
        min_value=1,
        value=17
    )

# ---------------------------------------------------
# Prepare Input
# ---------------------------------------------------

input_data = {
    "Product_Id": product_id,
    "Product_Weight": product_weight,
    "Product_Sugar_Content": product_sugar_content,
    "Product_Allocated_Area": product_allocated_area,
    "Product_Type": product_type,
    "Product_MRP": product_mrp,
    "Store_Id": store_id,
    "Store_Size": store_size,
    "Store_Location_City_Type": store_city,
    "Store_Type": store_type,
    "Store_Age": store_age
}

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

if st.button("Predict Sales", type="primary"):

    try:

        response = requests.post(
            f"{BACKEND_URL}/predict",
            json=input_data
        )

        if response.status_code == 200:

            prediction = response.json()[
                "Predicted_Product_Store_Sales_Total"
            ]

            st.success("Prediction Successful")

            st.metric(
                "Predicted Sales",
                f"₹ {prediction:,.2f}"
            )

        else:

            st.error(response.json()["error"])

    except Exception as e:

        st.error(str(e))

# =====================================================
# BATCH PREDICTION
# =====================================================

st.header("📂 Batch Prediction")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Preview")

    st.dataframe(df.head())

    if st.button("Predict Batch", type="primary"):

        uploaded_file.seek(0)

        try:

            response = requests.post(
                f"{BACKEND_URL}/predict_batch",
                files={"file": uploaded_file}
            )

            if response.status_code == 200:

                prediction_df = pd.DataFrame(
                    response.json()
                )

                st.success("Batch Prediction Completed")

                st.dataframe(prediction_df)

                csv = prediction_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "Download Predictions",
                    csv,
                    "sales_predictions.csv",
                    "text/csv"
                )

            else:

                st.error(response.json()["error"])

        except Exception as e:

            st.error(str(e))

# ---------------------------------------------------
# Footer
# ---------------------------------------------------

st.markdown("---")

st.caption(
    "Developed using Streamlit, Flask, Docker and Random Forest Regression."
)
