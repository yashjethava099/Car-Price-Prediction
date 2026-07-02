import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

# ----------------------------
# Load model and reference data
# ----------------------------
@st.cache_resource
def load_model():
    return pickle.load(open("model/LinearRegressionModel.pkl", "rb"))

@st.cache_data
def load_data():
    return pd.read_csv("dataset/Cleaned_Car_data.csv")

model = load_model()
car = load_data()

companies = sorted(car["company"].unique())
fuel_types = sorted(car["fuel_type"].unique())
year_list = sorted(car["year"].unique(), reverse=True)

# ----------------------------
# UI
# ----------------------------
st.title("🚗 Used Car Price Predictor")
st.write(
    "Estimate the resale price of a used car based on its model, "
    "manufacturing year, kilometers driven, and fuel type."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    company = st.selectbox("Select Company", companies)

    # filter car names belonging to the selected company
    names_for_company = sorted(car[car["company"] == company]["name"].unique())
    name = st.selectbox("Select Car Model", names_for_company)

    year = st.selectbox("Select Year of Purchase", year_list)

with col2:
    fuel_type = st.selectbox("Select Fuel Type", fuel_types)
    kms_driven = st.number_input(
        "Kilometers Driven", min_value=0, max_value=500000, value=30000, step=1000
    )

st.divider()

if st.button("Predict Price", type="primary", use_container_width=True):
    input_df = pd.DataFrame(
        [[name, company, year, kms_driven, fuel_type]],
        columns=["name", "company", "year", "kms_driven", "fuel_type"],
    )

    prediction = model.predict(input_df)[0]
    prediction = max(0, prediction)  # guard against negative predictions

    st.success(f"### Estimated Price: ₹ {prediction:,.0f}")

    with st.expander("See input details"):
        st.write(input_df)

st.divider()
st.caption(
    "Model: Linear Regression trained on a Quikr-style used car dataset "
    "(name, company, year, kms_driven, fuel_type → price). "
    "Predictions are estimates only."
)