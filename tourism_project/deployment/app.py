import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="arss25/tourism_mlops_model", filename="best_toursim_predict_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Wellness Tourism Package Prediction
st.title("Wellness Tourism Package Purchase Prediction")
st.write("""
This application predicts whether a customer will purchase the newly introduced Wellness Tourism Package.
Please enter the customer details and interaction data below to get a prediction.
""")

# User input for Customer Details
st.subheader("Customer Details")
Age = st.number_input("Age", min_value=18, max_value=100, value=30)
TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
Occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Small Business", "Large Business", "Government Sector"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=1)
PreferredPropertyStar = st.selectbox("Preferred Property Star Rating", [1.0, 2.0, 3.0, 4.0, 5.0])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
NumberOfTrips = st.number_input("Number of Trips Annually", min_value=0, max_value=50, value=1)
Passport = st.selectbox("Passport", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
OwnCar = st.selectbox("Own Car", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting (below 5 years)", min_value=0, max_value=5, value=0)
Designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "VP", "CEO"])
MonthlyIncome = st.number_input("Monthly Income", min_value=0.0, max_value=1000000.0, value=25000.0, step=1000.0)

# User input for Customer Interaction Data
st.subheader("Customer Interaction Data")
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score (1-5)", min_value=1, max_value=5, value=3)
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=2)
DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=1.0, max_value=60.0, value=10.0, step=1.0)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome,
    'ProductPitched': ProductPitched,
    'NumberOfFollowups': NumberOfFollowups
}])


if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Yes, the customer is likely to purchase the package!" if prediction == 1 else "No, the customer is not likely to purchase the package."
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
