
import streamlit as st
import joblib
import os

# Configure the Streamlit page
st.set_page_config(
    page_title="Patient Risk Stratification",
    page_icon="🏥",
    layout="centered"
)

# Load the trained model
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "random_forest_model.pkl"
)

model = joblib.load(MODEL_PATH)

# Display the application

with st.sidebar:
    st.title("About the Project")

    st.write(
        "Patient Risk Stratification is a machine learning "
        "course project using the CDC Diabetes Health Indicators dataset."
    )

    st.divider()

    st.subheader("Model Information")
    st.write("**Algorithm:** Random Forest Classifier")
    st.write("**Input features:** 21")
    st.write("**Task:** Binary classification")

    st.divider()

    st.subheader("Understanding the Score")
    st.write(
        "The score is the model-estimated probability of class 1 "
        "in the source dataset."
    )

    st.write(
        "- **Lower:** below 20%\n"
        "- **Moderate:** 20% to below 50%\n"
        "- **Higher:** 50% or above"
    )

    st.divider()

    st.subheader("Limitations")
    st.write(
        "This is an educational prototype. Its categories are "
        "experimental, not medically validated. It does not diagnose "
        "diabetes or predict future disease onset."
    )


st.title("Patient Risk Stratification")
st.write(
    "An educational machine learning prototype "
    "for stratifying diabetes-status risk."
)

st.success("Random Forest model loaded successfully!")

st.info(
    "This prototype is for educational purposes only. "
    "It is not a medical diagnosis or a prediction of future diabetes onset."
)

st.header("Patient Information")

with st.expander("Medical History", expanded=True):

    high_bp = st.selectbox("High Blood Pressure", options=[0,1], format_func=lambda x: "Yes" if x==1 else "No")

    high_chol = st.selectbox("High Cholesterol", options=[0,1],format_func=lambda x : "Yes" if x==1 else "No")

    chol_check = st.selectbox(
        "Cholesterol check in the last 5 years?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    bmi = st.number_input("BMI (Body Mass Index)", min_value=12.0, max_value=98.0, value=25.0, step=0.1)

    smoker = st.selectbox(
        "Have you smoked at least 100 cigarettes in your lifetime?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    stroke = st.selectbox(
        "History of stroke?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    heart_disease = st.selectbox(
        "History of coronary heart disease or heart attack?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )


with st.expander("Lifestyle and Healthcare", expanded=True):

    phys_activity = st.selectbox(
        "Physical activity outside work in the past 30 days?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    fruits = st.selectbox("Consume fruit one or more times per day?", options=[0,1], format_func=lambda x : "Yes" if x==1 else "No")

    veggies = st.selectbox(
        "Consume vegetables one or more times per day?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    heavy_alcohol = st.selectbox(
        "Heavy alcohol consumption?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    any_healthcare = st.selectbox(
        "Have any kind of healthcare coverage?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    no_doc_cost = st.selectbox(
        "Could not see a doctor because of cost in the past 12 months?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    gen_health = st.select_slider(
        "General health",
        options=[1, 2, 3, 4, 5],
        value=3,
        format_func=lambda x: {
            1: "Excellent",
            2: "Very good",
            3: "Good",
            4: "Fair",
            5: "Poor"
        }[x]
    )


with st.expander("Health Status and Demographics", expanded=True):
    ment_health = st.slider(
        "Number of days mental health was not good (past 30 days)",
        min_value=0,
        max_value=30,
        value=0
    )

    phys_health = st.slider(
        "Number of days physical health was not good (past 30 days)",
        min_value=0,
        max_value=30,
        value=0
    )

    diff_walk = st.selectbox(
        "Do you have serious difficulty in walking or climbing stairs?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    sex = st.selectbox(
        "Sex",
        options=[0, 1],
        format_func=lambda x: {
            0: "Female",
            1: "Male"
        }[x]
    )

    age = st.select_slider(
        "Age category",
        options=list(range(1, 14)),
        value=7,
        format_func=lambda x: {
            1: "18–24",
            2: "25–29",
            3: "30–34",
            4: "35–39",
            5: "40–44",
            6: "45–49",
            7: "50–54",
            8: "55–59",
            9: "60–64",
            10: "65–69",
            11: "70–74",
            12: "75–79",
            13: "80+"
        }[x]
    )

    education = st.select_slider(
        "Education level",
        options=list(range(1, 7)),
        value=4,
        format_func=lambda x: {
            1: "Never attended / kindergarten only",
            2: "Grades 1–8",
            3: "Grades 9–11",
            4: "Grade 12 / high school graduate",
            5: "College 1–3 years",
            6: "College 4+ years"
        }[x]
    )

    income = st.select_slider(
        "Annual household income category",
        options=list(range(1, 9)),
        value=5,
        format_func=lambda x: {
            1: "Less than $10,000",
            2: "$10,000–$14,999",
            3: "$15,000–$19,999",
            4: "$20,000–$24,999",
            5: "$25,000–$34,999",
            6: "$35,000–$49,999",
            7: "$50,000–$74,999",
            8: "$75,000 or more"
        }[x]
    )


import pandas as pd

st.divider()

if st.button("Assess Patient Risk", type="primary", width="stretch"):

    # Arrange inputs in the exact order used during model training
    patient_data = pd.DataFrame([{
        "HighBP": high_bp,
        "HighChol": high_chol,
        "CholCheck": chol_check,
        "BMI": bmi,
        "Smoker": smoker,
        "Stroke": stroke,
        "HeartDiseaseorAttack": heart_disease,
        "PhysActivity": phys_activity,
        "Fruits": fruits,
        "Veggies": veggies,
        "HvyAlcoholConsump": heavy_alcohol,
        "AnyHealthcare": any_healthcare,
        "NoDocbcCost": no_doc_cost,
        "GenHlth": gen_health,
        "MentHlth": ment_health,
        "PhysHlth": phys_health,
        "DiffWalk": diff_walk,
        "Sex": sex,
        "Age": age,
        "Education": education,
        "Income": income
    }])
    # Verify that the input matches the expected feature order
    expected_features = [
        "HighBP", "HighChol", "CholCheck", "BMI", "Smoker",
        "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits",
        "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
        "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth",
        "DiffWalk", "Sex", "Age", "Education", "Income"
    ]

    patient_data = patient_data[expected_features]

    # Get predicted probability for class 1
    probabilities = model.predict_proba(patient_data)[0]
    positive_class_index = list(model.classes_).index(1)
    risk_score = probabilities[positive_class_index]

    # Assign an exploratory risk category
    if risk_score < 0.20:
        category = "Lower"
    elif risk_score < 0.50:
        category = "Moderate"
    else:
        category = "Higher"

    st.subheader("Assessment Result")

    st.metric("Model Score", f"{risk_score:.1%}")

    st.progress(float(risk_score))

    st.write("### Assessment Summary")
    st.write(f"**Input features evaluated:** {patient_data.shape[1]}")

    if category == "Lower":
        st.success(f"Exploratory category: {category}")
    elif category == "Moderate":
        st.warning(f"Exploratory category: {category}")
    else:
        st.error(f"Exploratory category: {category}")

    st.caption(
        "These categories use experimental thresholds and are not "
        "medically validated. The model estimates recorded diabetes "
        "status from survey indicators; it does not predict future "
        "diabetes onset or provide a medical diagnosis."
    )
