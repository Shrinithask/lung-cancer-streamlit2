import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Lung Cancer EDA App", layout="wide")

st.title("🧬 Lung Cancer Risk - EDA & Single Entry App")
st.markdown("Upload your dataset or enter a single patient's health info to analyze or predict lung cancer risk.")

# Upload dataset
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("1️⃣ Dataset Overview")
    st.dataframe(df.head())

    st.subheader("2️⃣ Summary Statistics")
    st.write(df.describe(include='all'))

    st.subheader("3️⃣ Lung Cancer Distribution")
    st.bar_chart(df["LUNG_CANCER"].value_counts())

    st.subheader("4️⃣ Gender vs Lung Cancer")
    gender_dist = pd.crosstab(df["GENDER"], df["LUNG_CANCER"])
    st.bar_chart(gender_dist)

    st.subheader("5️⃣ Correlation Heatmap (for numeric columns)")
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.info("No numeric columns found for correlation.")

    st.subheader("6️⃣ Custom Column Selector")
    selected_col = st.selectbox("Select a column to view its value counts", df.columns)
    st.write(df[selected_col].value_counts())

# ========== Single Entry Section ========== #
st.header("🧾 Single Entry Form")

with st.form("single_entry_form"):
    col1, col2 = st.columns(2)

    gender = col1.selectbox("Gender", ["M", "F"])
    age = col2.slider("Age", 15, 90, 45)
    smoking = col1.selectbox("Smoking", [1, 0])
    yellow_fingers = col2.selectbox("Yellow Fingers", [1, 0])
    anxiety = col1.selectbox("Anxiety", [1, 0])
    peer_pressure = col2.selectbox("Peer Pressure", [1, 0])
    chronic_disease = col1.selectbox("Chronic Disease", [1, 0])
    fatigue = col2.selectbox("Fatigue", [1, 0])
    allergy = col1.selectbox("Allergy", [1, 0])
    wheezing = col2.selectbox("Wheezing", [1, 0])
    alcohol = col1.selectbox("Alcohol Consuming", [1, 0])
    coughing = col2.selectbox("Coughing", [1, 0])
    shortness = col1.selectbox("Shortness of Breath", [1, 0])
    swallowing = col2.selectbox("Swallowing Difficulty", [1, 0])
    chest_pain = col1.selectbox("Chest Pain", [1, 0])

    submitted = st.form_submit_button("Submit Entry")

if submitted:
    input_data = {
        "GENDER": gender,
        "AGE": age,
        "SMOKING": smoking,
        "YELLOW_FINGERS": yellow_fingers,
        "ANXIETY": anxiety,
        "PEER_PRESSURE": peer_pressure,
        "CHRONIC DISEASE": chronic_disease,
        "FATIGUE ": fatigue,
        "ALLERGY ": allergy,
        "WHEEZING": wheezing,
        "ALCOHOL_CONSUMING": alcohol,
        "COUGHING": coughing,
        "SHORTNESS OF BREATH": shortness,
        "SWALLOWING DIFFICULTY": swallowing,
        "CHEST PAIN": chest_pain
    }

    st.subheader("✅ Your Input Entry")
    st.json(input_data)

    input_df = pd.DataFrame([input_data])
    st.dataframe(input_df)

    st.info("You can now use this input for prediction with a trained model if added.")
