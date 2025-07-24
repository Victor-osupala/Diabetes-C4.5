import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)

# App Configuration
st.set_page_config(page_title="Diabetes Prediction System", layout="centered")

# === Load Data & Model === #
@st.cache_data
def load_data():
    return pd.read_csv("diabetes.csv")

@st.cache_resource
def load_model():
    model = joblib.load("diabetes_model_output/diabetes_model.joblib")
    selector = joblib.load("diabetes_model_output/feature_selector.joblib")
    return model, selector

df = load_data()
model, selector = load_model()

X = df.drop(columns=["target"])
y = df["target"]
selected_features = X.columns[selector.get_support()]
X_selected = df[selected_features]

# === Sidebar Navigation === #
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Predict", "📈 Evaluate"])


from PIL import Image

# Load and resize the image
img_path = "assets/Diabetes.jpeg"
image = Image.open(img_path)
resized_image = image.resize((600, 600))  # width, height in pixels


## === HOME === #
if page == "🏠 Home":
    st.title("🩺 Diabetes Prediction System")
    st.markdown("---")

    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        # Display banner image in first column
        st.image(resized_image, caption="AI-powered Diabetes Prediction", use_container_width=False)

    with col2:
        # Welcome text in second column
        st.markdown("""
        Welcome to the **Diabetes Prediction System**.  
        This app uses a **pretrained C4.5 Decision Tree model** with **Correlation-Based Feature Selection** to:
        
        - Predict whether a patient is diabetic based on medical input features  
        - Evaluate model performance using real data

        Navigate using the sidebar to get started.
        """)

    st.markdown("---")
    st.subheader("🔬 What is Diabetes?")
    st.markdown("""
    Diabetes is a **chronic medical condition** in which the body cannot properly regulate blood sugar (glucose) levels.  
    This can be due to either insufficient insulin production or the body not responding properly to insulin.

    If left unmanaged, diabetes can lead to serious complications such as:
    - Heart disease
    - Kidney failure
    - Vision loss
    - Nerve damage

    **Early prediction and monitoring** are critical to reduce long-term risks.
    """)

    st.markdown("---")
    st.subheader("🧠 About the Prediction Algorithm")

    # Create two columns for algorithm explanations
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("""
        ### 📌 C4.5 Decision Tree (Entropy-Based)
        This app uses a **C4.5 Decision Tree**, a well-known supervised learning algorithm that:
        - Splits data based on **entropy** and **information gain ratio**
        - Handles both **numerical and categorical features**
        - Is **interpretable and easy to visualize**, making it ideal for medical decision systems

        The model was trained using patient data and tuned to classify individuals as **Diabetic** or **Non-Diabetic**.
        """)

    with col4:
        st.markdown("""
        ### 📌 Correlation-Based Feature Selection (CFS)
        To improve accuracy and reduce noise, we use **Correlation-Based Feature Selection (CFS)** which:
        - Retains features that are highly correlated with the target (diabetes status)
        - Removes features that are redundant or weakly predictive

        This helps the decision tree make better splits and avoids overfitting.
        """)

    st.markdown("---")
    st.info("Use the **📊 Predict** tab to try out predictions and the **📈 Evaluate** tab to view model performance.")


# === PREDICT === #
elif page == "📊 Predict":
    st.header("🧪 Make a Prediction")

    st.markdown("Enter the patient's data below:")

    input_data = {}
    for col in selected_features:
        input_data[col] = st.number_input(
            label=col,
            value=float(df[col].median()),
            min_value=0.0,
            step=0.1
        )

    if st.button("🔍 Predict"):
        input_df = pd.DataFrame([input_data])
        input_selected = input_df[selected_features]
        prediction = model.predict(input_selected)[0]
        probability = model.predict_proba(input_selected)[0][1]

        st.markdown("---")
        st.subheader("✅ Prediction Result")
        st.success("**Diabetic**" if prediction == 1 else "**Non-Diabetic**")
        st.info(f"**Probability of Diabetes:** {probability:.2%}")

# === EVALUATE === #
elif page == "📈 Evaluate":
    st.header("📊 Model Evaluation on Dataset")

    y_pred = model.predict(X_selected)
    y_prob = model.predict_proba(X_selected)[:, 1]

    acc = accuracy_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_prob)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True)

    st.subheader("📋 Evaluation Metrics")
    st.write(f"**Accuracy:** `{acc:.4f}`")
    st.write(f"**ROC AUC Score:** `{roc_auc:.4f}`")

    st.subheader("📌 Classification Report")
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("📉 Confusion Matrix")
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

    st.subheader("📈 ROC Curve")
    fpr, tpr, _ = roc_curve(y, y_prob)
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="green")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend()
    st.pyplot(fig_roc)
