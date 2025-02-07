import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Solar Power Prediction", layout="wide")

st.markdown("<h1 style='text-align: center;'>🔆 Solar Power Prediction </h1>", unsafe_allow_html=True)
st.write("\n")  # Spacing

#Load CSV File
csv_file = "Dataset/solarpowergeneration.csv"

if not os.path.exists(csv_file):
    st.error(f"🚨 Error: The file `{csv_file}` was not found! Please ensure it is in the same directory.")
    st.stop()

@st.cache_data
def load_data():
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip().str.lower()  # Normalize column names
    return df

df = load_data()

# Layout: Dataset Preview & Detected Columns
st.write("\n")  # Spacing
col1, col2 = st.columns([1.5, 1]) 

with col1:
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

with col2:
    st.subheader(" Detected Columns")
    st.write(df.columns.tolist())

# Feature & Target Selection
feature_keywords = ["irradiance", "temperature", "humidity", "wind", "speed"]
target_keywords = ["power", "output", "generation"]

features = [col for col in df.columns if any(keyword in col for keyword in feature_keywords)]
target = next((col for col in df.columns if any(keyword in col for keyword in target_keywords)), None)

if target is None:
    target = df.columns[-1]  

X = df[features]
y = df[target]

#Handle Missing Values
X.fillna(X.mean(), inplace=True)
y.fillna(y.mean(), inplace=True)

#Train/Test Split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train Model 
model = LinearRegression()
model.fit(X_train, y_train)

# Model Evaluation 
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

#Display Model Metrics
st.write("\n")  
metric_col1, metric_col2 = st.columns(2)

with metric_col1:
    st.metric(label="📉 Mean Squared Error (MSE)", value=f"{mse:.2f}")

with metric_col2:
    st.metric(label="📈 R² Score", value=f"{r2:.2f}")

#Layout: Prediction Inputs & Graph
st.write("\n")  
col3, col4 = st.columns([1, 1.2])  

with col3:
    st.subheader("🌞 Make a Prediction")
    user_inputs = [st.number_input(f"Enter {feature.capitalize()}:", value=float(X[feature].mean())) for feature in features]

    if st.button("Predict Solar Power Output"):
        user_input_array = np.array([user_inputs])
        prediction = model.predict(user_input_array)
        st.session_state.prediction_value = prediction[0]  # Store prediction in session state

with col4:
    st.subheader("📈 Actual vs. Predicted Power Output")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(y_test, y_pred, alpha=0.5, color="blue", label="Predictions")
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red', label="Perfect Fit")
    ax.set_xlabel("Actual Power Output")
    ax.set_ylabel("Predicted Power Output")
    ax.set_title("Linear Regression Predictions")
    ax.legend()
    st.pyplot(fig)

# Display Predicted Output BELOW the Graph
st.write("\n")  
if "prediction_value" in st.session_state:
    st.markdown(f"<h3 style='text-align: center; color: green;'>🔋 Predicted Power Output: {st.session_state.prediction_value:.2f} kW</h3>", unsafe_allow_html=True)