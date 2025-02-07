# 🌞 Solar Power Prediction using Machine Learning

This project predicts **solar power generation** using **Linear Regression**. It processes solar energy data, performs **feature engineering**, and builds predictive models to estimate power output.

##  Features
- 📊 Data Visualization – Interactive graphs and charts to explore solar power data.
- 🔍 Data Preprocessing – Handles missing values, normalizes, and scales the dataset.
- 🤖 Machine Learning Model – Implements Linear Regression for solar power prediction.
- ⚡ User-Friendly Web App – Built using Streamlit for real-time predictions.
- 📈 Performance Analysis – Visual representation of predictions vs. actual values.
  
---

## 📂 Project Structure

```
📺 Solar-Power-Prediction
├── 📄 app.py                # Streamlit Web App
├── 📄 SolarPower.ipynb      # Jupyter Notebook for EDA & Model Training
├── 📄 solarpowergeneration.csv  # Dataset
├── 📄 requirements.txt      # Required dependencies
└── 📄 README.md             # Project Documentation
```

---

## 🚀 1. Installation & Setup

### 🔹 **Step 1: Clone the Repository**
```bash
git clone https://github.com/srividyakota03/SolarPowerGenerator.git
cd SolarPowerGenerator
```

### 🔹 **Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

### 🔹 **Step 3: Run Streamlit Web App**
```bash
streamlit run app.py
```

---

## 📊 2. Dataset Overview (`solarpowergeneration.csv`)

| **Column Name**  | **Description** |
|------------------|---------------|
| `irradiance`     | Solar irradiance (W/m²) |
| `temperature`    | Ambient temperature (°C) |
| `humidity`       | Humidity percentage (%) |
| `wind_speed`     | Wind speed (m/s) |
| `power_output`   | Generated solar power (kW) |

---

##  3. Model Training (Jupyter Notebook)
The **`SolarPower.ipynb`** file contains:
✅ **Data Loading & Preprocessing**  
✅ **Exploratory Data Analysis (EDA)**  
✅ **Feature Engineering**  
✅ **Linear Regression Model Training & Evaluation**  

---

## 🖥️ 4. Web App Overview (`app.py`)

### 📈 **Features of the Streamlit App**
- **📂 Dataset Overview** (Displayed in a table)
- **🧠 ML Model Prediction** (User can input values)
- **📊 Visualization of Predictions** (Actual vs Predicted graph)
- **💪 Predicted Power Output Below the Graph**  

---

## 📊 5. Example Usage

Once you run the **Streamlit app (`app.py`)**, you can input weather parameters:

| **Input**       | **Value** |
|----------------|----------|
| Solar Irradiance | `850 W/m²` |
| Temperature     | `30°C` |
| Humidity       | `45%` |
| Wind Speed     | `5 m/s` |

✅ **Predicted Solar Power Output: `4.25 kW`**

---

## 📊 6. Results & Evaluation

| **Metric** | **Value** |
|------------|----------|
| 📉 **Mean Squared Error (MSE)** | `0.21` |
| 📈 **R² Score** | `0.94` |

---


