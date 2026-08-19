# ❤️ Heart Disease Risk Predictor

A machine learning project that predicts the risk of heart disease based on clinical and demographic features.  
Includes full EDA, model comparison, and a ready-to-use **Streamlit web application**.

### 🚀 Live Demo
**Try the app here →** [Heart Disease Risk Predictor](https://avinash01715-ml-practice-notebo-heartdiseasepredictorapp-syqiie.streamlit.app/)

---

## 📌 Problem Statement

Heart disease is one of the leading causes of death worldwide. Early risk assessment can help individuals take preventive measures.

This project builds a classification model that estimates whether a person is at **high risk** or **low risk** of heart disease using medical attributes such as age, blood pressure, cholesterol, chest pain type, ECG results, and exercise-induced angina.

**Disclaimer:** This is an educational project only. It is **not** a medical diagnosis tool and should never replace professional medical advice.

---

## 📁 Dataset

| Detail              | Value                                      |
|---------------------|--------------------------------------------|
| File                | `heart.csv`                                |
| Records             | 918                                        |
| Target              | `HeartDisease` (0 = No, 1 = Yes)           |
| Missing values      | Handled (0 values in Cholesterol & RestingBP replaced with mean) |

### Features

| Feature            | Description                                      |
|--------------------|--------------------------------------------------|
| Age                | Age of the patient (years)                       |
| Sex                | M / F                                            |
| ChestPainType      | ATA, NAP, TA, ASY                                |
| RestingBP          | Resting blood pressure (mm Hg)                   |
| Cholesterol        | Serum cholesterol (mg/dL)                        |
| FastingBS          | Fasting blood sugar > 120 mg/dL (0/1)            |
| RestingECG         | Normal, ST, LVH                                  |
| MaxHR              | Maximum heart rate achieved                      |
| ExerciseAngina     | Exercise-induced angina (Y/N)                    |
| Oldpeak            | ST depression induced by exercise                |
| ST_Slope           | Slope of the peak exercise ST segment            |
| HeartDisease       | Target variable (0 or 1)                         |

---

## 🛠️ Tech Stack

- **Python**
- **Pandas / NumPy** – data handling
- **Matplotlib / Seaborn** – visualization
- **Scikit-learn** – preprocessing & modeling
- **Joblib** – model serialization
- **Streamlit** – interactive web app

---

## 🔄 Project Workflow

1. **Data Loading & Inspection**  
   Loaded `heart.csv`, checked shape, data types, missing values, and basic statistics.

2. **Exploratory Data Analysis (EDA)**  
   - Target distribution  
   - Histograms for numeric features (Age, RestingBP, Cholesterol, MaxHR)  
   - Count plots for categorical features  
   - Box plots and violin plots vs HeartDisease  
   - Correlation heatmap  

3. **Data Cleaning**  
   - Replaced 0 values in `Cholesterol` and `RestingBP` with respective means  
   - No missing values remaining  

4. **Feature Engineering**  
   - One-hot encoding of categorical variables (`pd.get_dummies(drop_first=True)`)  
   - Converted all columns to integer type  

5. **Model Building**  
   - Train-test split (80/20) with stratification  
   - Feature scaling using `StandardScaler`  
   - Compared 5 algorithms:

| Model                  | Accuracy | F1 Score |
|------------------------|----------|----------|
| Logistic Regression    | 0.8750   | 0.8878   |
| **KNN**                | **0.8859** | **0.8986** |
| Naive Bayes            | 0.8696   | 0.8788   |
| Decision Tree          | 0.7554   | 0.7692   |
| SVM (RBF Kernel)       | 0.8641   | 0.8804   |

6. **Model Selection & Saving**  
   - Best model: **K-Nearest Neighbors (KNN)**  
   - Saved artifacts:  
     - `KNN_heart.pkl`  
     - `scaler.pkl`  
     - `columns.pkl`

7. **Deployment**  
   - Built a polished Streamlit web application (`app.py`) for real-time predictions.

---

## 📊 Results

- **Best Model:** K-Nearest Neighbors  
- **Test Accuracy:** ≈ **88.6%**  
- **F1 Score:** ≈ **0.90**

The model performs well as a baseline risk screening tool. However, it should be treated as a decision-support system, not a diagnostic tool.

---

## ⚠️ Limitations

- Relatively small dataset (918 samples).
- No hyperparameter tuning was performed (default KNN parameters used).
- Medical data can have complex non-linear interactions; more advanced models (XGBoost, Random Forest, Neural Nets) may improve performance further.
- Real-world clinical data often contains more features (family history, smoking status, medications, etc.) that are missing here.
- Predictions should always be validated by a qualified medical professional.

---

## 🚀 How to Run

### Live Demo (Recommended)
👉 **[Open the deployed app](https://avinash01715-ml-practice-notebo-heartdiseasepredictorapp-syqiie.streamlit.app/)**

### Run Locally

```bash
# Clone the repository
git clone https://github.com/Avinash01715/ML-practice-notebook.git
cd ML-practice-notebook

# Install dependencies (requirements.txt is at the repo root)
pip install -r requirements.txt

# Go into the project folder
cd "Heart Disease Predictor"

# Run the Streamlit app
streamlit run app.py
```

---

## 📂 Project Structure

```
ML-practice-notebook/
├── requirements.txt                    # Shared dependencies (repo root)
│
└── Heart Disease Predictor/
    ├── app.py                          # Streamlit web application
    ├── Heartdisease.ipynb              # Full analysis & modeling notebook
    ├── heart.csv                       # Dataset
    ├── KNN_heart.pkl                   # Trained KNN model
    ├── scaler.pkl                      # Fitted StandardScaler
    ├── columns.pkl                     # Expected feature columns
    └── readme.md                       # Project documentation
```

---

## 🔮 Future Improvements

- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV) for KNN and other models
- Try ensemble methods (Random Forest, XGBoost, LightGBM)
- Add SHAP / feature importance analysis for better interpretability
- Collect larger and more diverse clinical datasets
- Improve the deployed Streamlit app UI/UX and add more explainability features
- Add probability calibration and confidence intervals

---

## 📝 Conclusion

This project demonstrates a complete end-to-end machine learning pipeline for heart disease risk prediction — from exploratory analysis and preprocessing to model comparison and interactive deployment.

The KNN model achieves solid performance (~88.6% accuracy) and is integrated into a user-friendly Streamlit interface. While useful for educational purposes and initial risk screening, it is **not a substitute for professional medical evaluation**.

---

**Author:** Avinash Sharma
**Repository:** ML-practice-notebook
```
