# Customer Churn Predictor 📞📉

A Machine Learning project designed to analyze customer behavior and predict the likelihood of customer churn. The project processes user demographic and billing data to train a **Logistic Regression** classifier.

---

## 📁 Repository Structure

```directory
├── main.ipynb                 # Jupyter notebook containing EDA, preprocessing, and model training
├── customer_churn_data.csv    # Raw dataset containing customer demographics, billing, and churn records
└── scaler.pkl                 # Serialized StandardScaler object for feature normalization
```

---

## 📊 Machine Learning Pipeline

### 1. Data Preprocessing & Features (`main.ipynb`)
- **Handling Missing Values**: Fills missing values in `InternetService` with an empty string.
- **Categorical Encoding**:
  - `Churn` (Target): Encoded as `Yes` -> `1`, `No` -> `0`.
  - `Gender`: Encoded as `Male` -> `1`, `Female` -> `0`.
- **Feature Selection**:
  - `Age` (Numerical)
  - `Tenure` (Numerical)
  - `MonthlyCharges` (Numerical)
  - `Gender` (Encoded binary: 1 = Male, 0 = Female)

### 2. Standardization
- Features are scaled using `StandardScaler` to ensure uniform scale.
- The fitted scaler is exported as `scaler.pkl` to normalize any future prediction inputs.

### 3. Model Training & Evaluation
- **Algorithm**: A `LogisticRegression` model is trained on the scaled training features.
- **Split**: 80/20 train-test split (`random_state=42`).

---

## 🛠️ Local Setup & Run

### Prerequisites
Make sure you have Python 3.9+ and the required packages installed:
```bash
pip install pandas numpy scikit-learn matplotlib joblib jupyter
```

### Running the Notebook
1. Start the Jupyter server:
   ```bash
   jupyter notebook
   ```
2. Open and run `main.ipynb` to execute the EDA plots, preprocess the data, and train the model.
