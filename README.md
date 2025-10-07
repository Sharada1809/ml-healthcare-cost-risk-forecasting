# Healthcare Risk & Medical Cost Prediction Web App

This project is a **Healthcare Risk & Cost Estimation** web application built using **Flask** and a combination of **classification and regression machine learning models**. Users can register, log in, and input personal health data to receive:

* Predictions for **5 disease risks** (e.g., diabetes, hypertension)
* An estimated **medical cost**

The application demonstrates a complete ML pipeline—from data preprocessing, model training, evaluation, and selection, to deployment in a user-friendly web interface.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Features](#features)
* [Dataset](#dataset)
* [Technologies](#technologies)
* [Installation](#installation)
* [Usage](#usage)
* [Model Training](#model-training)
* [Folder Structure](#folder-structure)
* [Notes & Future Improvements](#notes--future-improvements)
* [License](#license)

---

## Project Overview

Predictive analytics in healthcare can help patients and providers take proactive steps toward disease prevention and cost management. This project uses a health-related dataset to:

* Predict **five disease risk categories** (diabetes, hypertension, cholesterol, liver, and kidney) using classification models.
* Estimate **medical treatment costs** using regression models.

It uses multiple ML models (Random Forest, XGBoost, CatBoost, LightGBM), evaluates them, selects the best-performing ones, and deploys them through a web interface built with Flask.

---

## Features

* **Multi-output classification** for predicting disease risks
* **Regression model** for predicting estimated medical cost
* **Automated model training and evaluation** with best model selection
* **User authentication system** (registration, login, logout)
* **Flask-based web interface**
* **Interactive prediction form** to enter user health metrics
* **Data visualization** of model performance metrics
* **Trained models are saved and reused (no retraining needed on every run)**

---

## Dataset

The dataset used in this project includes:

* **Demographics:** Gender, age, height, weight
* **Medical metrics:** Blood pressure, blood sugar, BMI, cholesterol, protein in urine, etc.
* **Behavioral indicators:** Smoking status, alcohol consumption, vision, hearing
* **Target variables:**

  * Five binary disease risk categories:

    * `diabetes_risk`, `hypertension_risk`, `cholesterol_risk`, `liver_disease_risk`, `kidney_disease_risk`
  * One numeric output:

    * `estimated_medical_cost`

> Make sure your `Dataset.csv` file is placed in the root directory of the project.

---

## Technologies

* Python 3.x
* Flask
* pandas
* NumPy
* scikit-learn
* XGBoost
* LightGBM
* CatBoost
* joblib
* seaborn / matplotlib (for plots)
* MySQL (for storing user credentials)

---

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Sharada1809/ml-healthcare-cost-risk-forecasting.git
   cd ml-healthcare-cost-risk-forecasting
   ```

2. **Create a virtual environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up MySQL database**:

   * Create a database named `healthcare`
   * Create a table `user_details`:

     ```sql
     CREATE TABLE user_details (
         id INT AUTO_INCREMENT PRIMARY KEY,
         user_name VARCHAR(100),
         email_id VARCHAR(100),
         password VARCHAR(100)
     );
     ```

5. **Place the dataset**:

   * Add `Dataset.csv` to the project root.

---

## Usage

### Train Models (Optional Step)

If models are not already trained, uncomment and run the following line in `app.py`:

```python
# train_and_save()
```

This will:

* Train multiple classification and regression models
* Evaluate them
* Save the best ones in `Trained models/`
* Generate `class_result.json` and `reg_result.json`

### Run the Flask App

```bash
python app.py
```

Then, open your browser and visit:
`http://127.0.0.1:5000/`

---

### Login / Register

* Register a new user
* Log in to access the prediction form

---

### Predict Health Risks and Medical Costs

* Fill in health parameters on the form
* Submit to get:

  * Disease risk predictions (binary: 0 or 1 for each disease)
  * Estimated medical cost (numerical)

---

## Model Training

* **Input Features:**

  * Categorical: `gender`, `alcohol_consumption`
  * Ordinal: `smoking_status`, `urine_protein`, `left_ear_hearing`, `right_ear_hearing`
  * Numerical: Remaining features like `age`, `bmi`, `bp`, etc.

* **Output Features:**

  * Multi-label classification (`MultiOutputClassifier`)
  * Regression for medical cost estimation

* **Models used:**

  * **Classification:** Random Forest, XGBoost
  * **Regression:** Random Forest, XGBoost, LightGBM, CatBoost

* **Evaluation:**

  * Classification: Accuracy per target & overall
  * Regression: MSE, MAE, R² Score

* The **best model** is chosen based on accuracy (classification) or R² (regression)

---

## Folder Structure

```
healthcare-risk-prediction/
│
├── Trained models/               # Saved models and evaluation results
│   ├── RandomForestClassifier.pkl
│   ├── XGBClassifier.pkl
│   ├── CatBoostRegressor.pkl
│   ├── LGBMRegressor.pkl
│   ├── class_result.json
│   └── reg_result.json
│
├── templates/                    # HTML templates
│   ├── home.html
│   ├── login.html
│   ├── signup.html
│   ├── form.html
│   └── visualization.html
│
├── static/                       # Static files (CSS, JS)
│
├── Dataset.csv                   # Health dataset (user input features and labels)
│
├── app.py                        # Main Flask application and ML logic
├── requirements.txt              # Required Python packages
└── README.md                     # Project documentation
```

---

## Notes & Future Improvements

* Add detailed error messages on the prediction form
* Improve UI/UX with responsive design
* Add database logging for prediction records
* Include more advanced feature selection and hyperparameter tuning
* Add support for retraining via admin dashboard
* Add interactive data visualizations for user results

---
