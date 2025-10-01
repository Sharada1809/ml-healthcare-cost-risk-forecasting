import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_squared_error,mean_absolute_error,r2_score
from xgboost import XGBClassifier,XGBRegressor
from catboost import  CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputClassifier
import joblib
import warnings
import json
from flask import Flask, flash, redirect, render_template, request, session, url_for
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")


app = Flask(__name__)
app.secret_key = 'supersecretkey123'

# Home page
@app.route("/")
def home():
    return render_template("home.html")

# login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username == 'admin' and password == 'admin':
            session['user_logged_in'] = True
            flash('Login successful!', 'success')
            return redirect(url_for('final_prediction'))
        else:
            flash('Invalid username or password', 'danger')
            return redirect(url_for('login'))

    return render_template('login.html')

# Logout route
@app.route('/logout')
def logout():
    session.pop('user_logged_in', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


def pre_processing(class_model = None , reg_model = None):

    df = pd.read_csv("Dataset.csv")
    # print(df.info())

    target_cat = ["diabetes_risk","hypertension_risk","cholesterol_risk","liver_disease_risk","kidney_disease_risk"]
    target_num = ["estimated_medical_cost"]

    input_cat = ["gender","alcohol_consumption"]
    input_ord = ["left_ear_hearing","right_ear_hearing","urine_protein","smoking_status"]
    input_num = df.drop(columns=input_cat + input_ord + target_cat + target_num).columns.tolist()
    ordinal_order = [
        [1, 2],                      # left_ear_hearing
        [1, 2],                      # right_ear_hearing
        [1, 2, 3, 4, 5, 6],          # urine_protein
        [1, 2, 3]                   # smoking_status
    ]

    input_transformer = ColumnTransformer(transformers=[
        ('encode',OneHotEncoder(handle_unknown='ignore'),input_cat),
        ('ordinal', OrdinalEncoder(categories= ordinal_order),input_ord),
        ('scale', StandardScaler(),input_num)
    ])

    X = df[input_cat + input_ord + input_num] 
    y = df[target_cat + target_num]  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Classification Pipeline
    if class_model is not None:
        model = Pipeline(steps=[
        ('preprocessing', input_transformer),
        ('model', MultiOutputClassifier(class_model))   
    ])
        model.fit(X_train, y_train[target_cat])
        y_pred = model.predict(X_test)

        # Saving Trained Classification model files with model name
        file_name = str(class_model).split('(')[0] # Ex:- ['RandomForestClassifier',')']
        file_path = f"Trained models/{file_name}.pkl"
        joblib.dump(model,file_path)

        # Classification Evaluate matrics
        individual_accuracies = []
        for i, target in enumerate(target_cat): 
            acc = accuracy_score(y_test[target], y_pred[:, i])
            individual_accuracies.append(acc)
        average_accuracy = np.mean(individual_accuracies)
        #print(f"model= {class_model}")
        print(f"{file_name} accuracy : {average_accuracy}")

        return {"Model name": file_name, "Accuracy" : average_accuracy, "Saved Model File":file_path}

    # Regression Pipeline
    if reg_model is not None:
        model = Pipeline(steps=[
        ('preprocessing', input_transformer),
        ('model', reg_model)   
    ])
        model.fit(X_train, y_train["estimated_medical_cost"])
        y_pred = model.predict(X_test)

        # Saving Trained Regereesion model files with model name
        file_name = reg_model.__class__.__name__   #str(reg_model).split('(')[0]
        file_path = f"Trained models/{file_name}.pkl"
        joblib.dump(model,file_path)

        # Regereesion Evaluate matrics
        mse = mean_squared_error(y_test["estimated_medical_cost"], y_pred)
        mae = mean_absolute_error(y_test["estimated_medical_cost"], y_pred)
        r2 = r2_score(y_test["estimated_medical_cost"], y_pred)

        #print(f"model = {reg_model}")
        print(f"{file_name} MSE : {mse:.2f}")
        print(f"{file_name} MAE : {mae:.2f}")
        print(f"{file_name} R² Score : {r2:.2f}")

        return {"Model name": file_name, "Accuracy" : r2, "Saved Model File":file_path}

def best_model():
    
    # classification models:
    random_cls = pre_processing(class_model = RandomForestClassifier())
    print("--------------------------------------------------------------------------------")
    xgboost_cls = pre_processing(class_model = XGBClassifier())
    print("--------------------------------------------------------------------------------")
    # regression models:
    random_reg = pre_processing(reg_model= RandomForestRegressor())
    print("--------------------------------------------------------------------------------")
    xgboost_reg = pre_processing(reg_model = XGBRegressor())
    print("--------------------------------------------------------------------------------")
    lgbm_reg = pre_processing(reg_model = LGBMRegressor(force_col_wise=True))
    print("--------------------------------------------------------------------------------")
    catboost_reg = pre_processing(reg_model = CatBoostRegressor())
    print("--------------------------------------------------------------------------------")

    class_models = [random_cls,xgboost_cls]
    reg_models = [random_reg,xgboost_reg,lgbm_reg,catboost_reg]

    # Creating Json file for Classification models
    class_path = "Trained models/class_result.json"
    with open(class_path, "w") as file:
        json.dump(class_models,file,indent=4)
    
    with open(class_path, "r") as f1:
        cls_model_results = json.load(f1)

    cls_best_model = max(cls_model_results, key=lambda x : x["Accuracy"])
    cls_mdel_type = cls_best_model["Model name"]
    cls_model_file = cls_best_model["Saved Model File"]
    print(f"The best model is : {cls_mdel_type}")
    best_model_for_classification  = joblib.load(cls_model_file) 

    # Creating Json file for Regression models
    reg_path = "Trained models/reg_result.json"
    with open(reg_path, "w") as file:
        json.dump(reg_models,file,indent=4)
   
    with open(reg_path, "r") as f2:
        reg_model_results = json.load(f2)
    
    reg_best_model = max(reg_model_results, key=lambda x : x["Accuracy"])
    reg_mdel_type = reg_best_model["Model name"]
    reg_model_file = reg_best_model["Saved Model File"]
    print(f"The best model is : {reg_mdel_type}")
    best_model_for_regression  = joblib.load(reg_model_file) 

    return {"Classification": best_model_for_classification, "Regression" : best_model_for_regression}


@app.route('/form', methods=['GET', 'POST'])
def final_prediction():
    if not session.get('user_logged_in'):
        flash('Please login first.', 'danger')
        return redirect(url_for('login'))

    class_output = None
    reg_output = None
    
    if request.method == "POST":
        
        input = {
            "gender": request.form.get("gender"),
            "age": request.form.get("age"),
            "height": request.form.get("height"),
            "weight": request.form.get("weight"),
            "waist_circumference": request.form.get("waist_circumference"),
            "alcohol_consumption": request.form.get("alcohol_consumption"),
            "smoking_status": request.form.get("smoking_status"),
            "left_ear_hearing": request.form.get("left_ear_hearing"),
            "right_ear_hearing": request.form.get("right_ear_hearing"),
            "left_eye_vision": request.form.get("left_eye_vision"),
            "right_eye_vision": request.form.get("right_eye_vision"),
            "urine_protein": request.form.get("urine_protein"),
            "hemoglobin": request.form.get("hemoglobin"),
            "serum_creatinine": request.form.get("serum_creatinine"),
            "ast_sgot": request.form.get("ast_sgot"),
            "alt_sgpt": request.form.get("alt_sgpt"),
            "gamma_gtp": request.form.get("gamma_gtp"),
            "blood_sugar": request.form.get("blood_sugar"),
            "total_cholesterol": request.form.get("total_cholesterol"),
            "hdl_cholesterol": request.form.get("hdl_cholesterol"),
            "ldl_cholesterol": request.form.get("ldl_cholesterol"),
            "triglycerides": request.form.get("triglycerides"),
            "bmi": request.form.get("bmi"),
            "waist_to_height_ratio": request.form.get("waist_to_height_ratio"),
            "systolic_bp": request.form.get("systolic_bp"),
            "diastolic_bp": request.form.get("diastolic_bp")
            }

        df = pd.DataFrame([input])
        model = best_model()
        class_model = model["Classification"]
        reg_model = model["Regression"]

        # Prediction for classification model
        class_pred = class_model.predict(df) #[[0,0,1,0,1]]

        target_cat = ["diabetes_risk", "hypertension_risk", "cholesterol_risk", "liver_disease_risk", "kidney_disease_risk"]
        # Convert prediction array to a dictionary
        class_output = dict(zip(target_cat, class_pred[0]))

        # Prediction for Regression model
        reg_pred = reg_model.predict(df)
        reg_output = float(reg_pred[0])

        return render_template('form.html', class_output=class_output,reg_output=reg_output)
    return render_template("form.html")
    
##


if __name__ == '__main__':
    app.run(debug=True)




        
    




