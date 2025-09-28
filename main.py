import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_squared_error,mean_absolute_error,r2_score
from xgboost import XGBClassifier,XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMRegressor
import joblib
import warnings
import json
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")


def pre_processing(class_model = None , reg_model = None):

    df = pd.read_csv("Dataset.csv")
    # print(df.info())

    target_cat = ["alcohol_consumption","diabetes_risk","hypertension_risk","cholesterol_risk","liver_disease_risk","kidney_disease_risk"]
    target_num = ["estimated_medical_cost"]

    input_cat = ["gender"]
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

    # Target feature encoding
    le = LabelEncoder()
    df["alcohol_consumption"] = le.fit_transform(df["alcohol_consumption"])

    X = df[input_cat + input_ord + input_num] 
    y = df[target_cat + target_num]  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Classification Pipeline
    if class_model is not None:
        model = Pipeline(steps=[
        ('preprocessing', input_transformer),
        ('model', class_model)   
    ])
        model.fit(X_train, y_train[target_cat])
        y_pred = model.predict(X_test)

        # Saving Trained Classification model files with model name
        file_name = str(class_model).split('(')[0] # Ex:- ['RandomForestClassifier',')']
        file_path = f"Trained models/{file_name}.pkl"
        joblib.dump(model,file_path)

        # Classification Evaluate matrics
        accuracy = accuracy_score(y_test[target_cat],y_pred)
        #print(f"model= {class_model}")
        print(f"{file_name} accuracy : {accuracy}")

        return {"Model name": file_name, "Accuracy" : accuracy, "Saved Model File":file_path}

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

# classification models:
random_cls = pre_processing(class_model = RandomForestClassifier())
xgboost_cls = pre_processing(class_model = XGBClassifier())


# regression models:
# random_reg = pre_processing(reg_model= RandomForestRegressor())
# xgboost_reg = pre_processing(reg_model = XGBRegressor())
# lgbm_reg = pre_processing(reg_model = LGBMRegressor(force_col_wise=True))
# catboost_reg = pre_processing(reg_model = CatBoostRegressor())

# This is new update to the github