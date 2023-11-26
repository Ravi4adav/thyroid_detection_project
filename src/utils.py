import os
import sys

import numpy as np 
import pandas as pd
import dill
from sklearn.metrics import f1_score, roc_curve,accuracy_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from sklearn.feature_selection import SelectKBest, mutual_info_classif


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(models)):
            model = models[list(models.keys())[i]]
            params=param[list(param.keys())[i]]

            gs = GridSearchCV(model,param_grid=params,cv=3,scoring='accuracy')

            gs.fit(X_train,y_train)


            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            # model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)
            
            y_test_pred = model.predict(X_test)

            train_model_score = f1_score(y_train, y_train_pred,average='macro')

            test_model_score = f1_score(y_test, y_test_pred,average='macro')


            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


# function for feature selection
def best_feature(df,target_data):
    best_feature=SelectKBest(mutual_info_classif,k=5)
    best_feature.fit(df,target_data)
    feature_names=df.columns[best_feature.get_support()]
    return feature_names


