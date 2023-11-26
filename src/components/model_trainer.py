from src.logger import logging
from src.exception import CustomException
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from dataclasses import dataclass
import sys
import os
import pandas as pd
from src.utils import evaluate_models, save_object, best_feature, load_object
from src.components.data_transformation import InitiateDataTransformation



@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,x_train,y_train,x_test,y_test):
        try:
            logging.info("Reading train and test data")

            x_train,y_train=x_train,y_train
            x_test,y_test=x_test,y_test


            models={
                'knn classifier': KNeighborsClassifier(),
                'naive bayes': GaussianNB(),
                'random forest': RandomForestClassifier(class_weight='balanced'),
                'gradient boost': GradientBoostingClassifier(),
                'xgb': XGBClassifier(),
                'neural_network': MLPClassifier(),
                'adb': AdaBoostClassifier(),
            }

            params={
                'knn classifier':{'n_neighbors':[4,5,6,7,8], 'weights':['distance','uniform'], 'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']},
                'naive bayes': {},
                'random forest':{'min_samples_split':[25,30,35,40],'criterion':['log_loss','gini','entropy'],
                                 'max_features':['sqrt','log2'],'max_depth':[12]},
                'gradient boost':{'learning_rate':[0.09,0.1,0.2], 'max_features': ['log2','sqrt']},
                'xgb': {'booster': ['gbtree', 'gblinear'], 'learning_rate':[0.09,0.1],
                        'colsample_bylevel':[0.1,0.2,0.3], 'scale_pos_weight':[1,2],
                        'gamma':[4,5,6], 'max_depth':[5,7,9]},
                'neural_network': {'power_t':[0.3],
                                   'learning_rate_init':[0.01,0.1],'learning_rate':['invscaling'],
                                   },
                'adb':{'learning_rate':[0.001,0.01,0.08], 'algorithm':['SAMME','SAMME.R']},
            }


            logging.info("Performing model training and hyper parameter tuning")
            model_report: dict = evaluate_models(X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test, models=models,param=params)

            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]

            if best_model_score<0.75:
                raise CustomException("No best model found")
            

            logging.info("Saving best model to pickle file")
            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=best_model)

            predicted=best_model.predict(x_test)
            score=classification_report(y_test,predicted)


            return score

        except Exception as e:
            raise CustomException(e,sys)


# if __name__=="__main__":
#     train_data=pd.read_csv('./artifacts/train.csv')
#     test_data=pd.read_csv('./artifacts/test.csv')
#     transformer_obj=InitiateDataTransformation()
#     x_train,y_train,x_test,y_test=transformer_obj.get_transformed_data(test_data,test_data)
#     print(x_train.head())


#     trainer_obj=ModelTrainer()
#     score=trainer_obj.initiate_model_trainer(x_train,y_train,x_test,y_test)
#     print(score)
    
    