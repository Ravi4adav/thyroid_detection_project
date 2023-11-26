from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import InitiateDataTransformation
from src.components.model_trainer import ModelTrainer
import pandas as pd
import sys
from src.logger import logging
from src.exception import CustomException


class train_pipeline:
    data_ing_obj=DataIngestion()
    data_transf_obj=InitiateDataTransformation()
    model_train_obj=ModelTrainer()

    def __init__(self):
        logging.info("Reading data for training pipeline")
        print("1/3 Data_ingestion")
        self.data_ing_obj.initiate_data_ingestion()
        self.train_data=pd.read_csv('./artifacts/train.csv')
        self.test_data=pd.read_csv('./artifacts/test.csv')


    def model_training(self):
        try:
            print("2/3 Data Transformation")
            x_train,y_train,x_test,y_test=self.data_transf_obj.get_transformed_data(self.train_data,self.test_data)
            print(x_train.head())
            logging.info("Initiating model training process")
            print("3/3 Model Training")
            model_score=self.model_train_obj.initiate_model_trainer(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test)
            logging.info("Model trained successfully")
            print('Model Training Completed Successfully')
            print(f"Best Model Accuracy score: {model_score}")
        except Exception as e:
            raise CustomException(e,sys)


# if __name__=="__main__":
#     train_pipe_obj=train_pipeline()
#     train_pipe_obj.model_training()