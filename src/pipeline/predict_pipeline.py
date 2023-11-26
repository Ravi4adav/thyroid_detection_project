import numpy as np
import pandas as pd
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from src.components.data_transformation import InitiateDataTransformation


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor=InitiateDataTransformation()

            data=preprocessor.get_transform_data_pipeling(features)
            model=load_object(file_path='./artifacts/model.pkl')
            preds=pd.DataFrame(model.predict(data))
            pred_prob=pd.DataFrame(model.predict_proba(data))
            print(pred_prob)
            preds[0]=np.where(pred_prob[1]<0.38,0,1)
            result=preds.iloc[0,0]
            
            print("result: ",result)
            if result==0:
                return 'Thyroid disease status: Negative'
            if result==1:
                return 'Thyroid disease status: Positive'
            return result
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self,age:str,fti: str,t3: str,tsh: str,tt4: str):
        self.age=age
        self.fti=fti
        self.t3=t3
        self.tsh=tsh
        self.tt4=tt4

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict={
                'age':[self.age],
                'fti':[self.fti],
                't3':[self.t3],
                'tsh':[self.tsh],
                'tt4':[self.tt4]
            }

            dataframe=pd.DataFrame(custom_data_input_dict)
            for features in dataframe.columns:
                dataframe[features]=np.where(dataframe[features]=="",'0',dataframe[features])
                dataframe[features]=dataframe[features].str.strip()
                dataframe[features]=dataframe[features].astype('Float64')


            return dataframe
        except Exception as e:
            raise CustomException(e,sys)