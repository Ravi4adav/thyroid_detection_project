# Data Transformation will contains the code which will handle the transformation of categorical data to numerical and other conversions

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import warnings
from sklearn.preprocessing import StandardScaler
from src.exception import CustomException
from dataclasses import dataclass
import sys
import os
from src.utils import save_object, best_feature,load_object
from src.logger import logging

warnings.filterwarnings('ignore')


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")



class DataImputation:

    def duplicate_drop(self,data):
        self.data=data
        self.data=self.data.drop_duplicates()
        logging.info("Duplicate data dropped successfully")
        return self.data
    

    def num_impute(self,data):
        try:
            logging.info("Initiating imputation process of numeric data")
            self.num_features=[features for features in self.data.columns if self.data[features].dtypes!='object']
            self.data=data
            for features in self.num_features:
                if self.data[features].isnull().sum() == self.data[features].shape[0]:
                    self.data.drop([features],axis=1,inplace=True)
                    self.num_features.remove(features)
            self.knn_instance=KNNImputer()
            self.imp_val=self.knn_instance.fit_transform(self.data[self.num_features])

            self.data[self.num_features]=self.imp_val
            logging.info('Imputation of numeric features done successfully')
            return self.data
        except Exception as e:
            logging.info('Numeric features imputation failed')
            raise CustomException(e,sys)
    

    def cat_impute(self,data):
        try:
            logging.info("Initiating imputation process of categorical data")
            self.data=data
            self.cat_features=[features for features in self.data.columns if self.data[features].dtypes=='object']
            for features in self.cat_features:
                self.data[features]=np.where(self.data[features].isnull(),self.data[features].mode(),self.data[features])

            logging.info('Imputation of categorical features done successfully')
            return self.data
        except Exception as e:
            logging.info('Categorical features imputation failed')
            raise CustomException(e,sys)
    
    
    def init_data_imputation(self,data):
        try:
            logging.info("Initiating all imputation process")
            self.data=data
            self.data=self.duplicate_drop(self.data)
            self.data=self.num_impute(self.data)
            self.data=self.cat_impute(self.data)
            logging.info("All imputation process execute successfully")
            return self.data
        except Exception as e:
            raise CustomException(e,sys)
    
    

class Encoding:

    def target_feature_encoding(self,data, target_feature_name='disease'):
        try:
            logging.info("Encoding target features to binary digits")
            self.data=data
            self.data[target_feature_name]=np.where(self.data[target_feature_name]!='negative',1,0)
            logging.info("Encoding of target feature sucess")
            return self.data
        except Exception as e:
            logging.info("Target feature encoding failed")
            raise CustomException(e,sys)

    def cat_feature_encoding(self,data):
        self.data=data
        self.cat_features=['sex', 'referral_source']
        logging.info("Initiating categorical feature encoding")
        try:
            for features in self.cat_features:
                self.unique_val = {val: idx for idx,val in enumerate(np.sort(self.data[features].unique()), start=1)}
                self.data[features]=self.data[features].map(self.unique_val)
            logging.info("Categorical feature encoding performed successfully")
            return self.data
        except Exception as e:
            logging.info("Categorical feature encoding failed")
            raise CustomException(e,sys)

    def bool_feature_encoding(self,data):
        self.cat_features=['sex', 'referral_source']
        self.num_features=[features for features in self.data.columns if self.data[features].dtypes!='object']
        self.boolean_features=[features for features in self.data.columns if features not in self.cat_features+self.num_features]

        self.data=data
        try:
            logging.info("Initiating boolean feature encoding")
            for features in self.boolean_features:
                if features!='disease':
                    self.data[features]=np.where(self.data[features]==True,1,0)
            logging.info("Boolean feature encoding performed successfully")
            return self.data
        except Exception as e:
            logging.info("Boolean feature encoding failed")
            raise CustomException(e,sys)
    

    def init_data_encoding(self,data):
        try:
            logging.info("Initiating all encoding process steps")
            self.data=data
            self.data=self.target_feature_encoding(self.data)
            self.data=self.cat_feature_encoding(self.data)
            self.data=self.bool_feature_encoding(self.data)
            logging.info("All encoding process steps executes successfully")
            return self.data
        except Exception as e:
            raise CustomException(e,sys)




class InitiateDataTransformation:
    
    def preprocessed_data(self,data,target_feature='disease'):
        try:
            dt_imp=DataImputation()
            dt_encode=Encoding()
            logging.info("Initiating all imputation and encoding process executions")
            data=dt_imp.init_data_imputation(data)
            data=dt_encode.init_data_encoding(data)
            x=data.drop([target_feature],axis=1)
            y=data[target_feature]
            logging.info("All imputation and encoding process executes successfully")
            return x,y
        
        except Exception as e:
            logging.info("Failed to execute all imputation and encoding process")
            raise CustomException(e,sys)
    
        
    
    # Method for prediction pipeline specifically (Not for training pipeline)
    def get_transform_data_pipeling(self,data):
        try:
            logging.info("Scaling down prediction data")
            scale=load_object(file_path='./artifacts/scale.pkl')
            scaled_data=pd.DataFrame(scale.transform(data),columns=data.columns)
            # print(scaled_data)
            return scaled_data
        except Exception as e:
            try:
                logging.info("Saving training data standard scaling instance to pickle file")
                x=pd.read_csv('./artifacts/preprocessed_data.csv')
                scale=StandardScaler()
                scale.fit(x)
                save_object(file_path='./artifacts/scale.pkl',obj=scale.fit(x))
                scaled_data=pd.DataFrame(scale.transform(data),columns=data.columns)
                return scaled_data
            except Exception as e:
                raise CustomException(e,sys)
            

    def confirmation(self):
        print("""Is it first of model training?
              This may affect the prediction pipeline working.
              Note: Do not Choose 'Y' if model is already build""")
        response=str(input("Please Enter your response: "))
        if ((response=='Y') or (response=='Yes')) or ((response=='y') or (response=='yes')):
            return 1
        else:
            return 0

    
    def get_transformed_data(self,train_data,test_data):
        try:
            logging.info("Initiating preprocessing, feature selection and scale down process for training data")
            x,y=self.preprocessed_data(train_data)
            x_test,y_test=self.preprocessed_data(test_data)
            

            check=self.confirmation()
            print("Check:",check)
            if check==1:
                feature_names=best_feature(x,y)  # invoking feature selection function and storing all selected feature to variable
            else:
                feature_names=['age','fti','t3','tsh','tt4']
            
            x=x[feature_names]
            x_test=x_test[feature_names]
            x.to_csv('./artifacts/preprocessed_data.csv',index=False)
            scale=StandardScaler()
            scale.fit(x)
            save_object(file_path='./artifacts/scale.pkl',obj=scale.fit(x))
            x=pd.DataFrame(scale.transform(x),columns=x.columns)
            x_test=pd.DataFrame(scale.transform(x_test),columns=x_test.columns)
            
            return x,y,x_test,y_test
        
        except Exception as e:
            raise CustomException(e,sys)



# if __name__=="__main__":
#     di=InitiateDataTransformation()
#     train_data=pd.read_csv('./artifacts/train.csv')
#     test_data=pd.read_csv('./artifacts/test.csv')
#     x,y,x_test,y_test=di.get_transformed_data(train_data,test_data)
#     print(x,y)