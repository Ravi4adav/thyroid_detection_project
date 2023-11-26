# Data Transformation will contains the code which will handle the transformation of categorical data to numerical and other conversions
from database.db_connect import cassandra_db
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
import sys


class validate_data:
    db=cassandra_db('thyroid_disease','raw_data')
    raw_data=db.read_table()


class validation:
    ppdata=validate_data()
    data=ppdata.raw_data[::]
    dataframe=pd.DataFrame(data,).iloc[:,1:]
    df_cols=dataframe.columns
        
    
    def replace_val_with_nan(self, val_to_replace, colList=df_cols, df=dataframe):
        try:
            logging.info(f"Replacing {val_to_replace} with Nan")
            self.cols=colList
            self.df=df
            self.val=val_to_replace

            for features in self.cols:
                self.df[features]=np.where(self.df[features]==self.val, np.nan, self.df[features])
            
            logging.info(f"{val_to_replace} replaced with Nan successfully")
            return self.df
        except Exception as e:
            logging.info(f"Failed to replace {val_to_replace} with Nan")
            raise CustomException(e,sys)

    def obj_to_num(self, NumcolList, df):
        try:
            self.df=df
            self.col=NumcolList
            logging.info("Initiating conversion of object datatype to numeric datatype features")
            for features in self.col:
                self.df[features]=pd.to_numeric(self.df[features],errors='coerce')
            
            logging.info("conversion of object datatype to numeric datatype done successfully")
            return self.df
        except Exception as e:
            logging.info("conversion of object datatype to numeric datatype failed")
            raise CustomException(e,sys)
    

    def obj_to_bool(self, df, BoolcolList):
        self.df=df
        self.col=BoolcolList
        self.bool_dict={'y':True, 'n':False, 'f':False, 't':True}
        try:
            logging.info("Initiating object to boolean datatype conversion")
            for features in self.col:
                self.df[features]=self.dataframe[features].map(self.bool_dict)
            
            logging.info("Object to boolean datatype conversion done successfully")
            return self.df
        except Exception as e:
            logging.info("Object to boolean datatype conversion failed")
            raise CustomException(e,sys)
    

    def target_class_fixing(self, df, output_feature_name):
        try:
            logging.info("Initiating fixing of target feature values")
            self.df=df
            self.col=output_feature_name

            self.df[self.col]=self.df[self.col].str.split(".")
            self.df[self.col]=[val[0].replace(".","") for val in self.df[self.col]]
            self.df[self.col]=self.df[self.col].str.split("[")
            self.df[self.col]=[val[0].replace(".","") for val in self.df[self.col]]
            self.df[self.col]=self.df[self.col].str.strip()

            logging.info("Fixing of target feature values done successfully")
            return self.df
        except Exception as e:
            logging.info("Fixing of target feature values failed")
            raise CustomException(e,sys)
    


class raw_data:
    valid_obj=validation()
    data=valid_obj.replace_val_with_nan("?")
    data=valid_obj.target_class_fixing(data,'disease')
    data=data.drop(['tbg'],axis=1)
    data=valid_obj.obj_to_num(['age', 'tsh', 't3', 'tt4', 't4u', 'fti'],data)
    data=valid_obj.obj_to_bool(data,['on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication', 'sick', 'pregnant', 'thyroid_surgery',
                            'i131_treatment', 'query_hypothyroid', 'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary',
                            'psych', 'tsh_measured', 't3_measured', 'tt4_measured', 't4u_measured', 'fti_measured', 'tbg_measured'])
    
    def fetch_data(self):
        logging.info("Returning fixed data")
        return self.data




