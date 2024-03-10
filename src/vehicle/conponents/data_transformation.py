
import os
from src.vehicle.logger import logging
import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler,OrdinalEncoder
from sklearn.model_selection import train_test_split
import pickle
from src.vehicle.entity import DataTransformationConfig




class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def parse_present_year(self, final_dataset):
        final_dataset['Current Year'] = 2020
        final_dataset['no_year'] = final_dataset['Current Year'] - final_dataset['Year']
        final_dataset.drop(['Year'], axis=1, inplace=True)
        final_dataset.drop(['Current Year'], axis=1, inplace=True)  # Corrected inplace parameter

    def get_data_transformer_obj(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            # Define which columns should be ordinal-encoded and which should be scaled
            numerical_columns=['Present_Price', 'Kms_Driven', 'Owner', 'no_year']
            categorical_columns=['Fuel_Type', 'Seller_Type', 'Transmission']
            
            # Define the custom ranking for each ordinal variable
            Fuel_Type = ['Petrol' ,'Diesel', 'CNG']
            Seller_Type = ['Dealer' ,'Individual']
            Transmission = ['Manual', 'Automatic']

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            # Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinal_encoder', OrdinalEncoder(categories=[Fuel_Type, Seller_Type, Transmission])),
                    ('scaler', StandardScaler())
                ]
            )

            logging.info(f'Categorical Columns : {categorical_columns}')
            logging.info(f'Numerical Columns   : {numerical_columns}')

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )
            
            return preprocessor

        except Exception as e:
            logging.error(f"Error in get_data_transformer_object: {str(e)}")

    def initiate_data_transformation(self):
        try:
            data = pd.read_csv(self.config.data_path)

            # Parse present year before splitting data
            self.parse_present_year(data)

            # Split the data into training and test sets. (0.75, 0.25) split.
            train, test = train_test_split(data)

            train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
            test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

            logging.info("Split data into training and test sets")
            logging.info(train.shape)
            logging.info(test.shape)

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformer_obj()

            target_column_name = 'Selling_Price'

            # Separate input features and target features
            input_feature_train_df = train.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train[target_column_name]

            input_feature_test_df = test.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test[target_column_name]

            # Apply the preprocessing object on training and test input features
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine input features and target features
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save preprocessing object
            preprocessing_obj_file = os.path.join("artifacts", 'data_transformation', 'preprocessing_obj.pkl')
            with open(preprocessing_obj_file, 'wb') as file:
                pickle.dump(preprocessing_obj, file)

            logging.info("Saved preprocessing object.")
            logging.info("Transformation of the data is completed")

            return train_arr, test_arr, preprocessing_obj_file

        except Exception as e:
            logging.error(f"Error in initiate_data_transformation: {str(e)}")
            # Raise the exception so it can be caught in the calling code
            raise e

