import sys
import pandas as pd
import dill
from src.vehicle.exception import CustomException
from src.vehicle.logger import logger
from src.vehicle.utils.common import load_object

class CustomData:
    def __init__(self,
                 Year: int,
                 Selling_Price: float,
                 Present_Price: float,
                 Kms_Driven: int,
                 Fuel_Type: object,
                 Seller_Type: object,
                 Transmission: object,
                 Owner: object):
    
        self.Year = Year
        self.Selling_Price = Selling_Price
        self.Present_Price = Present_Price
        self.Kms_Driven = Kms_Driven
        self.Fuel_Type = Fuel_Type
        self.Seller_Type = Seller_Type
        self.Transmission = Transmission
        self.Owner = Owner

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Year': [self.Year],
                'Selling_Price': [self.Selling_Price],
                'Present_Price': [self.Present_Price],
                'Kms_Driven': [self.Kms_Driven],
                'Fuel_Type': [self.Fuel_Type],
                'Seller_Type': [self.Seller_Type],
                'Transmission': [self.Transmission],
                'Owner': [self.Owner]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logger.info('Dataframe Gathered')
            return df
        except Exception as e:
            logger.info('Exception Occurred in prediction pipeline')
            raise CustomException(e, sys)
