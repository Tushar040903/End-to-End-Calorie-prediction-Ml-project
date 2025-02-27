import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifact", "model.pkl")
            preprocessor_path = os.path.join("artifact", "preprocessor.pkl")
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 User_ID: int,
                 Gender: str,
                 Age: float,
                 Height: float,
                 Weight: float,
                 Duration: float,
                 Heart_Rate: float,
                 Body_Temp: float):
        self.User_ID = User_ID
        self.Gender = Gender
        self.Age = Age
        self.Height = Height
        self.Weight = Weight
        self.Duration = Duration
        self.Heart_Rate = Heart_Rate
        self.Body_Temp = Body_Temp

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "User_ID": [self.User_ID],
                "Gender": [self.Gender],
                "Age": [self.Age],
                "Height": [self.Height],
                "Weight": [self.Weight],
                "Duration": [self.Duration],
                "Heart_Rate": [self.Heart_Rate],
                "Body_Temp": [self.Body_Temp]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
