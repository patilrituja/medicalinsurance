import pickle
import json
import config1
import numpy as np

class MedicalInsurence():

    def __init__(self, age, gender, bmi, children, smoker, region) :
        self.age = age
        self.gender = gender
        self.bmi = bmi
        self.children = children
        self.smoker = smoker
        self.region = region
        return 

    def __load_model(self): # Private Method
        # Load Model File
        with open(r'artifacts/knn_reg_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
            print('self.model >>',self.model)

        # Load Project Data 
        with open(r'artifacts/project_data.json','r') as f:
            self.project_data = json.load( f)
            print("Project Data :",self.project_data)

        # Load Normal Scaler File
        with open(r'artifacts/normal_scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
            print('self.scaler >>',self.scaler)

    def get_predicted_price(self): # Public Method
        self.__load_model()
        test_array = np.zeros((1,self.model.n_features_in_))
        test_array[0][0] = self.age
        test_array[0][1] = self.project_data['Gender'][self.gender]
        test_array[0][2] = self.bmi
        test_array[0][3] = self.children
        test_array[0][4] = self.project_data['Smoker'][self.smoker]
        region = 'region_' + self.region
        index = self.project_data['Column Names'].index(region)

        test_array[0][index] = 1

        print("Test Array is :",test_array)
        scaled_test_array = self.scaler.transform(test_array)

        predicted_charges = np.around(self.model.predict(scaled_test_array)[0],3)
        print("Predicted Charges :", predicted_charges)
        return predicted_charges