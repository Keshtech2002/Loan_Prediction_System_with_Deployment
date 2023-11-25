import numpy as np 
import joblib
from sklearn.preprocessing import OrdinalEncoder


def preprocessdata(Gender, Married, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area):
    test_data = [[Gender, Married, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area]]   

    ord_enc = OrdinalEncoder()
    
    # Convert test_data to a 2D array
    test_data_array = np.array(test_data)
    
    
#     test_data[["Gender",'Married','Education','Self_Employed','Property_Area']] = ord_enc.fit_transform(test_data[["Gender",'Married','Education','Self_Employed','Property_Area']])
    
#     test_data[["Gender",'Married','Education','Self_Employed','Property_Area']] = test_data[["Gender",'Married','Education','Self_Employed','Property_Area']].astype("int")
    columns_to_convert = [0, 1, 2, 3, 9]

    test_data_array[:, [0, 1, 2, 3, 9]] = ord_enc.fit_transform(test_data_array[:, [0, 1, 2, 3, 9]])
    
#     test_data_array[:, [0, 1, 2, 3, 9]] = test_data_array[:, [0, 1, 2, 3, 9]].astype("int")

    test_data_array[:, columns_to_convert] = np.where(test_data_array[:, columns_to_convert].astype(str).astype(float) == 0.0, 0, test_data_array[:, columns_to_convert].astype(float)).astype(int)
    
    trained_model = joblib.load("model.pkl")
    prediction = trained_model.predict(test_data_array) 

    return prediction