import numpy as np 
import joblib
from sklearn.preprocessing import OrdinalEncoder


def preprocessdata(Gender, Married, Education, Self_Employed, ApplicantIncome,
       CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History,
       Property_Area):
    test_data = [[Gender, Married, Education, Self_Employed, ApplicantIncome,
       CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History,
       Property_Area] ]   

    ord_enc = OrdinalEncoder() 
    columns_to_encode = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
    test_data[columns_to_encode] = ord_enc.fit_transform(test_data[[columns_to_encode]])
    
    test_data[["Gender",'Married','Education','Self_Employed','Property_Area']] = test_data[["Gender",'Married','Education','Self_Employed','Property_Area']].astype("int")
    
    trained_model = joblib.load("model.pkl")
    prediction = trained_model.predict(test_data) 

    return prediction