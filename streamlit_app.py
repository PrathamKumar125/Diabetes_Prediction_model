import numpy as np
import pickle
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

scaler=StandardScaler()
encoder= LabelEncoder()

# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# creating a function for Prediction
def diabetes_prediction(input_data):
    # Convert categorical data to numerical values
    map_name = {'Male': 1, 'Female': 2, 'Others': 3}
    gender = input_data[0]
    input_data[0] = map_name[gender]

    # Fit encoder on the original dataset values for 'Smoking history'
    sample_smoking_history = ['No Info', 'Current', 'Ever', 'Former', 'Never', 'Not Current']  # Provide your dataset values
    encoder.fit(sample_smoking_history)
    input_data[7] = encoder.transform([input_data[7]])[0]

    # Create a DataFrame for the input data
    df = pd.DataFrame([input_data])

    # Load the dataset for scaling
    dataset_for_scaling = pd.read_csv('output.csv')  # Replace with your actual dataset path

    # Fit the scaler on the dataset for scaling
    scaler.fit(dataset_for_scaling)

    # Transform the input data using the fitted scaler
    X = scaler.transform(df)
    prediction = loaded_model.predict(X)
    
    if prediction[0] == 0:
        return 'The person is not diabetic'
    else :
        return 'The person is diabetic'

  
def main():
    # giving a title
    st.title('Diabetes Prediction Web App')
    
    
    # getting the input data from the user
    
    gender= st.selectbox(
    'Please select your Gender',
    ('Male', 'Female', 'Others'))
    st.write('You selected:', gender)      
    
    age = st.text_input('Enter your Age')
    
   
    hypertension = st.checkbox('suffering from Hypertension or not?')
    if hypertension:
      st.write(1)
    else:
       st.write(0)
        
    heart_disease = st.checkbox('suffering from Heart Disease or not?')
    if heart_disease:
      st.write(1)
    else:
       st.write(0)

    
    bmi = st.text_input('Enter your BMI Index') 
    HbA1c_level = st.text_input('Enter your Hemoglobin level')
    blood_glucose_level = st.text_input('Enter your blood glucose level')
    
    smoking_history = st.selectbox(
    "Please select your Smoking history",
    ("No Info", "Current", "Ever", "Former", "Never", "Not Current"))
    st.write('You selected:', smoking_history) 

    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([gender,age,hypertension,heart_disease,bmi,HbA1c_level,blood_glucose_level,smoking_history])
        
        
    st.success(diagnosis)
      
if __name__ == '__main__':
    main()
