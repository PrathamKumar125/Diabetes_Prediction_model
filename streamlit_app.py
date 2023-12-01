import numpy as np
import pickle
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

scaler=StandardScaler()
encoder= LabelEncoder()

# df_=pd.read_csv('diabetes_prediction_dataset.csv')
# scaler.fit_transform(df_)

# loading the saved model
loaded_model = pickle.load(open('C:/Users/prath/Downloads/Diabetes/trained_model.sav', 'rb'))


# creating a function for Prediction


# def diabetes_prediction(input_data):

#     # changing the input_data to numpy array
#     input_data_as_numpy_array = np.asarray(input_data)

#     # reshape the array as we are predicting for one instance
#     input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#     print(input_data_reshaped)
#     df = pd.DataFrame(input_data_reshaped)

#     map_name={'Male':1,'Female':2,'Other':3} 
#     df[0]=df[0].map(map_name)

#     df[7]= encoder.fit_transform(df[7])

#     scaler.fit_transform(df)
#     X= scaler.transform(df)

#     prediction = loaded_model.predict(X)
#     print(prediction)

#     if (prediction[0] == 0):
#          return 'The person is not diabetic'
#     else:
#        return 'The person is diabetic'


# ... (Previous code remains the same)

def diabetes_prediction(input_data):
    # Convert categorical data to numerical values
    map_name = {'Male': 1, 'Female': 2, 'Others': 3}
    input_data[0] = map_name[input_data[0]]

    # Fit encoder on the original dataset values for 'Smoking history'
    sample_smoking_history = ['No Info', 'Current', 'Ever', 'Former', 'Never', 'Not Current']  # Provide your dataset values
    encoder.fit(sample_smoking_history)
    input_data[7] = encoder.transform([input_data[7]])[0]

    # Create a DataFrame for the input data
    df = pd.DataFrame([input_data])

    # Load the dataset for scaling
    dataset_for_scaling = pd.read_csv('diabetes_prediction_dataset.csv')  # Replace with your actual dataset path

    # Fit the scaler on the dataset for scaling
    scaler.fit(dataset_for_scaling)

    # Convert columns to numeric (age, BMI, HbA1c_level, blood_glucose_level)
    numeric_columns = [1, 4, 5, 6]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col])

    # Transform the input data using the fitted scaler
    X = scaler.transform(df)

    prediction = loaded_model.predict(X)

    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

# ... (Remaining code remains the same)

  
    
  
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

    smoking_history = st.selectbox(
    "Please select your Smoking history",
    ("No Info", "Current", "Ever", "Former", "Never", "Not Current"))
    st.write('You selected:', smoking_history) 
    
    bmi = st.text_input('Enter your BMI Index') 


    HbA1c_level = st.text_input('Enter your Hemoglobin level')
    blood_glucose_level = st.text_input('Enter your blood glucose level')
    

    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([gender,age,hypertension,heart_disease,bmi,HbA1c_level,blood_glucose_level,smoking_history])
        
        
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
  
    
  