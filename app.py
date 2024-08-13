import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model from a pickle file
@st.cache_resource
def load_model():
    with open('xgboost_model_fraud_detection.pkl', 'rb') as f:  # Replace 'model_rf.pkl' with your actual pickle file path
        model = pickle.load(f)
    return model

model = load_model()

# Define the Streamlit app
def main():
    st.title('Credit Card Fraud Detection')
    st.write('Enter transaction details to predict if it is fraudulent.')

    # Define input fields based on the features used during training
    feature_names = ['Time_hour', 'Time_min', 'V2', 'V3', 'V4', 'V9', 
                     'V10', 'V11', 'V12', 'V14', 'V16', 'V17', 'V18', 
                     'V19', 'V27', 'Amount']

    input_data = {}
    for feature in feature_names:
        input_data[feature] = st.number_input(f'Enter {feature}:', value=0.0)

    # Convert input data to a DataFrame
    input_df = pd.DataFrame([input_data])

    # Apply the log transformation to the Amount feature
    input_df['Amount'] = np.log(input_df['Amount'] + 0.001)

    # Predict button
    if st.button('Predict'):
        prediction = model.predict(input_df)
        result = 'Fraudulent' if prediction[0] == 1 else 'Not Fraudulent'
        st.write(f'The transaction is predicted to be: **{result}**')

if __name__ == '__main__':
    main()
