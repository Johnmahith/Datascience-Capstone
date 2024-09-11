import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('your_saved_model.h5')

# Function to preprocess user input
def preprocess_input(df):
    # Perform the same preprocessing as in the training
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['year'] = df['DATE'].dt.year
    df['month'] = df['DATE'].dt.month
    df['day'] = df['DATE'].dt.day
    df = df.drop(columns=['DATE'])
    
    # Encode categorical variables
    label_encoder = LabelEncoder()
    df['STATUS'] = label_encoder.fit_transform(df['STATUS'])

    # Standard scaling
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df)
    
    return scaled_features

# Streamlit UI
st.title('Employee Termination Prediction')

# Upload CSV file
uploaded_file = st.file_uploader("/content/drive/MyDrive/MFG10YearTerminationData.csv", type=["csv"])

if uploaded_file is not None:
    # Load the uploaded data
    data = pd.read_csv(uploaded_file)

    # Display the uploaded data
    st.write("Uploaded Data:")
    st.write(data)

    # Preprocess the input data
    processed_data = preprocess_input(data)

    # Make predictions
    predictions = model.predict(processed_data)
    predicted_classes = np.where(predictions > 0.5, 'Terminated', 'Active')

    # Display predictions
    st.write("Predictions:")
    st.write(pd.DataFrame(predicted_classes, columns=["Employee Status"]))

