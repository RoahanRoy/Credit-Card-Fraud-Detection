import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# load data
cc_df = pd.read_csv(r"C:\Users\Roahan Roy\Desktop\Data Science Projects\Data\creditcard.csv")


# The above file shows that the data is imbalanced as there are way less '1s' (Fraud) compared to '0s' (Legitimate)

# We have to balance our data before using Logistic Regression

legit = cc_df[cc_df.Class == 0]
fraud = cc_df[cc_df.Class == 1]

# We need to match the shape of Legit to Fraud to make it balanced
# Undersampling Legit Transactions
legit_sample = legit.sample(n=len(fraud), random_state=2)
cc_df = pd.concat([legit_sample,fraud], axis=0)

# split data into training and testing sets
X = cc_df.drop('Class', axis=1) # All the other 30 features
Y = cc_df['Class'] # Dependent features i.e. 0,1
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, Y_train) # Training the model with 80% data
#ypred = model.predict(X_test) # Using remaining data (20%) for prediction


# Comparing the remaining 20% data in the dataset with the predicted 20% from our model

# Evaluating model performance
train_accuracy = accuracy_score(model.predict(X_train), Y_train)
test_accuracy = accuracy_score(model.predict(X_test), Y_test)

# create Streamlit app
st.title("Credit Card Fraud Detection Model")
st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")

# create input fields for user to enter feature values
input_df = st.text_input('Enter All features')
input_df_splitted = input_df.split(',')
# create a button to submit input and get prediction
submit = st.button("Submit")

if submit:
    # get input feature value
    features = np.array(input_df_splitted, dtype=np.float64)
    # make prediction
    prediction = model.predict(features.reshape(1,-1))
    # display result
    if prediction[0] == 0:
        st.write("Legitimate transaction")
    else:
        st.write("Fraudulent transaction")
