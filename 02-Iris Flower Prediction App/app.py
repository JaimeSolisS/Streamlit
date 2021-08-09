import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Simple Iris Flower Prediction
 
This app predicts the **Iris flower** type!
 """)

#Sidebar panel 
st.sidebar.header('User Input Parameters')

#Extract the input parameters from the sidebar and create a Panda's dataframe
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4) #Min Value, #Max Value, #Default Value
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

#assign the values to df variables
df = user_input_features()

st.subheader('User Input parameters')
#print the dataframe
st.write(df)

iris = datasets.load_iris()
#The 4 features: sepal_lenght, sepal_width, petal_lenght, petal_width
X = iris.data
# 0,1,2 
Y = iris.target

# Train model
clf = RandomForestClassifier()
clf.fit(X,Y)

#Make prediction and prediction probability
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)


st.subheader('Class labels and their corresponding index numer')
st.write(iris.target_names)

st.subheader('Prediciton')
st.write(iris.target_names[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)

