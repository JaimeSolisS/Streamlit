import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Penguin Clasification Type

Data obtained from the [palmerpenguins library](https://github.com/allisonhorst/palmerpenguins) in R by Allison Horst.

""")
st.image('https://github.com/allisonhorst/palmerpenguins/blob/master/man/figures/logo.png?raw=true')
st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/JaimeSolisS/Streamlit/master/Penguin%20Classification/example.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
st.sidebar.write("Or directly input the parameters below")
if uploaded_file is not None: 
    input_df = pd.read_csv(uploaded_file)
else: 
    def user_input_features():
        island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
        sex = st.sidebar.selectbox('Sex',('male','female'))
        bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1,59.6,43.9)
        bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1,21.5,17.2)
        flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0,231.0,201.0)
        body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4207.0)
        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combine user input feature with entire dataset (this will be useful for encoding phase)
penguins_raw = pd.read_csv('penguins.csv')
penguins = penguins_raw.drop(columns=['species'])
df = pd.concat([input_df, penguins], axis=0)

# Encoding ordinal features
encode = ['sex', 'island']
for col in encode: 
        dummy = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dummy], axis=1)
        del df[col]
df = df[:1] # Select the first row (user input data)

# Display user input features
st.subheader('User Input features')

if uploaded_file is not None: 
    st.write(df)
else: 
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters')
    st.write(df)

# Read in saved classification model
load_clf = pickle.load(open('model-building/penguins_clf.pkl', 'rb'))

# Apply model to make predicitons
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

st.subheader('Prediction')
penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.write(penguins_species[prediction])

st.subheader('Prediciton Probability')
st.write(prediction_proba)

