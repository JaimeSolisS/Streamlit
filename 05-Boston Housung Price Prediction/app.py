import streamlit as st
import pandas as pd
import shap 
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import pickle
st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
# Boston House Price Prediction App

This app predicts the **Boston House Price**
""")
st.write('---')

# Loads the Boston House Price Dataset
boston = datasets.load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
Y = pd.DataFrame(boston.target, columns=["MEDV"])

# Sidebar 
# Header of Specify Input Parameters 
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    CRIM = st.sidebar.slider('CRIM --- per capita crime rate by town', X.CRIM.min(), X.CRIM.max(), X.CRIM.mean())
    ZN = st.sidebar.slider('ZN - prop. of residential land zoned with 25K+ sq.ft', X.ZN.min(), X.ZN.max(), X.ZN.mean())
    INDUS = st.sidebar.slider('INDUS - prop. of non-retail business acres per town', X.INDUS.min(), X.INDUS.max(), X.INDUS.mean())
    CHAS = st.sidebar.slider('CHAS - Charles River dummy variable', X.CHAS.min(), X.CHAS.max(), X.CHAS.mean())
    NOX = st.sidebar.slider('NOX - nitrogen oxides concentration (pp 10 million)', X.NOX.min(), X.NOX.max(), X.NOX.mean())
    RM = st.sidebar.slider('RM - average number of rooms per dwelling', X.RM.min(), X.RM.max(), X.RM.mean())
    AGE = st.sidebar.slider('AGE - proportion of owner-occupied units built prior to 1940', X.AGE.min(), X.AGE.max(), X.AGE.mean())
    DIS = st.sidebar.slider('DIS - weighted mean of distances to employment centres', X.DIS.min(), X.DIS.max(), X.DIS.mean())
    RAD = st.sidebar.slider('RAD - index of accessibility to radial highways', X.RAD.min(), X.RAD.max(), X.RAD.mean())
    TAX = st.sidebar.slider('TAX - full-value property-tax rate per \$10,000.', X.TAX.min(), X.TAX.max(), X.TAX.mean())
    PTRATIO = st.sidebar.slider('PTRATIO - pupil-teacher ratio by town', X.PTRATIO.min(), X.PTRATIO.max(), X.PTRATIO.mean())
    B = st.sidebar.slider('B - 1000(Bk-0.63)^2 where Bk is the prop. of african american by town', X.B.min(), X.B.max(), X.B.mean())
    LSTAT = st.sidebar.slider('LSTAT - lower status of the population (percent)', X.LSTAT.min(), X.LSTAT.max(), X.LSTAT.mean())
    data = {'CRIM': CRIM,
            'ZN': ZN,
            'INDUS': INDUS,
            'CHAS': CHAS,
            'NOX': NOX,
            'RM': RM,
            'AGE': AGE,
            'DIS': DIS,
            'RAD': RAD,
            'TAX': TAX,
            'PTRATIO': PTRATIO,
            'B': B,
            'LSTAT': LSTAT}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel 

# Print specified input parameters
st.header('Specified Input Parameters')
st.write(df)
st.write('---')

# Read in saved classification model
model = pickle.load(open('boston_model.pkl', 'rb'))

# Apply Model to Make Prediction
prediction = model.predict(df)

st.header('Prediction of MEDV')
st.subheader('median value of owner-occupied homes in \$1000s')
st.write(prediction)
st.write('---')

#Explaining the model's predicitions using SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches="tight")