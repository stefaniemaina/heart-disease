from altair import Stroke
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ML Libraries
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split

#add title and headers
st.title("Heart Disease Health Indicators")

st.subheader("This is a heart disease health indicator predictor")

st.write("Heart disease refers to a range of conditions that affect the heart and blood vessels.")
st.write("These conditions can lead to heart attacks, irregular heart beats, heart failure; etc")
st.write("This application will assist to predict chances of heart disease")

#load data
df= pd.read_csv("heart_disease_health_indicators.csv")

#show sample data
st.subheader("Sample Data")
st.dataframe(df.head())

# columns to update
columns_to_update = ['GenHlth','Age','Stroke','HighBP']


# Replace 0 with NaN in the specified columns
df[columns_to_update] = df[columns_to_update].replace(0, np.nan)

# Replacing missing values
def median_target(var):   
    temp = df[df[var].notnull()]
    temp = round(temp[[var, 'HeartDiseaseorAttack']].groupby(['HeartDiseaseorAttack'])[[var]].mean().reset_index(), 1)
    return temp

# GenHlth
df.loc[(df['HeartDiseaseorAttack'] == 0 ) & (df['GenHlth'].isnull()), 'GenHlth'] = 1
df.loc[(df['HeartDiseaseorAttack'] == 1 ) & (df['GenHlth'].isnull()), 'GenHlth'] = 5

# Age
df.loc[(df['HeartDiseaseorAttack'] == 0 ) & (df['Age'].isnull()), 'Age'] = 3
df.loc[(df['HeartDiseaseorAttack'] == 1 ) & (df['Age'].isnull()), 'Age'] = 13

# Stroke
df.loc[(df['HeartDiseaseorAttack'] == 0 ) & (df['Stroke'].isnull()), 'Stroke'] = 0
df.loc[(df['HeartDiseaseorAttack'] == 1 ) & (df['Stroke'].isnull()), 'Stroke'] = 1

# HighBP
df.loc[(df['HeartDiseaseorAttack'] == 0 ) & (df['HighBP'].isnull()), 'HighBP'] = 0
df.loc[(df['HeartDiseaseorAttack'] == 1 ) & (df['HighBP'].isnull()), 'HighBP'] = 1


# Add two columns for data visualization
col1, col2 = st.columns(2)

with col1:
    st.header("Correlation Heatmap")
    f, ax = plt.subplots(figsize=(10, 6))
    mask = np.triu(np.ones_like(df.corr(), dtype=bool))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", mask=mask, 
                cmap='coolwarm', vmin=-1, vmax=1) 
    st.pyplot(f)   

with col2:
    # Display a clustermap
    st.header("Clustermap")
    f, ax = plt.subplots(figsize=(10, 6))
    mask = np.triu(np.ones_like(df.corr(), dtype=bool))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", mask=mask, 
                cmap='coolwarm', vmin=-1, vmax=1) 
    st.pyplot(f) 

#data features and target
X = df.drop(columns='HeartDiseaseorAttack')
y = df['HeartDiseaseorAttack']

#columns to fit the model
columns = ['GenHlth','Age','Stroke','HighBP']
X = X[columns]

#scale the data
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# getting user input
name = st.text_input('What is your name?').capitalize()
if name != "":
    st.write("Hello {} please fill below form by selecting from the slider on the sidebar".format(name))
else:
    st.write("Please enter your name")

# Get user input
def get_user_input():
    GenHlth = st.sidebar.slider("GenHlth", 1, 5, 3)
    Age = st.sidebar.slider("Age",1,13,8)
    Stroke = st.sidebar.slider("Stroke",0,1,0)
    HighBP = st.sidebar.slider("HighBP",0,1,0)

    #Store in a dictionary
    user_data = {
        'GenHlth': GenHlth,
        'Age': Age,
        'Stroke': Stroke,
        'HighBP': HighBP

    }
    # Create features dataframe
    features = pd.DataFrame(user_data, index=[0])
    return features
user_input = get_user_input()

#Displaying
st.subheader("Below is your input")
st.dataframe(user_input)

# Button for user to get results
bt = st.button("Get my results")

if bt:
    #create a gradient boosting classifier
    model= GradientBoostingClassifier()
    model.fit(X_train, y_train)
    #get user input features
    prediction = model.predict(user_input)
    if prediction == 1:
        st.write("{}, you have a heart disease".format(name))
    else:
        st.write("{}, you do not have a heart disease".format(name))
                 
#Get model accuracy
model= GradientBoostingClassiffier()
model.fit(X_train, y_train)
st.write("Model accuracy: ", round(metrics.accuracy_score(y_test, model.predict(X_test)),2)*100)
