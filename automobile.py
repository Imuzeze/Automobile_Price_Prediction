import pandas as pd 
import streamlit as st
import joblib 
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('automobile.csv')

#Add Header and Sub Header
st.markdown("<h1 style = 'color: #00224D text-align: center; font-size: 60px; font-family: geneva'>AUTOMOBILE PRICE PREDICTOR</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #F11A7B; text-align: center; font-family: cursive '>Built By Joshua</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html= True)

#Add An Image
st.image('pngwing.com (3).png',  caption = 'Built By Josh')

#Add Project Problem Statement

st.header('Project Background Information',divider = True)
st.write('The primary objective of this machine learning project is to develop an accurate and robust predictive model for estimating the price of a car based on its various features. By leveraging advanced machine learning algorithms, the aim is to create a model that can analyze and learn from historical car data, encompassing attributes such as make, model, year, mileage, engine type, fuel efficiency, and other relevant parameters')

st.markdown("<h2 style = 'color: #132043; text-align: center; font-family: montserrat '>Background Of Study</h2>", unsafe_allow_html = True)



#Sidebar Designs

#To add anything to the side bar of the page
st.sidebar.image('pngwing.com (2).png')



st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)

#To add the project dataset
st.divider()
st.header('Automobile Data')
st.dataframe(data, use_container_width = True)


#To add space to the sidebar before adding writeup to give line spaace
st.sidebar.markdown("<br>", unsafe_allow_html= True)
st.sidebar.markdown("<br>", unsafe_allow_html= True)


#User Inputs
make = st.sidebar.selectbox('make', data['make'].unique())
num_of_doors = st.sidebar.selectbox('num-of-doors', data['num-of-doors'].unique())
body_style = st.sidebar.selectbox('body-style', data['body-style'].unique())
engine_type = st.sidebar.selectbox('engine-type', data['engine-type'].unique())
num_of_cylinders = st.sidebar.selectbox('num-of-cylinders', data['num-of-cylinders'].unique())
engine_size = st.sidebar.number_input('engine-size', data['engine-size'].min(), data['engine-size'].max())
compression_ratio = st.sidebar.number_input('compression-ratio', data['compression-ratio'].min(), data['compression-ratio'].max())
#horse_power = st.sidebar.number_input('horsepower', data['horsepower'].min(), data['horsepower'].max())
# highway_mpg = st.sidebar.number_input('highway-mpg', data['highway-mpg'].min(), data['highway-mpg'].max())


#import transformers.... used to bring our scaler 
make_encoder = joblib.load('make_encoder.pkl')
num_of_doors_encoder = joblib.load('num-of-doors_encoder.pkl')
body_style_encoder = joblib.load('body-style_encoder.pkl')
engine_type_encoder = joblib.load('engine-type_encoder.pkl')
num_of_cylinders_encoder = joblib.load('num-of-cylinders_encoder.pkl')

#user input DataFrame
user_input = pd.DataFrame()
user_input['make']= [make]
user_input['num-of-doors']= [num_of_doors]
user_input['body-style']= [body_style]
user_input['engine-type']= [engine_type]
user_input['num-of-cylinders']= [num_of_cylinders]
user_input['engine-size']= [engine_size]
user_input['compression-ratio']= [compression_ratio]


st.markdown("<br>", unsafe_allow_html= True)
st.header('Input Variable')
st.dataframe(user_input, use_container_width = True)

st.header('Transformed Input Variable')
st.dataframe(user_input, use_container_width = True)



# transform users input according to training scale and encoding
user_input['make'] = make_encoder.transform(user_input[['make']])
user_input['num-of-doors'] = num_of_doors_encoder.transform(user_input[['num-of-doors']])
user_input['body-style'] = body_style_encoder.transform(user_input[['body-style']])
user_input['engine-type'] = engine_type_encoder.transform(user_input[['engine-type']])
user_input['num-of-cylinders'] = num_of_cylinders_encoder.transform(user_input[['num-of-cylinders']])



# Display prediction
st.header('Predicted Price')

# Predict using the loaded model
model = joblib.load('Automobile.pkl')

predicted_price = model.predict(user_input)

# Display the predicted price value
st.write(f"The predicted price is: {predicted_price[0]}")