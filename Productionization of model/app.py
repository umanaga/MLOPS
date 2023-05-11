import os
import streamlit as st
import numpy as np
import pickle
from pickle import load
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

scaler = load(open('models/standard_scaler.pkl', 'rb'))
knn = load(open('models/knn_model.pkl', 'rb'))

st.title(":red[Concrete Strength Prediction]")

cement = float(st.number_input("Enter cemnet(kg in a m^3 mixture): "))

bfs = float(st.number_input("enter Blast Furnace Slag(kg in a m^3 mixture): "))

fa = st.number_input("enter Fly Ash(kg in a m^3 mixture): ")

Water = st.number_input('enter Water(kg in a m^3 mixture): ') 

Superplasticizer = float(st.number_input('enter Superplasticizer(kg in a m^3 mixture): '))

coarse_aggregate = float(st.number_input('enter Coarse Aggregate(kg in a m^3 mixture): '))

Fine_Aggregate = float(st.number_input('enter Fine Aggregate(kg in a m^3 mixture): '))

age = float(st.number_input('enter age(in days): '))



if st.button('Predict'):
    query_point = np.array([cement,bfs,fa,Water,Superplasticizer,coarse_aggregate,Fine_Aggregate,age])
    query_point = query_point.reshape(1, -1)
    query_point_transformed = scaler.transform(query_point)
    prediction = knn.predict(query_point_transformed)
    st.subheader(":blue[Concrete Strength Prediction :] :green{}".format((prediction)))
else:
    pass


        
    