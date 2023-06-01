import streamlit as st
import numpy as np
import pandas as pd
import pickle

model = pickle.load(open("knn.pkl", "rb"))

cuts = ['Fair', 'Good', 'Ideal', 'Premium', 'Very Good']
colors = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
clarities = ['I1', 'IF', 'SI1', 'SI2', 'VS1', 'VS2', 'VVS1', 'VVS2']

st.title("Diamond Price Predicton")
carat = st.number_input("Carat")
depth = st.number_input("Depth")
table = st.number_input("Table")
x = st.number_input("X")
y = st.number_input("Y")
z = st.number_input("Z")

if st.button("Predict"):
	test = np.array([[carat, depth, table, x, y, z]])
	res = model.predict(test)
	print(res)
	st.success("Predicted: " + str(res[0]))
