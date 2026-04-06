import streamlit as st
import pandas as pd
import gdown
from sklearn.tree import DecisionTreeClassifier

# --- DOWNLOAD DATA ---
file_id = "1FBCKZI4KKtlvZaY8XgvAcvLHo5We0ORF"
url = f"https://drive.google.com/uc?export=download&id={file_id}"
output = "Iris.csv"

@st.cache_data
def load_data():
    gdown.download(url, output, quiet=False)
    return pd.read_csv(output)

df = load_data()

st.write("Iris Dataset Preview")
st.write(df)

# --- PREPARE DATA ---
# Example assumes Iris dataset format
X = df.drop(columns=["Id","Species"])   # features
y = df["Species"]                 # target

# --- TRAIN MODEL ---
model = DecisionTreeClassifier()
model.fit(X, y)

# --- USER INPUT ---
st.sidebar.header("Input Features")

sepal_length = st.sidebar.number_input("Sepal Length", 4.0, 8.0)
sepal_width  = st.sidebar.number_input("Sepal Width", 2.0, 5.0)
petal_length = st.sidebar.number_input("Petal Length", 1.0, 7.0)
petal_width  = st.sidebar.number_input("Petal Width", 0.1, 2.5)

# --- PREDICTION ---
input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                          columns=X.columns)

prediction = model.predict(input_data)

predictBtn = st.sidebar.button("Predict")

if predictBtn:
    st.write("### Prediction:")
    st.write(prediction[0])
    st.success("Successfully Predict")
