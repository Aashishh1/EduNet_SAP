import streamlit as st
import pickle
import numpy as np

# Set the page configuration
st.set_page_config(page_title="Air Quality Forecast Machine Learning Model", layout="centered")

# Cache the model using the new caching mechanism
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

# Load the model
model = load_model()

# Sidebar for inputs
st.sidebar.title("Enter Pollutant Levels")
st.sidebar.write("Adjust the sliders or enter values manually.")

# Input fields for pollutants in the sidebar
PM2_5 = st.sidebar.slider("PM2.5 (µg/m³)", 0.0, 500.0, step=0.1, value=50.0)
PM10 = st.sidebar.slider("PM10 (µg/m³)", 0.0, 500.0, step=0.1, value=50.0)
NO = st.sidebar.slider("NO (µg/m³)", 0.0, 500.0, step=0.1, value=20.0)
NO2 = st.sidebar.slider("NO2 (µg/m³)", 0.0, 500.0, step=0.1, value=20.0)
NOx = st.sidebar.slider("NOx (µg/m³)", 0.0, 500.0, step=0.1, value=20.0)
NH3 = st.sidebar.slider("NH3 (µg/m³)", 0.0, 500.0, step=0.1, value=20.0)
CO = st.sidebar.slider("CO (mg/m³)", 0.0, 10.0, step=0.01, value=0.5)
SO2 = st.sidebar.slider("SO2 (µg/m³)", 0.0, 500.0, step=0.1, value=20.0)
O3 = st.sidebar.slider("O3 (µg/m³)", 0.0, 500.0, step=0.1, value=50.0)
Benzene = st.sidebar.slider("Benzene (µg/m³)", 0.0, 500.0, step=0.1, value=1.0)
Toluene = st.sidebar.slider("Toluene (µg/m³)", 0.0, 500.0, step=0.1, value=1.0)
Xylene = st.sidebar.slider("Xylene (µg/m³)", 0.0, 500.0, step=0.1, value=1.0)

# Title and description in the main area
st.title("Air Quality Forecast: Machine Learning Model")
st.write("""
Enter the pollutant levels using the sliders in the sidebar.  
Click the "Predict AQI" button to calculate the Air Quality Index (AQI).
""")

# Predict button and output
if st.button("Predict AQI"):
    try:
        # Prepare input
        input_features = np.array([[PM2_5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene]])
        
        # Make prediction
        prediction = model.predict(input_features)[0]
        
        # Display result
        st.success(f"Predicted AQI: {prediction:.2f}")
        st.write("""
        The Air Quality Index (AQI) is a measure used to communicate how polluted the air currently is or how polluted it is forecasted to become.
        """)
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Footer
st.write("---")
st.caption("Ashish Mishra 🧑‍💻")
