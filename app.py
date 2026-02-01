import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="🍕 Food Delivery Time Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling with fixed text visibility
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 2rem;
        background-color: #0e1117;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #FF4B4B 0%, #FF6B6B 100%);
        color: white !important;
        font-weight: bold;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        border: none;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8B8B 100%);
        box-shadow: 0 6px 20px rgba(255, 75, 75, 0.4);
        transform: translateY(-2px);
    }
    
    /* Prediction box with fixed text colors */
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.4);
        margin: 1rem 0;
    }
    .prediction-box h2 {
        margin: 0;
        color: #ffffff !important;
        font-size: 1.5rem;
        font-weight: 600;
    }
    .prediction-box h1 {
        margin: 0.5rem 0;
        font-size: 4rem;
        color: #ffffff !important;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .prediction-box h3 {
        margin: 0;
        color: #ffffff !important;
        font-size: 1.3rem;
        font-weight: 500;
    }
    
    /* Metric cards with proper contrast */
    .metric-card {
        background: linear-gradient(145deg, #1e2130 0%, #2a2d3e 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        text-align: center;
        border: 1px solid #3a3d4f;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
    }
    .metric-card h4 {
        margin: 0 0 0.5rem 0;
        color: #a0a0a0 !important;
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card h2 {
        margin: 0;
        color: #ffffff !important;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    /* Headers */
    h1 {
        color: #ffffff !important;
        font-weight: 700;
        text-align: center;
    }
    h2, h3 {
        color: #e0e0e0 !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d2e 0%, #16192a 100%);
    }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    
    /* Input widgets */
    .stSlider > div > div > div {
        background-color: #FF4B4B;
    }
    
    /* Success/Info/Warning messages */
    .stSuccess, .stInfo, .stWarning {
        border-radius: 10px;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #3a3d4f, transparent);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the pre-trained model"""
    model_path = 'model.pkl'
    
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            return model, None
        except Exception as e:
            return None, f"Error loading model: {str(e)}"
    else:
        return None, "model.pkl file not found. Please ensure the model file is in the same directory as app.py"

def predict_delivery_time(model, input_data):
    """Make prediction using the trained model"""
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    return prediction

# Main app
def main():
    # Header with emoji
    st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h1 style='font-size: 3rem; margin-bottom: 0;'>🍕 Food Delivery Time Predictor</h1>
            <p style='color: #a0a0a0; font-size: 1.2rem; margin-top: 0.5rem;'>
                AI-Powered Delivery Time Estimation using Random Forest 🚀
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        # Logo/Icon at top
        st.markdown("""
            <div style='text-align: center; padding: 1rem 0;'>
                <div style='font-size: 4rem;'>🚚</div>
                <h2 style='margin-top: 0.5rem;'>Delivery Predictor</h2>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 📊 About This App")
        st.markdown("""
        <div style='background: linear-gradient(145deg, #1e2130 0%, #2a2d3e 100%); 
                    padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
            <p style='color: #e0e0e0; margin: 0; line-height: 1.6;'>
                This AI-powered application predicts food delivery times based on multiple factors including distance, weather, traffic, and more.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🎯 Features Used")
        features = [
            "📍 Distance (km)",
            "🌤️ Weather Conditions",
            "🚦 Traffic Level",
            "⏰ Time of Day",
            "🚗 Vehicle Type",
            "🍳 Preparation Time",
            "👨‍💼 Courier Experience"
        ]
        for feature in features:
            st.markdown(f"- {feature}")
        
        st.markdown("---")
        
        st.markdown("### 🤖 Model Info")
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1rem; border-radius: 10px; color: white;'>
            <p style='margin: 0; font-weight: 600;'>Algorithm: Random Forest</p>
            <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem;'>Ensemble learning with multiple decision trees</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Load model
    model, error = load_model()
    
    if model is None:
        st.error(f"⚠️ {error}")
        st.info("💡 Please ensure 'model.pkl' is in the same directory as this app.")
        return
    
    st.success("✅ Model loaded successfully! Ready to make predictions.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Create two columns for input
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("### 📍 Delivery Details")
        
        distance = st.slider(
            "Distance (km)", 
            min_value=0.5, 
            max_value=25.0, 
            value=10.0, 
            step=0.5,
            help="Distance between restaurant and delivery location"
        )
        
        preparation_time = st.slider(
            "Preparation Time (minutes)", 
            min_value=1, 
            max_value=40, 
            value=15, 
            step=1,
            help="Time required to prepare the order"
        )
        
        courier_experience = st.slider(
            "Courier Experience (years)", 
            min_value=0.0, 
            max_value=15.0, 
            value=3.0, 
            step=0.5,
            help="Experience level of the delivery courier"
        )
    
    with col2:
        st.markdown("### 🌤️ Environmental Conditions")
        
        weather = st.selectbox(
            "Weather Condition", 
            options=['Clear', 'Rainy', 'Cloudy', 'Foggy', 'Windy', 'Snowy'],
            help="Current weather conditions"
        )
        
        traffic_level = st.select_slider(
            "Traffic Level", 
            options=['Low', 'Medium', 'High'],
            value='Medium',
            help="Current traffic conditions"
        )
        
        time_of_day = st.selectbox(
            "Time of Day", 
            options=['Morning', 'Afternoon', 'Evening', 'Night'],
            help="Time period of the delivery"
        )
        
        vehicle_type = st.selectbox(
            "Vehicle Type", 
            options=['Bike', 'Scooter', 'Car'],
            help="Type of delivery vehicle"
        )
    
    # Prediction section
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    
    col_pred1, col_pred2, col_pred3 = st.columns([1, 2, 1])
    
    with col_pred2:
        predict_btn = st.button("🔮 Predict Delivery Time", use_container_width=True)
    
    if predict_btn:
        # Prepare input data
        input_data = {
            'Distance_km': distance,
            'Weather': weather,
            'Traffic_Level': traffic_level,
            'Time_of_Day': time_of_day,
            'Vehicle_Type': vehicle_type,
            'Preparation_Time_min': preparation_time,
            'Courier_Experience_yrs': courier_experience
        }
        
        try:
            # Make prediction
            predicted_time = predict_delivery_time(model, input_data)
            
            # Display prediction in a beautiful box
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
                <div class="prediction-box">
                    <h2>⏱️ Estimated Delivery Time</h2>
                    <h1>{predicted_time:.1f}</h1>
                    <h3>minutes</h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Additional insights with metric cards
            st.markdown("<br>", unsafe_allow_html=True)
            
            col_a, col_b, col_c = st.columns(3, gap="medium")
            
            with col_a:
                st.markdown(f"""
                    <div class="metric-card">
                        <h4>🚗 Distance</h4>
                        <h2>{distance:.1f} km</h2>
                    </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                st.markdown(f"""
                    <div class="metric-card">
                        <h4>🌤️ Conditions</h4>
                        <h2>{weather} / {traffic_level}</h2>
                    </div>
                """, unsafe_allow_html=True)
            
            with col_c:
                st.markdown(f"""
                    <div class="metric-card">
                        <h4>📦 Prep Time</h4>
                        <h2>{preparation_time} min</h2>
                    </div>
                """, unsafe_allow_html=True)
            
            # Smart feedback based on prediction
            st.markdown("<br>", unsafe_allow_html=True)
            
            if predicted_time < 30:
                st.success("🎉 Excellent! Your order should arrive quickly. Fast delivery expected!")
            elif predicted_time < 45:
                st.info("⏳ Good! Your order is on the way. Average delivery time expected.")
            elif predicted_time < 60:
                st.warning("🕐 Your order might take a bit longer due to current conditions.")
            else:
                st.error("⚠️ Longer delivery time expected. Consider factors like distance and traffic.")
        
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.info("Please check that all input features match the training data format.")
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p>Made with ❤️ using Streamlit & Random Forest ML</p>
            <p style='font-size: 0.8rem;'>© 2026 Food Delivery Time Predictor</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()