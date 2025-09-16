import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🩺 Diabetes Prediction App")
st.markdown("### Predict diabetes risk based on health indicators")
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode", 
                                ["Home", "Train Model", "Make Prediction", "Model Performance"])

# Load or initialize session state for model
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

def load_data():
    """Function to load the diabetes dataset"""
    try:
        # You'll need to upload your diabetes.csv file
        uploaded_file = st.file_uploader("Upload your diabetes dataset (CSV)", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            return df
        else:
            # Sample data structure for demonstration
            st.info("Please upload your diabetes dataset to proceed with training.")
            return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def train_model(df):
    """Function to train the diabetes prediction model"""
    try:
        # Prepare features and target
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store in session state
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.model_trained = True
        st.session_state.accuracy = accuracy
        st.session_state.X_test = X_test_scaled
        st.session_state.y_test = y_test
        st.session_state.y_pred = y_pred
        st.session_state.feature_names = X.columns.tolist()
        
        return True, accuracy
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return False, 0

def make_prediction(input_data):
    """Function to make diabetes prediction"""
    try:
        if st.session_state.model is not None and st.session_state.scaler is not None:
            # Scale the input data
            input_scaled = st.session_state.scaler.transform([input_data])
            
            # Make prediction
            prediction = st.session_state.model.predict(input_scaled)[0]
            prediction_proba = st.session_state.model.predict_proba(input_scaled)[0]
            
            return prediction, prediction_proba
        else:
            st.error("Model not trained. Please train the model first.")
            return None, None
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

# Home Page
if app_mode == "Home":
    st.markdown("""
    ## Welcome to the Diabetes Prediction App! 
    
    This application uses machine learning to predict the likelihood of diabetes based on various health indicators.
    
    ### How it works:
    1. **Train Model**: Upload your diabetes dataset and train the machine learning model
    2. **Make Prediction**: Input health parameters to get diabetes risk prediction
    3. **Model Performance**: View detailed performance metrics of the trained model
    
    ### Dataset Features:
    - **Pregnancies**: Number of times pregnant
    - **Glucose**: Plasma glucose concentration after 2 hours in an oral glucose tolerance test
    - **BloodPressure**: Diastolic blood pressure (mm Hg)
    - **SkinThickness**: Triceps skin fold thickness (mm)
    - **Insulin**: 2-Hour serum insulin (mu U/ml)
    - **BMI**: Body mass index (weight in kg/(height in m)^2)
    - **DiabetesPedigreeFunction**: Diabetes pedigree function
    - **Age**: Age in years
    
    ### Get Started:
    Navigate to the **Train Model** section to upload your dataset and train the model!
    """)

# Train Model Page
elif app_mode == "Train Model":
    st.header("🎯 Train Diabetes Prediction Model")
    
    # Load data
    df = load_data()
    
    if df is not None:
        st.success("Dataset loaded successfully!")
        
        # Display dataset info
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dataset Overview")
            st.write(f"Shape: {df.shape}")
            st.write(f"Features: {df.columns.tolist()}")
        
        with col2:
            st.subheader("Target Distribution")
            outcome_counts = df['Outcome'].value_counts()
            st.write(f"No Diabetes: {outcome_counts[0]}")
            st.write(f"Diabetes: {outcome_counts[1]}")
        
        # Display first few rows
        st.subheader("First 5 rows of the dataset:")
        st.dataframe(df.head())
        
        # Train model button
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training model... This may take a few moments."):
                success, accuracy = train_model(df)
                
                if success:
                    st.success(f"✅ Model trained successfully!")
                    st.success(f"🎯 Model Accuracy: {accuracy:.2%}")
                    st.balloons()
                else:
                    st.error("❌ Failed to train model. Please check your dataset format.")

# Make Prediction Page
elif app_mode == "Make Prediction":
    st.header("🔮 Make Diabetes Prediction")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first in the 'Train Model' section.")
    else:
        st.success("✅ Model is ready for predictions!")
        
        # Create input form
        st.subheader("Enter Health Parameters:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
            glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=100)
            blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
            skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
        
        with col2:
            insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
            bmi = st.number_input("BMI", min_value=0.0, max_value=80.0, value=25.0, format="%.1f")
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
            age = st.number_input("Age", min_value=18, max_value=120, value=30)
        
        # Prediction button
        if st.button("🔍 Predict Diabetes Risk", type="primary"):
            input_data = [pregnancies, glucose, blood_pressure, skin_thickness, 
                         insulin, bmi, dpf, age]
            
            prediction, prediction_proba = make_prediction(input_data)
            
            if prediction is not None:
                # Display results
                st.markdown("---")
                st.subheader("🎯 Prediction Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.error("🚨 High Risk: The model predicts DIABETES")
                        st.markdown("**Recommendation:** Please consult with a healthcare professional for proper evaluation and testing.")
                    else:
                        st.success("✅ Low Risk: The model predicts NO DIABETES")
                        st.markdown("**Recommendation:** Continue maintaining a healthy lifestyle.")
                
                with col2:
                    st.subheader("📊 Confidence Levels")
                    no_diabetes_prob = prediction_proba[0] * 100
                    diabetes_prob = prediction_proba[1] * 100
                    
                    st.metric("No Diabetes", f"{no_diabetes_prob:.1f}%")
                    st.metric("Diabetes", f"{diabetes_prob:.1f}%")
                
                # Progress bars for probabilities
                st.subheader("📈 Risk Assessment")
                st.write("No Diabetes Probability:")
                st.progress(no_diabetes_prob/100)
                st.write("Diabetes Probability:")
                st.progress(diabetes_prob/100)
                
                # Disclaimer
                st.markdown("---")
                st.warning("⚠️ **Disclaimer:** This prediction is for educational purposes only and should not replace professional medical advice. Please consult healthcare professionals for accurate diagnosis and treatment.")

# Model Performance Page
elif app_mode == "Model Performance":
    st.header("📊 Model Performance Analysis")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first to view performance metrics.")
    else:
        # Display accuracy
        st.subheader(f"🎯 Overall Accuracy: {st.session_state.accuracy:.2%}")
        
        # Feature importance
        if hasattr(st.session_state.model, 'feature_importances_'):
            st.subheader("📈 Feature Importance")
            
            feature_importance = pd.DataFrame({
                'feature': st.session_state.feature_names,
                'importance': st.session_state.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=feature_importance, x='importance', y='feature', ax=ax)
            ax.set_title('Feature Importance in Diabetes Prediction')
            st.pyplot(fig)
            plt.close()
        
        # Confusion Matrix
        st.subheader("🔍 Confusion Matrix")
        cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)
        plt.close()
        
        # Classification Report
        st.subheader("📋 Detailed Classification Report")
        report = classification_report(st.session_state.y_test, st.session_state.y_pred, 
                                     target_names=['No Diabetes', 'Diabetes'])
        st.text(report)

# Footer
st.markdown("---")
st.markdown("*Built with ❤️ using Streamlit and scikit-learn*")