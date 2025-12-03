"""
Automated Medical Email Triage and Patient Information Extraction System
Complete Implementation with ML Model Training
"""

import pandas as pd
import numpy as np
import re
import pickle
import warnings
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st

warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: DATA PREPARATION AND MODEL TRAINING
# ============================================================================

def download_and_prepare_dataset():
    """
    Download and prepare the Disease Symptom Prediction dataset
    This creates a comprehensive symptom-disease dataset
    """
    print("Creating comprehensive medical dataset...")
    
    # Comprehensive disease-symptom dataset
    diseases_symptoms = {
        'Migraine': ['headache', 'blurred vision', 'dizziness', 'nausea', 'sensitivity to light', 'vomiting'],
        'Hypertension': ['headache', 'chest pain', 'shortness of breath', 'dizziness', 'fatigue', 'irregular heartbeat'],
        'Gastritis': ['abdominal pain', 'nausea', 'vomiting', 'loss of appetite', 'bloating', 'indigestion'],
        'Asthma': ['cough', 'wheezing', 'breathing difficulty', 'chest tightness', 'shortness of breath'],
        'Arthritis': ['joint pain', 'swelling', 'stiffness', 'reduced range of motion', 'fatigue'],
        'Eczema': ['rash', 'itching', 'dry skin', 'red patches', 'skin inflammation'],
        'Angina': ['chest pain', 'shortness of breath', 'palpitation', 'sweating', 'fatigue', 'nausea'],
        'Vertigo': ['dizziness', 'nausea', 'balance problems', 'spinning sensation', 'vomiting'],
        'Bronchitis': ['cough', 'mucus production', 'fatigue', 'shortness of breath', 'chest discomfort', 'fever'],
        'Pneumonia': ['cough', 'fever', 'chest pain', 'breathing difficulty', 'fatigue', 'sweating'],
        'Diabetes': ['increased thirst', 'frequent urination', 'fatigue', 'blurred vision', 'weight loss'],
        'Osteoporosis': ['back pain', 'bone fracture', 'loss of height', 'stooped posture'],
        'Conjunctivitis': ['red eyes', 'itching', 'discharge', 'watery eyes', 'blurred vision'],
        'Sinusitis': ['facial pain', 'nasal congestion', 'headache', 'runny nose', 'cough'],
        'Influenza': ['fever', 'cough', 'sore throat', 'muscle aches', 'fatigue', 'headache'],
        'Appendicitis': ['abdominal pain', 'nausea', 'vomiting', 'fever', 'loss of appetite'],
        'Urinary Tract Infection': ['painful urination', 'frequent urination', 'abdominal pain', 'fever', 'cloudy urine'],
        'Hypothyroidism': ['fatigue', 'weight gain', 'cold sensitivity', 'constipation', 'dry skin'],
        'Hyperthyroidism': ['weight loss', 'rapid heartbeat', 'anxiety', 'sweating', 'tremors'],
        'Anemia': ['fatigue', 'weakness', 'pale skin', 'dizziness', 'shortness of breath']
    }
    
    # Generate training data with variations
    training_data = []
    
    for disease, symptoms in diseases_symptoms.items():
        # Generate multiple examples per disease
        for _ in range(50):  # 50 examples per disease
            # Randomly select 2-5 symptoms
            num_symptoms = np.random.randint(2, min(6, len(symptoms) + 1))
            selected_symptoms = np.random.choice(symptoms, num_symptoms, replace=False)
            
            # Create text representation
            symptom_text = ' '.join(selected_symptoms)
            training_data.append({
                'symptoms': symptom_text,
                'disease': disease
            })
    
    df = pd.DataFrame(training_data)
    print(f"Dataset created: {len(df)} samples across {len(diseases_symptoms)} diseases")
    return df

def train_disease_prediction_model(df):
    """
    Train Random Forest model for disease prediction
    """
    print("\n" + "="*60)
    print("TRAINING DISEASE PREDICTION MODEL")
    print("="*60)
    
    # Prepare features and labels
    X = df['symptoms']
    y = df['disease']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # TF-IDF Vectorization
    print("\nVectorizing text data...")
    vectorizer = TfidfVectorizer(max_features=200, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train Random Forest model
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_vec, y_train)
    
    # Evaluate model
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nAccuracy: {accuracy*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Save models
    print("\nSaving trained models...")
    with open('disease_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("✓ Models saved successfully!")
    return model, vectorizer, accuracy

# ============================================================================
# PART 2: EMAIL PROCESSING AND NLP
# ============================================================================

class MedicalEmailProcessor:
    """
    Main class for processing medical emails
    """
    
    SPECIALTY_MAPPINGS = {
        'Cardiology': ['chest pain', 'palpitation', 'shortness of breath', 'heart', 'cardiac', 'angina', 'irregular heartbeat'],
        'Neurology': ['headache', 'dizziness', 'seizure', 'numbness', 'migraine', 'blurred vision', 'vertigo', 'spinning'],
        'Dermatology': ['rash', 'skin lesion', 'itching', 'acne', 'eczema', 'psoriasis', 'dry skin', 'red patches'],
        'Gastroenterology': ['abdominal pain', 'nausea', 'diarrhea', 'vomiting', 'stomach', 'digestive', 'bloating', 'indigestion'],
        'Orthopedics': ['joint pain', 'fracture', 'back pain', 'bone', 'sprain', 'arthritis', 'stiffness', 'swelling'],
        'Pulmonology': ['cough', 'breathing difficulty', 'wheezing', 'asthma', 'lung', 'respiratory', 'chest tightness'],
        'Endocrinology': ['diabetes', 'thyroid', 'weight loss', 'weight gain', 'fatigue', 'frequent urination'],
        'Urology': ['urination', 'bladder', 'kidney', 'painful urination'],
        'Ophthalmology': ['eye', 'vision', 'blurred vision', 'red eyes', 'conjunctivitis']
    }
    
    DOCTORS = {
        'Cardiology': ['Dr. Rajesh Kumar', 'Dr. Anita Desai', 'Dr. Vikram Singh'],
        'Neurology': ['Dr. Rajesh Menon', 'Dr. Anjali Verma', 'Dr. Pradeep Shah'],
        'Dermatology': ['Dr. Meera Patel', 'Dr. Suresh Reddy', 'Dr. Kavita Sharma'],
        'Gastroenterology': ['Dr. Arun Gupta', 'Dr. Kavita Nair', 'Dr. Sanjay Joshi'],
        'Orthopedics': ['Dr. Ramesh Iyer', 'Dr. Pooja Sharma', 'Dr. Deepak Rao'],
        'Pulmonology': ['Dr. Deepak Rao', 'Dr. Sunita Joshi', 'Dr. Amit Patel'],
        'Endocrinology': ['Dr. Priya Nair', 'Dr. Rakesh Malhotra'],
        'Urology': ['Dr. Ashok Reddy', 'Dr. Sunita Desai'],
        'Ophthalmology': ['Dr. Kiran Kumar', 'Dr. Neha Gupta'],
        'General Medicine': ['Dr. Available Physician', 'Dr. General Practitioner']
    }
    
    PRIORITY_KEYWORDS = {
        'High': ['severe', 'urgent', 'emergency', 'chest pain', 'breathing difficulty', 'fever', 'bleeding'],
        'Medium': ['moderate', 'persistent', 'recurrent', 'uncomfortable'],
        'Low': ['mild', 'occasional', 'minor']
    }
    
    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer
    
    def extract_patient_name(self, text):
        """Extract patient name using regex patterns"""
        patterns = [
            r"my name is ([A-Z][a-z]+ [A-Z][a-z]+)",
            r"I'?m ([A-Z][a-z]+ [A-Z][a-z]+)",
            r"([A-Z][a-z]+ [A-Z][a-z]+),?\s*\d",
            r"patient:?\s*([A-Z][a-z]+ [A-Z][a-z]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return "Unknown Patient"
    
    def extract_symptoms(self, text):
        """Extract symptoms from text"""
        text_lower = text.lower()
        found_symptoms = []
        
        # Get all unique symptoms from specialty mappings
        all_symptoms = set()
        for symptoms in self.SPECIALTY_MAPPINGS.values():
            all_symptoms.update(symptoms)
        
        # Find symptoms in text
        for symptom in all_symptoms:
            if symptom in text_lower:
                found_symptoms.append(symptom)
        
        return list(set(found_symptoms))
    
    def predict_disease(self, symptoms):
        """Predict disease using trained ML model"""
        if not symptoms:
            return "General Consultation", 0.0
        
        symptom_text = ' '.join(symptoms)
        symptom_vec = self.vectorizer.transform([symptom_text])
        
        # Get prediction and probability
        prediction = self.model.predict(symptom_vec)[0]
        probabilities = self.model.predict_proba(symptom_vec)[0]
        confidence = max(probabilities) * 100
        
        return prediction, confidence
    
    def recommend_department(self, symptoms, predicted_disease):
        """Recommend department based on symptoms and disease"""
        dept_scores = {}
        
        for dept, keywords in self.SPECIALTY_MAPPINGS.items():
            score = sum(1 for s in symptoms if any(k in s for k in keywords))
            dept_scores[dept] = score
        
        if dept_scores and max(dept_scores.values()) > 0:
            return max(dept_scores, key=dept_scores.get)
        return 'General Medicine'
    
    def assign_priority(self, text, symptoms):
        """Assign priority based on keywords and symptoms"""
        text_lower = text.lower()
        
        for priority, keywords in self.PRIORITY_KEYWORDS.items():
            if any(keyword in text_lower for keyword in keywords):
                return priority
        
        # Default based on number of symptoms
        if len(symptoms) >= 4:
            return 'High'
        elif len(symptoms) >= 2:
            return 'Medium'
        return 'Low'
    
    def process_email(self, subject, body):
        """Process a single email and extract all information"""
        full_text = f"{subject} {body}"
        
        # Extract information
        patient_name = self.extract_patient_name(full_text)
        symptoms = self.extract_symptoms(full_text)
        disease, confidence = self.predict_disease(symptoms)
        department = self.recommend_department(symptoms, disease)
        priority = self.assign_priority(full_text, symptoms)
        
        # Get doctor recommendation
        doctors = self.DOCTORS.get(department, ['Dr. Available Physician'])
        recommended_doctor = np.random.choice(doctors)
        
        return {
            'Patient Name': patient_name,
            'Email Summary': subject,
            'Symptoms': ', '.join(symptoms) if symptoms else 'No specific symptoms detected',
            'Predicted Condition': disease,
            'Confidence': f"{confidence:.1f}%",
            'Department': department,
            'Recommended Doctor': recommended_doctor,
            'Priority': priority,
            'Date': datetime.now().strftime('%Y-%m-%d'),
            'Full Email': body[:200] + '...' if len(body) > 200 else body
        }

# ============================================================================
# PART 3: STREAMLIT DASHBOARD
# ============================================================================

def create_streamlit_dashboard():
    """
    Create interactive Streamlit dashboard
    """
    st.set_page_config(page_title="Medical Email Triage System", layout="wide", page_icon="🏥")
    
    # Custom CSS
    st.markdown("""
        <style>
        .big-font {font-size:30px !important; font-weight:bold;}
        .metric-card {background-color:#f0f2f6; padding:20px; border-radius:10px; margin:10px 0;}
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("<p class='big-font'>🏥 Medical Email Triage System</p>", unsafe_allow_html=True)
    st.markdown("**Automated Patient Information Extraction & Routing with ML**")
    st.markdown("---")
    
    # Load or train model
    try:
        with open('disease_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        st.success("✓ Trained model loaded successfully!")
    except FileNotFoundError:
        st.warning("⚠️ No trained model found. Training new model...")
        with st.spinner("Training model... This will take a moment..."):
            df = download_and_prepare_dataset()
            model, vectorizer, accuracy = train_disease_prediction_model(df)
            st.success(f"✓ Model trained successfully! Accuracy: {accuracy*100:.2f}%")
    
    # Initialize processor
    processor = MedicalEmailProcessor(model, vectorizer)
    
    # Sidebar
    st.sidebar.header("📋 Navigation")
    page = st.sidebar.radio("Select Page", ["Email Processing", "Model Information", "Statistics"])
    
    if page == "Email Processing":
        show_email_processing_page(processor)
    elif page == "Model Information":
        show_model_info_page(model, vectorizer)
    else:
        show_statistics_page()

def show_email_processing_page(processor):
    """Email processing page"""
    st.header("📧 Email Processing")
    
    # Sample emails
    sample_emails = [
        {
            'subject': 'Medical Consultation Request',
            'body': 'My name is Priya Sharma. I have been experiencing severe headaches for the past week, along with blurred vision and dizziness. Please help.'
        },
        {
            'subject': 'Urgent - Chest Pain',
            'body': "I'm Rajesh Kumar, 45 years old. I've had chest pain and shortness of breath since yesterday. Also experiencing palpitations and sweating."
        },
        {
            'subject': 'Skin Issue',
            'body': 'Hello, my name is Anita Patel. I have developed a red itchy rash on my arms and legs with dry skin patches that won\'t go away. It\'s been 3 days.'
        },
        {
            'subject': 'Stomach Problems',
            'body': "I'm Vikram Singh. Having severe abdominal pain with nausea and vomiting for 2 days. Can't keep food down. Also experiencing bloating."
        },
        {
            'subject': 'Breathing Issues',
            'body': 'My name is Deepa Reddy. I have persistent cough with wheezing and chest tightness. Breathing difficulty especially at night.'
        }
    ]
    
    # Option to use sample or custom email
    email_option = st.radio("Select Email Source:", ["Use Sample Emails", "Enter Custom Email"])
    
    if email_option == "Use Sample Emails":
        st.subheader("Sample Emails")
        for i, email in enumerate(sample_emails):
            with st.expander(f"Email {i+1}: {email['subject']}"):
                st.write(f"**Subject:** {email['subject']}")
                st.write(f"**Body:** {email['body']}")
        
        if st.button("🔄 Process All Sample Emails", type="primary"):
            with st.spinner("Processing emails..."):
                results = []
                for email in sample_emails:
                    result = processor.process_email(email['subject'], email['body'])
                    results.append(result)
                
                df_results = pd.DataFrame(results)
                st.session_state['results'] = df_results
                st.success(f"✓ Processed {len(results)} emails successfully!")
    
    else:
        st.subheader("Enter Custom Email")
        subject = st.text_input("Email Subject:")
        body = st.text_area("Email Body:", height=150)
        
        if st.button("🔄 Process Email", type="primary"):
            if subject and body:
                with st.spinner("Processing..."):
                    result = processor.process_email(subject, body)
                    df_results = pd.DataFrame([result])
                    st.session_state['results'] = df_results
                    st.success("✓ Email processed successfully!")
            else:
                st.error("Please enter both subject and body.")
    
    # Display results
    if 'results' in st.session_state:
        st.markdown("---")
        st.header("📊 Triage Results")
        
        df = st.session_state['results']
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Processed", len(df))
        with col2:
            st.metric("High Priority", len(df[df['Priority'] == 'High']))
        with col3:
            st.metric("Medium Priority", len(df[df['Priority'] == 'Medium']))
        with col4:
            st.metric("Low Priority", len(df[df['Priority'] == 'Low']))
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            dept_filter = st.multiselect("Filter by Department:", 
                                        options=df['Department'].unique(),
                                        default=df['Department'].unique())
        with col2:
            priority_filter = st.multiselect("Filter by Priority:",
                                            options=['High', 'Medium', 'Low'],
                                            default=['High', 'Medium', 'Low'])
        
        # Apply filters
        filtered_df = df[
            (df['Department'].isin(dept_filter)) & 
            (df['Priority'].isin(priority_filter))
        ]
        
        # Display table
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"medical_triage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def show_model_info_page(model, vectorizer):
    """Model information page"""
    st.header("🤖 Model Information")
    
    st.subheader("Model Architecture")
    st.write(f"**Algorithm:** Random Forest Classifier")
    st.write(f"**Number of Estimators:** {model.n_estimators}")
    st.write(f"**Feature Extraction:** TF-IDF Vectorization")
    st.write(f"**Vocabulary Size:** {len(vectorizer.vocabulary_)}")
    st.write(f"**Number of Classes:** {len(model.classes_)}")
    
    st.subheader("Diseases the Model Can Predict")
    diseases_df = pd.DataFrame({
        'Disease': model.classes_,
        'Index': range(len(model.classes_))
    })
    st.dataframe(diseases_df, use_container_width=True, hide_index=True)
    
    st.subheader("Top Features")
    feature_names = vectorizer.get_feature_names_out()
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        top_indices = np.argsort(importances)[-20:][::-1]
        top_features = [(feature_names[i], importances[i]) for i in top_indices]
        
        top_features_df = pd.DataFrame(top_features, columns=['Feature', 'Importance'])
        st.dataframe(top_features_df, use_container_width=True, hide_index=True)

def show_statistics_page():
    """Statistics page"""
    st.header("📈 Statistics Dashboard")
    
    if 'results' not in st.session_state:
        st.info("Process some emails first to see statistics.")
        return
    
    df = st.session_state['results']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Priority Distribution")
        priority_counts = df['Priority'].value_counts()
        st.bar_chart(priority_counts)
    
    with col2:
        st.subheader("Department Distribution")
        dept_counts = df['Department'].value_counts()
        st.bar_chart(dept_counts)
    
    st.subheader("Most Common Conditions")
    condition_counts = df['Predicted Condition'].value_counts().head(10)
    st.bar_chart(condition_counts)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Check if running in Streamlit
    try:
        create_streamlit_dashboard()
    except:
        # Command line execution - Train model
        print("="*60)
        print("MEDICAL EMAIL TRIAGE SYSTEM - MODEL TRAINING")
        print("="*60)
        
        # Download and prepare dataset
        df = download_and_prepare_dataset()
        
        # Train model
        model, vectorizer, accuracy = train_disease_prediction_model(df)
        
        print("\n" + "="*60)
        print("TESTING THE SYSTEM")
        print("="*60)
        
        # Test the system
        processor = MedicalEmailProcessor(model, vectorizer)
        
        test_email = {
            'subject': 'Medical Consultation Request',
            'body': 'My name is Priya Sharma. I have been experiencing severe headaches for the past week, along with blurred vision and dizziness.'
        }
        
        result = processor.process_email(test_email['subject'], test_email['body'])
        
        print("\nTest Email Processing Result:")
        print("-" * 60)
        for key, value in result.items():
            print(f"{key}: {value}")
        
        print("\n" + "="*60)
        print("✓ MODEL TRAINING COMPLETE!")
        print("✓ Run with: streamlit run <filename>.py")
        print("="*60)

'''streamlit run medical_triage_system.py'''

import os
os.system("pip install streamlit")

import os
os.system("pip install pandas")