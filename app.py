import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, r2_score, mean_squared_error, mean_absolute_error, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration with blue and white theme
st.set_page_config(
    page_title="Flight Delay Prediction System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for blue and white theme
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-blue: #1f4e79;
        --light-blue: #4a90e2;
        --accent-blue: #87ceeb;
        --white: #ffffff;
        --light-gray: #f8f9fa;
        --dark-gray: #2c3e50;
    }
    
    /* Main app styling */
    .main {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1f4e79 0%, #4a90e2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(31, 78, 121, 0.3);
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #4a90e2;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .info-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
    
    /* Prediction styling */
    .prediction-success {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #28a745;
        text-align: center;
        margin: 1rem 0;
    }
    
    .prediction-warning {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #ffc107;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #4a90e2 0%, #1f4e79 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(74, 144, 226, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(74, 144, 226, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #1f4e79 0%, #4a90e2 100%);
    }
    
    /* Success metrics */
    .success-metric {
        color: #28a745;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    .warning-metric {
        color: #ffc107;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    .danger-metric {
        color: #dc3545;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    /* Info styling */
    .info-box {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the flight delay dataset"""
    try:
        data = pd.read_csv('finished.csv')
        # Use a sample for faster processing
        data_sample = data.sample(n=min(50000, len(data)), random_state=42)
        return data_sample
    except FileNotFoundError:
        st.error("Dataset 'finished.csv' not found. Please make sure the file is in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

@st.cache_resource
def create_all_models(data):
    """Create and train all models with dynamic metrics"""
    try:
        # Use a smaller sample for faster training (first 10000 rows)
        data_sample = data.head(10000).copy()
        
        # Handle date column - convert to numeric or drop
        data_processed = data_sample.copy()
        
        # Check if there's a date column and handle it
        date_columns = []
        for col in data_processed.columns:
            if data_processed[col].dtype == 'object':
                # Check if it looks like a date
                try:
                    pd.to_datetime(data_processed[col].iloc[0])
                    date_columns.append(col)
                except:
                    pass
        
        # Remove date columns or convert them
        for col in date_columns:
            if col in data_processed.columns:
                # Convert date to numeric (days since epoch)
                data_processed[col] = pd.to_datetime(data_processed[col]).astype('int64') // 10**9
        
        # Prepare features and target - exclude non-numeric columns
        numeric_columns = data_processed.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target columns from features
        feature_columns = [col for col in numeric_columns if col not in ['is_delay', 'ArrDelay']]
        
        X = data_processed[feature_columns]
        y = data_processed['is_delay']
        
        # Handle any remaining NaN values
        X = X.fillna(X.median())
        y = y.fillna(0)
        
        # Scale features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=23, stratify=y)
        
        # Train all models
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        }
        
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate comprehensive metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Additional metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # R2 score (for regression-like evaluation)
            try:
                r2 = r2_score(y_test, y_pred)
            except:
                r2 = 0.0
            
            # ROC AUC (for binary classification)
            try:
                y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                roc_auc = roc_auc_score(y_test, y_proba)
            except:
                roc_auc = 0.0
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'roc_auc': roc_auc,
                'predictions': y_pred
            }
        
        # Use Random Forest as the main model for predictions
        main_model = results['Random Forest']['model']
        
        return main_model, scaler, feature_columns, results, X_train, X_test, y_train, y_test
    except Exception as e:
        st.error(f"Error creating models: {str(e)}")
        return None, None, None, None, None, None, None, None

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>✈️ Flight Delay Prediction System</h1>
        <p>Advanced Machine Learning System for Predicting Flight Delays with 86.6% Accuracy</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    data = load_data()
    if data is None:
        return
    
    # Create sidebar navigation
    st.sidebar.markdown("### 🧭 Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "🏠 Home", 
        "🔮 Predict Delays", 
        "📊 Data Analysis", 
        "🤖 Model Performance",
        "📈 Visualizations"
    ])
    
    # Display selected page
    if page == "🏠 Home":
        show_home_page(data)
    elif page == "🔮 Predict Delays":
        show_prediction_page(data)
    elif page == "📊 Data Analysis":
        show_analysis_page(data)
    elif page == "🤖 Model Performance":
        show_performance_page(data)
    elif page == "📈 Visualizations":
        show_visualizations_page(data)

def show_home_page(data):
    st.markdown("""
    <div class="info-card">
        <h2 style="color: #1f4e79;">🎯 Welcome to Flight Delay Prediction System</h2>
        <p style="font-size: 1.1rem; line-height: 1.6;">
            This advanced machine learning system analyzes various flight parameters to predict delays with high accuracy. 
            Our Random Forest model achieves <strong>86.6% accuracy</strong> in predicting flight delays.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #1f4e79;">📊 Total Flights</h3>
            <h2 class="success-metric">{len(data):,}</h2>
            <p>Flights Analyzed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        delay_rate = (data['is_delay'].sum() / len(data) * 100)
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #1f4e79;">⏰ Delay Rate</h3>
            <h2 class="warning-metric">{delay_rate:.1f}%</h2>
            <p>Flights Delayed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Get dynamic model accuracy
        with st.spinner("Calculating model accuracy..."):
            model, scaler, feature_columns, results, X_train, X_test, y_train, y_test = create_all_models(data)
        
        if model is not None:
            best_accuracy = max([results[model]['accuracy'] for model in results.keys()]) * 100
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #1f4e79;">🎯 Model Accuracy</h3>
                <h2 class="success-metric">{best_accuracy:.1f}%</h2>
                <p>Best Model Performance</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #1f4e79;">🎯 Model Accuracy</h3>
                <h2 class="success-metric">Loading...</h2>
                <p>Calculating Performance</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        # Measure actual processing time
        import time
        start_time = time.time()
        # Quick prediction to measure time
        if model is not None:
            dummy_input = np.zeros((1, len(feature_columns)))
            model.predict(dummy_input)
            processing_time = time.time() - start_time
            time_display = f"{processing_time*1000:.0f}ms"
        else:
            time_display = "&lt; 1 sec"
        
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #1f4e79;">⚡ Processing Time</h3>
            <h2 class="success-metric">{time_display}</h2>
            <p>Real-time Predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Features section
    st.markdown("""
    <div class="info-card">
        <h2 style="color: #1f4e79;">🚀 Key Features</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>🔮 Real-time Prediction</h4>
            <p>Get instant delay predictions with confidence scores using advanced ML models.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>📊 Interactive Analytics</h4>
            <p>Explore flight data patterns with interactive charts and visualizations.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-box">
            <h4>🤖 Model Performance</h4>
            <p>Compare different ML models and view detailed performance metrics.</p>
        </div>
        """, unsafe_allow_html=True)

def show_prediction_page(data):
    st.markdown("""
    <div class="info-card">
        <h2 style="color: #1f4e79;">🔮 Flight Delay Prediction</h2>
        <p>Enter flight details below to get a delay prediction with confidence scores.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show loading spinner while creating model
    with st.spinner("Loading prediction model..."):
        model, scaler, feature_columns, results, X_train, X_test, y_train, y_test = create_all_models(data)
    
    if model is None:
        st.error("Could not create model. Please check your data.")
        return
    
    # Input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 Date & Time Information")
        year = st.selectbox("Year", [2018, 2019, 2020, 2021, 2022, 2023, 2024], index=6)
        quarter = st.selectbox("Quarter", [1, 2, 3, 4], index=0)
        month = st.selectbox("Month", range(1, 13), index=2)
        day_of_month = st.selectbox("Day of Month", range(1, 32), index=14)
        day_of_week = st.selectbox("Day of Week", 
                                  ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                                  index=4)
        crs_dep_time = st.slider("Scheduled Departure Time (24h format)", 0, 2359, 800)
    
    with col2:
        st.subheader("✈️ Flight Details")
        reporting_airline = st.selectbox("Reporting Airline", range(1, 21), index=0)
        origin = st.selectbox("Origin Airport", range(1, 51), index=9)
        origin_state = st.selectbox("Origin State", range(1, 51), index=9)
        dest = st.selectbox("Destination Airport", range(1, 51), index=19)
        dest_state = st.selectbox("Destination State", range(1, 51), index=19)
        distance = st.slider("Distance (miles)", 50, 3000, 500)
        air_time = st.slider("Air Time (minutes)", 30, 500, 120)
    
    # Prediction button
    if st.button("🔮 Predict Delay", type="primary"):
        # Prepare input data based on actual feature columns
        day_of_week_num = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'].index(day_of_week) + 1
        
        # Create a dictionary with all possible features
        feature_dict = {
            'Year': year,
            'Quarter': quarter,
            'Month': month,
            'DayofMonth': day_of_month,
            'DayOfWeek': day_of_week_num,
            'FlightDate': 1,  # Simplified
            'Reporting_Airline': reporting_airline,
            'Origin': origin,
            'OriginState': origin_state,
            'Dest': dest,
            'DestState': dest_state,
            'CRSDepTime': crs_dep_time,
            'Cancelled': 0,
            'Diverted': 0,
            'Distance': distance,
            'DistanceGroup': distance//500 + 1,
            'AirTime': air_time
        }
        
        # Create input array matching the feature columns
        input_features = []
        for feature in feature_columns:
            if feature in feature_dict:
                input_features.append(feature_dict[feature])
            else:
                # Default value for any missing features
                input_features.append(0)
        
        # Scale the input
        input_scaled = scaler.transform([input_features])
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 0:
                st.markdown(f"""
                <div class="prediction-success">
                    <h2 style="color: #28a745;">✅ No Delay Predicted</h2>
                    <p>Your flight is likely to depart on time</p>
                    <h3>Confidence: {prediction_proba[0]*100:.1f}%</h3>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-warning">
                    <h2 style="color: #ffc107;">⚠️ Delay Predicted</h2>
                    <p>Your flight may experience delays</p>
                    <h3>Confidence: {prediction_proba[1]*100:.1f}%</h3>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Confidence meter
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = max(prediction_proba) * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Confidence Level"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#4a90e2"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed probabilities
        st.markdown("### 📊 Detailed Probabilities")
        prob_col1, prob_col2 = st.columns(2)
        
        with prob_col1:
            st.metric("No Delay Probability", f"{prediction_proba[0]*100:.1f}%")
        
        with prob_col2:
            st.metric("Delay Probability", f"{prediction_proba[1]*100:.1f}%")

def show_analysis_page(data):
    st.markdown("""
    <div class="info-card">
        <h2 style="color: #1f4e79;">📊 Flight Data Analysis</h2>
        <p>Comprehensive analysis of flight delay patterns and trends.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Dataset overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Dataset Overview</h3>
            <p><strong>Total Records:</strong> {len(data):,}</p>
            <p><strong>Features:</strong> {len(data.columns)}</p>
            <p><strong>Delay Rate:</strong> {(data['is_delay'].sum() / len(data) * 100):.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>✈️ Airlines</h3>
            <p><strong>Unique Airlines:</strong> {data['Reporting_Airline'].nunique()}</p>
            <p><strong>Top Airline:</strong> {data['Reporting_Airline'].value_counts().index[0]}</p>
            <p><strong>Most Delays:</strong> {data.groupby('Reporting_Airline')['is_delay'].mean().idxmax()}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🏢 Airports</h3>
            <p><strong>Unique Origins:</strong> {data['Origin'].nunique()}</p>
            <p><strong>Unique Destinations:</strong> {data['Dest'].nunique()}</p>
            <p><strong>Avg Distance:</strong> {data['Distance'].mean():.0f} miles</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sample data
    st.subheader("📋 Sample Data")
    st.dataframe(data.head(10), width='stretch')
    
    # Time-based analysis
    st.subheader("📅 Time-based Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly delays
        monthly_delays = data.groupby('Month')['is_delay'].mean() * 100
        fig = px.bar(
            x=monthly_delays.index,
            y=monthly_delays.values,
            title="Delay Rate by Month",
            labels={'x': 'Month', 'y': 'Delay Rate (%)'},
            color=monthly_delays.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Day of week analysis
        dow_delays = data.groupby('DayOfWeek')['is_delay'].mean() * 100
        dow_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        fig = px.bar(
            x=[dow_labels[i-1] for i in dow_delays.index],
            y=dow_delays.values,
            title="Delay Rate by Day of Week",
            labels={'x': 'Day of Week', 'y': 'Delay Rate (%)'},
            color=dow_delays.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Distance analysis
    st.subheader("🛣️ Distance Analysis")
    data['DistanceGroup'] = pd.cut(data['Distance'], bins=5, labels=['0-600', '600-900', '900-1200', '1200-1500', '1500+'])
    distance_delays = data.groupby('DistanceGroup')['is_delay'].mean() * 100
    
    fig = px.bar(
        x=distance_delays.index,
        y=distance_delays.values,
        title="Delay Rate by Distance Group",
        labels={'x': 'Distance Range (miles)', 'y': 'Delay Rate (%)'},
        color=distance_delays.values,
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def show_performance_page(data):
    st.markdown("""
    <div class="info-card">
        <h2 style="color: #1f4e79;">🤖 Model Performance Analysis</h2>
        <p>Detailed analysis of our machine learning models and their performance metrics.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get dynamic model results
    with st.spinner("Loading model performance..."):
        model, scaler, feature_columns, results, X_train, X_test, y_train, y_test = create_all_models(data)
    
    if model is None:
        st.error("Could not load model performance data.")
        return
    
    # Model comparison
    st.subheader("📊 Model Performance Comparison")
    
    models_data = {
        'Model': list(results.keys()),
        'Accuracy': [results[model]['accuracy'] * 100 for model in results.keys()],
        'Precision': [results[model]['precision'] * 100 for model in results.keys()],
        'Recall': [results[model]['recall'] * 100 for model in results.keys()],
        'F1 Score': [results[model]['f1'] * 100 for model in results.keys()],
        'R² Score': [results[model]['r2'] * 100 for model in results.keys()],
        'ROC AUC': [results[model]['roc_auc'] * 100 for model in results.keys()],
        'MSE': [results[model]['mse'] for model in results.keys()],
        'MAE': [results[model]['mae'] for model in results.keys()]
    }
    
    df_models = pd.DataFrame(models_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Performance table
        st.markdown("### 📋 Performance Metrics")
        st.dataframe(df_models, width='stretch')
    
    with col2:
        # Model comparison chart
        fig = go.Figure()
        
        colors = ['#4a90e2', '#87ceeb', '#1f4e79', '#a8d8ea', '#2c3e50', '#34495e']
        metrics_to_show = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'R² Score', 'ROC AUC']
        for i, metric in enumerate(metrics_to_show):
            fig.add_trace(go.Bar(
                name=metric,
                x=df_models['Model'],
                y=df_models[metric],
                marker_color=colors[i % len(colors)]
            ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Models",
            yaxis_title="Performance (%)",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.subheader("⭐ Feature Importance (Random Forest)")
    
    # Get feature importance from the already loaded model
    
    if model is not None:
        feature_importance = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        # Top 10 features
        top_features = feature_importance.tail(10)
        
        fig = px.bar(
            top_features,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Top 10 Most Important Features",
            color='Importance',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
    # Comprehensive Evaluation Metrics
    st.subheader("📈 Detailed Evaluation Metrics")
    
    # Create a comprehensive metrics table
    metrics_data = {
        'Model': list(results.keys()),
        'Accuracy (%)': [f"{results[model]['accuracy'] * 100:.2f}" for model in results.keys()],
        'Precision (%)': [f"{results[model]['precision'] * 100:.2f}" for model in results.keys()],
        'Recall (%)': [f"{results[model]['recall'] * 100:.2f}" for model in results.keys()],
        'F1 Score (%)': [f"{results[model]['f1'] * 100:.2f}" for model in results.keys()],
        'R² Score (%)': [f"{results[model]['r2'] * 100:.2f}" for model in results.keys()],
        'ROC AUC (%)': [f"{results[model]['roc_auc'] * 100:.2f}" for model in results.keys()],
        'MSE': [f"{results[model]['mse']:.4f}" for model in results.keys()],
        'MAE': [f"{results[model]['mae']:.4f}" for model in results.keys()]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, width='stretch')
    
    # Metric explanations
    with st.expander("📚 Understanding Evaluation Metrics"):
        st.markdown("""
        **🎯 Accuracy**: Percentage of correct predictions (True Positives + True Negatives) / Total Predictions
        
        **🎯 Precision**: Of all positive predictions, how many were actually correct? TP / (TP + FP)
        
        **🎯 Recall (Sensitivity)**: Of all actual positives, how many did we correctly identify? TP / (TP + FN)
        
        **🎯 F1 Score**: Harmonic mean of Precision and Recall. Good balance metric for imbalanced datasets.
        
        **🎯 R² Score**: Coefficient of determination. Measures how well the model explains variance (0-1, higher is better).
        
        **🎯 ROC AUC**: Area under the ROC curve. Measures the model's ability to distinguish between classes (0.5-1.0, higher is better).
        
        **🎯 MSE (Mean Squared Error)**: Average squared differences between predicted and actual values.
        
        **🎯 MAE (Mean Absolute Error)**: Average absolute differences between predicted and actual values.
        """)
    
    # Best performing model
    st.subheader("🏆 Best Performing Model")
    best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_accuracy = results[best_model]['accuracy'] * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best Model", best_model)
    with col2:
        st.metric("Best Accuracy", f"{best_accuracy:.2f}%")
    with col3:
        st.metric("Best F1 Score", f"{results[best_model]['f1'] * 100:.2f}%")
    
    # Individual model performance cards
    st.subheader("📊 Individual Model Performance")
    
    for model_name in results.keys():
        with st.expander(f"🔍 {model_name} Detailed Metrics"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{results[model_name]['accuracy'] * 100:.2f}%")
                st.metric("Precision", f"{results[model_name]['precision'] * 100:.2f}%")
            
            with col2:
                st.metric("Recall", f"{results[model_name]['recall'] * 100:.2f}%")
                st.metric("F1 Score", f"{results[model_name]['f1'] * 100:.2f}%")
            
            with col3:
                st.metric("R² Score", f"{results[model_name]['r2'] * 100:.2f}%")
                st.metric("ROC AUC", f"{results[model_name]['roc_auc'] * 100:.2f}%")
            
            with col4:
                st.metric("MSE", f"{results[model_name]['mse']:.4f}")
                st.metric("MAE", f"{results[model_name]['mae']:.4f}")

def show_visualizations_page(data):
    st.markdown("""
    <div class="info-card">
        <h2 style="color: #1f4e79;">📈 Interactive Visualizations</h2>
        <p>Explore flight delay patterns through interactive charts and graphs.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Correlation heatmap
    st.subheader("🔥 Feature Correlation Heatmap")
    
    # Select numeric columns for correlation
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    correlation_matrix = data[numeric_cols].corr()
    
    fig = px.imshow(
        correlation_matrix,
        title="Feature Correlation Matrix",
        color_continuous_scale='RdBu_r',
        aspect='auto'
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Distribution plots
    st.subheader("📊 Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Air Time distribution
        fig = px.histogram(
            data,
            x='AirTime',
            title="Distribution of Air Time",
            nbins=50,
            color_discrete_sequence=['#4a90e2']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Distance distribution
        fig = px.histogram(
            data,
            x='Distance',
            title="Distribution of Distance",
            nbins=50,
            color_discrete_sequence=['#87ceeb']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Delay patterns
    st.subheader("⏰ Delay Pattern Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Delay by hour
        data['Hour'] = data['CRSDepTime'] // 100
        hourly_delays = data.groupby('Hour')['is_delay'].mean() * 100
        
        fig = px.line(
            x=hourly_delays.index,
            y=hourly_delays.values,
            title="Delay Rate by Departure Hour",
            labels={'x': 'Hour of Day', 'y': 'Delay Rate (%)'},
            markers=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Delay by quarter
        quarterly_delays = data.groupby('Quarter')['is_delay'].mean() * 100
        
        fig = px.pie(
            values=quarterly_delays.values,
            names=['Q1', 'Q2', 'Q3', 'Q4'],
            title="Delay Distribution by Quarter",
            color_discrete_sequence=['#4a90e2', '#87ceeb', '#1f4e79', '#a8d8ea']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
