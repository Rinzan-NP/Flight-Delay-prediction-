# Flight Delay Prediction System

A comprehensive machine learning application for predicting flight delays using multiple algorithms and interactive visualizations.

## 📁 Project Structure

The project contains the following files:

```
├── app.py                  # Main Streamlit application with all functionality
├── requirements.txt        # Python dependencies
├── finished.csv           # Flight delay dataset
├── notebooke7394a32f4 (2).ipynb  # Jupyter notebook (development/testing)
├── __pycache__/           # Python cache files (auto-generated)
└── README.md              # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📊 Features

### 🏠 Home Page
- **Dynamic Statistics**: Real-time calculation of total flights, delay rates, model accuracy, and processing time
- **Dataset Overview**: Comprehensive information about the loaded dataset
- **Quick Insights**: Key statistics and data quality metrics

### 🔮 Prediction Page
- **Interactive Form**: User-friendly interface for entering flight details
- **Real-time Prediction**: Instant delay predictions using trained models
- **Confidence Scores**: Detailed prediction confidence and probability metrics
- **Multiple Models**: Support for Logistic Regression, Decision Tree, and Random Forest

### 📊 Analysis Page
- **Delay Patterns**: Visual analysis of delay trends and patterns
- **Airline Performance**: Comparative analysis of airline reliability
- **Route Analysis**: Identification of most/least reliable flight routes
- **Time-based Insights**: Monthly, daily, and hourly delay patterns

### 🤖 Model Performance Page
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1 Score, R² Score, ROC AUC, MSE, MAE
- **Model Comparison**: Side-by-side comparison of all trained models
- **Feature Importance**: Analysis of which factors most influence delays
- **Performance Visualization**: Interactive charts and detailed metrics

### 📈 Visualizations Page
- **Interactive Charts**: Plotly-powered visualizations with zoom, pan, and hover features
- **Multi-tab Interface**: Organized visualization categories
- **Export Capabilities**: Save charts and data for further analysis

## 🧠 Machine Learning Models

The system includes three state-of-the-art machine learning models:

### 1. Logistic Regression
- **Use Case**: Linear relationship analysis
- **Advantages**: Fast training, interpretable results
- **Best For**: Baseline predictions and feature importance analysis

### 2. Decision Tree
- **Use Case**: Non-linear pattern recognition
- **Advantages**: Easy to interpret, handles categorical data well
- **Best For**: Understanding decision boundaries and rules

### 3. Random Forest
- **Use Case**: Ensemble learning for improved accuracy
- **Advantages**: High accuracy, robust to overfitting
- **Best For**: Production predictions and feature importance ranking

## 📈 Evaluation Metrics

The system provides comprehensive evaluation metrics:

- **Accuracy**: Overall prediction correctness
- **Precision**: Correct positive predictions ratio
- **Recall**: Actual positives correctly identified
- **F1 Score**: Harmonic mean of precision and recall
- **R² Score**: Coefficient of determination
- **ROC AUC**: Area under the receiver operating characteristic curve
- **MSE**: Mean squared error
- **MAE**: Mean absolute error

## 🛠️ Technical Architecture

### Main Application (`app.py`)
The application is built as a single comprehensive Streamlit file containing:

- **Data Loading**: Efficient CSV loading with sampling for large datasets (50,000 records max)
- **Feature Preprocessing**: Date conversion, scaling, and encoding
- **Model Training**: Automated training of multiple ML models (Logistic Regression, Decision Tree, Random Forest)
- **Prediction Engine**: Real-time prediction with confidence scoring
- **Evaluation**: Comprehensive model performance assessment
- **UI Components**: Modular page system with navigation
- **Styling**: Custom CSS for blue and white theme
- **Visualizations**: Interactive charts and graphs using Plotly
- **User Experience**: Responsive design and loading indicators

## 📊 Dataset Requirements

The application expects a CSV file named `finished.csv` with the following structure:

### Required Columns:
- `is_delay`: Binary target variable (0 = on-time, 1 = delayed)
- `ArrDelay`: Arrival delay in minutes
- `DepDelay`: Departure delay in minutes

### Optional Columns:
- `FlightDate`: Flight date
- `Year`, `Quarter`, `Month`, `DayofMonth`, `DayOfWeek`: Temporal features
- `DepTime`, `ArrTime`: Departure and arrival times
- `Airline`: Airline code
- `Origin`, `Dest`: Origin and destination airports
- `OriginState`, `DestState`: Origin and destination states
- `Distance`: Flight distance in miles
- `AirTime`: Flight duration in minutes
- `WeatherDelay`, `CarrierDelay`: Delay causes

## 🎨 User Interface

### Theme
- **Color Scheme**: Blue and white professional theme
- **Layout**: Wide layout optimized for data visualization
- **Responsive**: Adapts to different screen sizes

### Navigation
- **Sidebar**: Easy navigation between pages
- **Breadcrumbs**: Clear page hierarchy
- **Loading States**: User-friendly loading indicators

## 🔧 Configuration

### Performance Optimization
- **Data Sampling**: Automatic sampling for large datasets (50,000 records max)
- **Model Parameters**: Optimized for speed vs. accuracy balance
- **Caching**: Streamlit caching for improved performance

### Customization
- **Model Parameters**: Easily adjustable in the logic module
- **UI Styling**: Customizable CSS in the UI module
- **Feature Selection**: Configurable feature engineering pipeline

## 🐛 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **FileNotFoundError**: Ensure `finished.csv` is in the project directory

3. **Memory Issues**: The application automatically samples large datasets

4. **Slow Loading**: Reduce sample size in `load_and_preprocess_data()`

### Performance Tips

- Use smaller sample sizes for faster development
- Adjust model parameters for speed vs. accuracy trade-offs
- Clear Streamlit cache if experiencing issues: `streamlit cache clear`

## 📝 Development

### Adding New Features

1. **Main Application**: Add new ML models or preprocessing steps directly in `app.py`
2. **UI Components**: Add new pages or visualization components in the main file
3. **Integration**: Update function calls and imports as needed

### Code Structure

- **Functions**: Well-documented functions with clear parameters and returns
- **Comments**: Comprehensive inline documentation for each function
- **Error Handling**: Robust error handling with user-friendly messages
- **Modularity**: Organized into logical function groups within the single file


