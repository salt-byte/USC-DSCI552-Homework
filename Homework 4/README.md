# DSCI 552 - Homework 4: Time Series Classification

**Student:** Yudie Deng  
**GitHub Username:** yudiedeng  
**USC ID:** 7162812062

## Project Overview

This project implements comprehensive time series classification using the AReM (Activity Recognition from Mobile sensors) dataset. The project covers:

1. **Feature Extraction**: Time-domain feature extraction with bootstrap confidence intervals
2. **Binary Classification**: Bending vs. other activities using logistic regression and L1-penalized regression
3. **Multiclass Classification**: All 7 activity types using multinomial logistic regression and Naive Bayes
4. **Feature Selection**: Comparison of p-value based selection vs. L1-penalty based selection
5. **Theoretical Analysis**: Mathematical derivations of QDA and Bayes classifiers
6. **Model Evaluation**: Cross-validation, confusion matrices, ROC curves, and performance metrics

## Dataset Description

The AReM dataset contains sensor data from mobile devices measuring different human activities:
- **Activities:** bending1, bending2, cycling, lying, sitting, standing, walking
- **Sensors:** 6 sensor readings per time step (avg_rss12, var_rss12, avg_rss13, var_rss13, avg_rss23, var_rss23)
- **Sampling:** 20 Hz frequency, 120 seconds duration per activity
- **Data Split:** 
  - Test: datasets 1 & 2 in bending1/bending2; datasets 1, 2, 3 in other folders
  - Train: remaining datasets

## Project Structure

```
Homework 4/
├── data/
│   ├── AReM_split/
│   │   ├── train/          # Training data (69 samples)
│   │   └── test/           # Test data (19 samples)
│   ├── bendingType.pdf     # Activity type documentation
│   └── sensorsPlacement.pdf # Sensor placement documentation
├── notebook/
│   └── Deng_Yudie_HW4.ipynb # Main analysis notebook
├── README.md               # This file
└── requirements.txt       # Python dependencies
```

## Features

### Time Domain Feature Extraction
The project extracts comprehensive time-domain features for each sensor series:
- **Minimum** (min): Captures lower bound of sensor readings
- **Maximum** (max): Captures upper bound of sensor readings  
- **Mean**: Central tendency of sensor readings
- **Median**: Robust central tendency measure
- **Standard Deviation**: Variability measure
- **First Quartile (Q1)**: 25th percentile
- **Third Quartile (Q3)**: 75th percentile

### Selected Features
For classification, three key features are selected:
- **min, mean, max** - These features capture amplitude envelope, central tendency, and extrema, providing complementary information for activity discrimination.

### Advanced Feature Engineering
- **Time Series Segmentation**: Breaking time series into multiple segments for enhanced feature extraction
- **Bootstrap Confidence Intervals**: Statistical analysis of feature stability
- **Feature Selection**: Using Recursive Feature Elimination (RFE) and p-value based selection
- **Cross-validation**: Proper model evaluation with stratified k-fold validation

## Machine Learning Methods

### Binary Classification
1. **Logistic Regression**: Standard logistic regression with class balancing
2. **L1-Penalized Logistic Regression**: Regularized logistic regression with automatic feature selection
3. **Feature Selection Comparison**: Analysis of p-value based vs. L1-penalty based feature selection

### Multiclass Classification
1. **L1-Penalized Multinomial Logistic Regression**: Extension to multiple classes
2. **Gaussian Naive Bayes**: Probabilistic classifier assuming Gaussian distributions
3. **Multinomial Naive Bayes**: Alternative probabilistic approach

### Model Evaluation
- **Confusion Matrices**: Detailed performance analysis
- **ROC Curves**: Binary classification performance
- **Cross-validation**: Robust performance estimation
- **Class Balancing**: Handling imbalanced datasets

## Installation

1. **Clone or download** this repository
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Notebook Structure

The notebook is organized into four main sections:

### 1. Time Series Classification - Part 1: Feature Creation/Extraction
- **1(a) Data**: Loading and preprocessing AReM dataset
- **1(b) Train/Test Split**: Folder-based data splitting rules
- **1(c) Feature Extraction**: 
  - Time-domain feature computation (min, max, mean, median, std, Q1, Q3)
  - Bootstrap confidence intervals for feature stability
  - Feature selection (min, mean, max)

### 2. Time Series Classification - Part 2: Binary and Multiclass Classification
- **2(a) Binary Classification Using Logistic Regression**:
  - Scatter plots for feature visualization
  - Time series segmentation (2 equal parts)
  - Model training with cross-validation
  - Confusion matrix and parameter analysis
  - Test set evaluation
  - Class imbalance handling
- **2(b) Binary Classification Using L1-penalized Logistic Regression**:
  - Class balancing techniques
  - Comparison with p-value based feature selection
- **2(c) Multi-class Classification**:
  - L1-penalized multinomial regression
  - Gaussian and Multinomial Naive Bayes
  - Method comparison and evaluation

### 3. ISLR 4.8.3: QDA Mathematical Derivation
- Step-by-step mathematical derivation of Quadratic Discriminant Analysis
- Bayes classifier principle and class-conditional densities
- Log-discriminant functions and decision boundaries

### 4. ISLR 4.8.7: Bayes' Theorem Application
- Practical application of Bayes' theorem
- Dividend prediction problem with numerical calculations

## Usage

1. **Run the Jupyter notebook:**
   ```bash
   jupyter notebook notebook/Deng_Yudie_HW4.ipynb
   ```

2. **Execute cells sequentially** to:
   - Load and preprocess the AReM dataset
   - Extract time-domain features with bootstrap analysis
   - Visualize feature distributions and separability
   - Implement binary classification (bending vs. other activities)
   - Compare different feature selection methods
   - Implement multiclass classification
   - Analyze theoretical aspects of QDA and Bayes classifiers

## Technical Implementation Details

### Data Processing Pipeline
- **Robust CSV Loading**: Handles multiple delimiter formats (comma, space-separated)
- **Data Validation**: Checks for expected column count (6 sensor readings)
- **Error Handling**: Comprehensive error reporting for malformed files
- **Data Cleaning**: Automatic conversion to numeric format with NaN handling

### Feature Engineering Pipeline
- **Time-Domain Features**: 7 statistical features per sensor (min, max, mean, median, std, Q1, Q3)
- **Bootstrap Analysis**: 2000 resamples with 90% confidence intervals
- **Time Series Segmentation**: Breaking time series into equal-length segments
- **Feature Selection**: RFE (Recursive Feature Elimination) with cross-validation

### Machine Learning Pipeline
- **Cross-Validation**: 5-fold stratified validation to preserve class proportions
- **Feature Scaling**: StandardScaler for normalization
- **Class Balancing**: Automatic class weight balancing for imbalanced datasets
- **Hyperparameter Tuning**: Grid search over segmentation levels and feature counts
- **Model Comparison**: Multiple algorithms with comprehensive evaluation metrics

### Evaluation Metrics
- **Confusion Matrices**: Detailed classification performance analysis
- **ROC Curves**: Binary classification performance visualization
- **AUC Scores**: Area under the curve for model comparison
- **Cross-Validation Scores**: Robust performance estimation
- **Statistical Significance**: P-value analysis for feature importance

## Results Summary

- **Training samples:** 69
- **Test samples:** 19
- **Classes:** 7 activity types (bending1, bending2, cycling, lying, sitting, standing, walking)
- **Feature dimensions:** 43 features per sample (7 features × 6 sensors + instance ID)
- **Selected features:** min, mean, max for classification
- **Binary classification:** Bending vs. other activities
- **Multiclass classification:** All 7 activity types
- **Cross-validation:** 5-fold stratified validation
- **Feature selection methods:** RFE and p-value based selection compared

## Dependencies

See `requirements.txt` for complete list of Python packages required.

## Notes

- The dataset contains sensor data from mobile devices measuring RSS (Received Signal Strength) values
- Data preprocessing handles various CSV formats and missing values
- Bootstrap analysis provides confidence intervals for feature stability assessment
- Feature selection focuses on robust, interpretable time-domain characteristics
- Cross-validation ensures proper model evaluation without data leakage
- Class balancing techniques address imbalanced dataset issues
- Multiple classification algorithms are compared for comprehensive analysis
- Theoretical derivations provide mathematical foundation for QDA and Bayes classifiers

## Contact

For questions about this project, please contact:
- **Email:** [Your email]
- **GitHub:** yudiedeng
