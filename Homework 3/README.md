# DSCI 552 - Homework 3: Time Series Classification

**Student:** Yudie Deng  
**Course:** DSCI 552 - Machine Learning for Data Science  
**Assignment:** Homework 3

## Overview

This project implements time series classification using the AReM (Activity Recognition system based on Multisensor data fusion) dataset. The main focus is on feature extraction from time series data and analysis of linear vs cubic regression models.

## Project Structure

```
Homework 3, 4 Data/
├── data/
│   ├── AReM_split/
│   │   ├── train/          # Training data (69 samples)
│   │   └── test/           # Test data (19 samples)
│   ├── bendingType.pdf
│   └── sensorsPlacement.pdf
├── notebook/
│   └── Deng_Yudie_HW3.ipynb
├── requirements.txt
└── README.md
```

## Dataset

The AReM dataset contains 7 types of human activities:
- **bending1, bending2**: Bending activities
- **cycling**: Cycling motion
- **lying**: Lying down
- **sitting**: Sitting position
- **standing**: Standing position
- **walking**: Walking motion

Each activity instance contains 6 time series (480 consecutive values each):
- `avg_rss12`, `var_rss12`: Average and variance of RSS between sensors 1-2
- `avg_rss13`, `var_rss13`: Average and variance of RSS between sensors 1-3
- `avg_rss23`, `var_rss23`: Average and variance of RSS between sensors 2-3

## Features

### Part 1: Time Series Classification
1. **Data Loading**: Robust loading of CSV files with different formats (comma and space separated)
2. **Train/Test Split**: 
   - Test: datasets 1 & 2 in bending1/bending2; datasets 1, 2, 3 in other folders
   - Train: remaining datasets
3. **Feature Extraction**: Time-domain features for each of the 6 time series:
   - Minimum, Maximum, Mean, Median
   - Standard Deviation, First Quartile, Third Quartile
4. **Bootstrap Analysis**: 90% bootstrap confidence intervals for feature standard deviations
5. **Feature Selection**: Three most important features (min, mean, max)

### Part 2: Linear vs Cubic Regression Analysis
1. **Theoretical Analysis**: Comparison of training and test RSS for linear vs cubic regression
2. **Empirical Demonstration**: Code examples with different relationship types
3. **Visualization**: Comprehensive plots showing bias-variance tradeoff

## Key Results

- **Total Samples**: 88 instances (69 train, 19 test)
- **Feature Matrix**: 42 features per instance (7 features × 6 time series)
- **Selected Features**: min, mean, max (justified by their complementary nature)
- **Bootstrap CI**: 90% confidence intervals for feature standard deviations

## Installation

1. Clone or download this repository
2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook notebook/Deng_Yudie_HW3.ipynb
```

2. Run all cells in order (Cell → Run All)

## Dependencies

- Python 3.7+
- NumPy, Pandas, SciPy
- Scikit-learn, Statsmodels
- Matplotlib, Seaborn
- Jupyter Notebook

## Methodology

### Feature Extraction
- **Time-domain features**: Statistical measures capturing signal characteristics
- **Robust loading**: Handles mixed CSV formats (comma and space separated)
- **Bootstrap validation**: Quantifies uncertainty in feature estimates

### Model Analysis
- **Bias-variance tradeoff**: Theoretical and empirical analysis
- **Overfitting analysis**: Training vs test performance comparison
- **Visualization**: Comprehensive plots for intuitive understanding

## Results Summary

1. **Feature Extraction**: Successfully extracted 42 time-domain features from 88 instances
2. **Bootstrap Analysis**: Computed 90% confidence intervals for all feature standard deviations
3. **Feature Selection**: Identified min, mean, max as most important features
4. **Regression Analysis**: Demonstrated bias-variance tradeoff in linear vs cubic models

## Files

- `notebook/Deng_Yudie_HW3.ipynb`: Main analysis notebook
- `requirements.txt`: Python dependencies
- `README.md`: This documentation
- `data/AReM_split/`: Processed dataset with train/test split

## Contact

For questions about this project, please contact the author.

---
*This project is part of DSCI 552 - Machine Learning for Data Science coursework.*
