# Telco Customer Churn Analysis

## Overview
This project analyzes customer churn patterns in a telecommunications company using machine learning techniques. The analysis follows the SEMMA (Sample, Explore, Modify, Model, Assess) methodology and includes both supervised learning for churn prediction and unsupervised learning for customer segmentation.

## Key Features
- Data preprocessing and feature engineering
- Churn prediction using Random Forest and Logistic Regression
- Customer segmentation using K-means clustering
- Comprehensive model evaluation and insights

## Requirements
```
pandas
numpy
scikit-learn
```

## Dataset
The analysis uses the telco customer churn dataset (`telco-customer-churn.csv`) which includes:
- Customer demographics
- Account information
- Services subscribed
- Churn status

## Key Findings

### 1. Model Performance
- Random Forest Classifier achieved 80% accuracy
- Strong performance in identifying loyal customers (92% recall)
- Key predictors include:
  - Total Charges (16.5%)
  - Monthly Charges (15.8%)
  - Tenure (14.6%)
  - Contract Type (10.7%)

### 2. Customer Segments

#### Loyal Basic Users (22% of customers)
- Lowest churn rate: 7.4%
- Moderate tenure: 30.5 months
- Lowest monthly charges: $21.89
- Highest service adoption: 7.0 services

#### Premium Long-term Customers (33%)
- Moderate churn risk: 15.8%
- Longest tenure: 57.1 months
- Highest monthly charges: $89.78
- High service adoption: 5.9 services

#### High-Risk New Customers (45%)
- High churn rate: 43.7%
- Short tenure: 15.3 months
- Moderate charges: $67.60
- Low service adoption: 3.4 services

## Recommendations

### 1. Early Intervention Program
- Target customers in their first 15 months
- Focus on increasing service adoption
- Offer contract incentives

### 2. Service Bundle Strategy
- Create attractive service bundles
- Price optimization for multiple services
- Focus on security and support services

### 3. Contract Optimization
- Promote longer-term contracts
- Develop loyalty rewards program
- Implement proactive retention measures

## Usage
```bash
python telco_churn_analysis.py
```

## Implementation KPIs
- Early-stage churn rate (< 15 months)
- Service adoption rate
- Contract conversion rate
- Customer Lifetime Value (CLV)
- Monthly retention rate by segment