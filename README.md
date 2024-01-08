# Supply Chain Analytics Project

## Overview

This project focuses on in-depth analytics, modeling, and mining of a comprehensive supply chain dataset. Through a series of notebooks, the project covers data analysis, modeling, and mining, providing valuable insights into various facets of the supply chain, including sales patterns, revenue analysis, delivery performance, and customer behavior.

## Project Structure

### 1. [Data_Analysis.ipynb](./Data_Analysis.ipynb)

- **Objective**: Conduct exploratory data analysis (EDA) and preprocessing of the supply chain dataset.
  
- **Key Features**:
  - Utilizes visualizations like histograms and bar charts to understand categorical data distribution.
  - Performs correlation analysis and box plots to uncover relationships between different features.
  - Drops unused columns, preparing the dataset for modeling.

### 2. [Data_Modeling.ipynb](./Data_Modeling.ipynb)

- **Objective**: Create dimension tables and form a fact table for subsequent analysis.
  
- **Key Features**:
  - Establishes dimension tables for categorical columns.
  - Merges dimension tables to form a fact table, extracting relevant date-related features.
  - Saves dimension tables and the fact table for further analysis.

### 3. [Data_Mining.ipynb](./Data_Mining.ipynb)

- **Objective**: Analyze sales and revenue, evaluate delivery performance, and apply customer RFM (Recency, Frequency, Monetary) analysis.
  
- **Key Features**:
  - Conducts monthly and yearly sales analysis.
  - Evaluates delivery performance, including average delivery time and on-time delivery rate.
  - Applies RFM analysis to identify top customers.
  - Conducts time series analysis of sales and geospatial analysis of customer locations.

## Usage

1. **Data_Analysis.ipynb**: Explore and visualize the supply chain dataset, perform preprocessing, and save the cleaned dataset.

2. **Data_Modeling.ipynb**: Create dimension tables, form a fact table, and extract relevant features for analysis.

3. **Data_Mining.ipynb**: Analyze sales, revenue, and delivery performance. Apply RFM analysis, time series analysis, and geospatial analysis.

## Requirements

- Python 3.x
- Libraries: pandas, matplotlib, seaborn, scikit-learn, geopandas (for geospatial analysis)

## Conclusion

This project provides a comprehensive view of the supply chain, enabling data-driven decision-making and strategic insights. Whether you are interested in sales patterns, customer segmentation, or delivery performance, this project serves as a valuable resource for supply chain analytics.
