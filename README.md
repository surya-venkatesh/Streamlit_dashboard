## Overview

The main objective of this project is to gain hands-on experience with **Streamlit and Docker containerization**.
Streamlit is a popular Python library for building interactive, web-based dashboards, while Docker is widely used to containerize applications and ensure consistent portability across environments.

## Background

The dashboard in this repository is based on a student assignment I designed during my PhD for master students. The original assignment was intended to provide students with basic, practical experience in:

* Data cleaning

* Data analysis

* Building simple forecasting models

The dataset used is the Household Power Consumption data from PECAN STREET (https://www.pecanstreet.org/dataport/
).
Because this was originally created for instructional purposes, the included code and models are meant to be tutorial-level. The techniques and model performance are not optimized for production use. **Note that Streamlit and Docker were not part of the original assignment**.

## What This Project Adds

I adapted the assignment to explore Streamlit and Docker by:

* Building a Streamlit dashboard that:

  - Summarizes the dataset

  - Allows users to select a household

  - Displays consumption insights

  - Fits and evaluates an ARIMA forecasting model on selected household data

* Containerizing the Streamlit application using Docker to ensure easy deployment and portability.

## Current Limitations

The app may feel slow because all data processing, analysis, and model training occur dynamically when the user selects a household. Results are not precomputed or cached.

## Dashboard Preview

Screenshots of the Streamlit dashboard are provided below.
![pic_1](https://github.com/surya-venkatesh/Streamlit_dashboard/blob/main/Dashboard_screenshots/pic_1.png)
![pic_2](https://github.com/surya-venkatesh/Streamlit_dashboard/blob/main/Dashboard_screenshots/pic_2.png)
![pic_3](https://github.com/surya-venkatesh/Streamlit_dashboard/blob/main/Dashboard_screenshots/pic_3.png)
![pic_4](https://github.com/surya-venkatesh/Streamlit_dashboard/blob/main/Dashboard_screenshots/pic_4.png)
![pic_5](https://github.com/surya-venkatesh/Streamlit_dashboard/blob/main/Dashboard_screenshots/pic_5.png)
 
