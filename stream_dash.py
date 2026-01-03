# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 05:32:49 2025

@author: suryavp
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error,mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
import warnings
import itertools

# Create title and tabs
st.title('Dashboard I Created Based on the Student Assignment I Designed for a course')
css = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:1.5rem;
    font-weight: 700;
    }
</style>
'''
st.markdown(css, unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["Summary", "Insights", "Forecasting"])


temp_df=pd.read_csv("15minute_data_austin.csv", sep=";")

with tab1:
    #count how many houses' consumption data are in dataframe.
    num_houses=len(temp_df['dataid'].unique())
    
    st.markdown(
    f"<h3 style='font-size:26px;'>Number of houses in the dataset: {num_houses}</h3>",
    unsafe_allow_html=True)

    st.subheader("Columns in the dataset")
    cols = st.columns(4)
    for i, col_name in enumerate(temp_df.columns):
        with cols[i % 4]:
            st.markdown(f"<div style='background-color:#f0f2f6;padding:6px;border-radius:6px;text-align:center;'>{col_name}</div>", unsafe_allow_html=True)
    
    
    # get grid and solar and solar1
    df=temp_df[['dataid','localminute','grid','solar','solar2']]    
    
    #convert localminute datatype into datetime
    df['localminute']=pd.to_datetime(df['localminute'], infer_datetime_format=True)    
    # print unique values in dataid
    data_id=df['dataid'].unique()
    
    
    # create a function to fill NaN values in df columns: grid,solar,solar2
    def check_nan(temp_df):
       
       temp_df['grid']=temp_df['grid'].fillna(method='ffill') 
       
       temp_df['solar']=temp_df['solar'].fillna(0)
       
       temp_df['solar2']=temp_df['solar2'].fillna(0)
          
       return temp_df
   
    # fill missing timestamp.
    def check_timestamp(temp_df):
        
        temp_df=temp_df.sort_values(by=['localminute'])        
        temp_df=temp_df.set_index('localminute')         
                
        temp_df=temp_df.reindex(pd.date_range(start=temp_df.index[0], end=temp_df.index[-1], freq='15T'),copy='false',method='ffill') 
        temp_df.index.name = "localminute"
             
        
        return temp_df
    
    # Create total consumption of household and combine solar
    def sum_data(temp_df):
        
        total=temp_df['grid']+temp_df['solar']+temp_df['solar2']
        temp_df['total_consumption']=total  
        temp_df['solar']=temp_df['solar']+temp_df['solar2']
        temp_df = temp_df.drop('solar2', axis=1)
        
        return temp_df
    
    # function to check for negative consumption since its not possible
    # Note there can be negative values since basic imputation is used for solar
    def check_negative_consumption(temp_df):
        negative_rows=temp_df.loc[temp_df['total_consumption']<0]
    
    # resample data
    def resample_data(temp_df):    
        temp_df=temp_df.resample('60T').mean()
        # check for any missing or NaN values in the dataset    
        #check_nan=temp_df['total_consumption'].isnull().values.any()
        #print(check_nan)    
        #nan_rows=temp_df[temp_df.isna().any(axis=1)]
        #print(nan_rows)    
        return temp_df
    
    # create time related features
    def extract_time_features(wat):
        wat=wat.reset_index()            
        wat['day']=pd.DatetimeIndex(wat['localminute']).day
        wat['month']=pd.DatetimeIndex(wat['localminute']).month
        wat['day_of_week']=pd.DatetimeIndex(wat['localminute']).day_of_week
        wat['hour']=pd.DatetimeIndex(wat['localminute']).hour    
        wat['weekend']=(wat['day_of_week'] > 4).astype(float)
        wat.set_index('localminute')
            
        return wat
    
    # create columns representing last three consumption
    def extract_cons_features(temp_df):
        
        temp_df['c-1']=temp_df['total_consumption'].shift(1)    
        temp_df['c-2']=temp_df['c-1'].shift(1)
        temp_df['c-3']=temp_df['c-2'].shift(1)    
        temp_df=temp_df.fillna(0)
        
        return temp_df
    
    # check for outliers and replace it with ffill values
    def check_outliers(temp_df):
           
        temp_df['total_consumption']=temp_df['total_consumption'].mask(np.abs(stats.zscore(temp_df['total_consumption'])) > 2,np.nan)
        temp_df['solar']=temp_df['solar'].mask(np.abs(stats.zscore(temp_df['solar'])) > 2,np.nan)
        temp_df['total_consumption']=temp_df['total_consumption'].fillna(method='ffill')
        temp_df['solar']=temp_df['solar'].fillna(method='ffill')        
        
        return temp_df
    
    client_data=list()
    #seaprate the data based on dataid and store it as a separate dataframe in a list.
    for j in range(num_houses):        
        name=df.loc[df['dataid']==data_id[j]]
        name=name.drop(['dataid'], axis=1)
        name=check_nan(name)
        name=check_timestamp(name) 
        name=sum_data(name)
        name=resample_data(name)
        name=extract_time_features(name)
        name=extract_cons_features(name)
        check_negative_consumption(name)
        name=check_outliers(name)    
        client_data.append(name)    
    
    # allow users to select a house for analysis
    st.markdown("""
<style>
div[class*="stNumberInput"] label p {
  font-size: 26px;  
}

input[type=number] {
    font-size: 20px !important;
    font-weight: 600 !important;
    height: 34px !important;
}
</style>
""", unsafe_allow_html=True)
    cl_num = st.number_input("Select a house (1-"+str(num_houses)+")",value=1)
    cl_num=cl_num-1
    # display the table with selected columns
    st.write('The table provides information on hourly energy imported from or exported to the grid, solar generation, household consumption, and time-related features. Columns C-1, C-2, and C-3 represent the consumption values from the previous three hours.  ')
    st.dataframe(client_data[cl_num])



with tab2:
    
    # data analysis
    # find monthly consumption and plot bar graph
    def monthly_consumption(temp_df):
        consumption=list()
        solar_consumption=list()
        month=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']    
        for j in range(12):       
            val=temp_df.loc[temp_df['month']==j+1]
            t_cons=val['total_consumption'].sum()
            sol_cons=val['solar'].sum()
            consumption.append(t_cons)
            solar_consumption.append(sol_cons)
        
        fig, ax = plt.subplots()
    
        ax.bar(month, consumption, label="Total Consumption")
        ax.bar(month, solar_consumption, label="Solar Consumption")
        
        plt.xlabel('Months')
        plt.ylabel('Consumption (kWh)')
        plt.legend()
        st.subheader("Consumption data")
        st.write('This plot shows the total monthly energy consumption and the share covered by solar generation.')
        st.pyplot(fig)
        
        
        
    monthly_consumption(client_data[cl_num])
    
    # create correlation table
    def correlation(temp_df):
        cor=temp_df.corr()
        st.subheader("Correlation")
        st.write('Correlation tells us:')
        st.write('How much does one variable change when the other changes?')
        st.write('If one never changes (all zeros), the question is meaningless, value is undefined and will be NaN')
        st.write(cor['total_consumption'][1:])
        
        
    correlation(client_data[cl_num])
    
    # function for creating box-plot
    def boxplot(temp_df):
        st.subheader("Boxplot")        
        st.write('Box plot provides key statistics such as the median, quartiles, and potential outliers in a visual manner')
        fig, ax = plt.subplots()  
        temp_df[['total_consumption', 'solar']].boxplot(ax=ax)
        ax.set_ylabel('Consumption (kWh)')
        st.pyplot(fig)   
    boxplot(client_data[cl_num])
    
    # function for creating histogram
    def histogram(temp_df):
        st.subheader("Histogram")
        st.write('Histogram is a graph showing the number of observations within each given interval.')
        fig, ax = plt.subplots()
        temp_df.hist(ax=ax,column='total_consumption',bins=7)
        plt.title('Total_consumption')
        plt.ylabel('Frequency')
        plt.xlabel('Consumption (kWh)')
        st.pyplot(fig)
        fig, ax = plt.subplots()
        temp_df.hist(ax=ax,column='solar',bins=7)
        plt.title('Solar')
        plt.ylabel('Frequency')
        plt.xlabel('Consumption (kWh)')
        st.pyplot(fig)
        
    histogram(client_data[cl_num])
    
    # function to find minimum and maximum daily consumption and its corresponding dates
    def min_max_avg(temp_df):
        st.subheader("Some basic findings")
        temp_df=temp_df.set_index('localminute')
        temp_df=temp_df['total_consumption'].resample('D').sum()        
        minimum_cons=temp_df.min()
        st.write('Minimum daily consumption is',minimum_cons,'kWh')    
        minimum_cons_date=temp_df.idxmin()
        st.write('And the date is',minimum_cons_date)
        maximum_cons=temp_df.max()
        st.write('Maximum daily consumption is',maximum_cons,'kWh')    
        maximum_cons_date=temp_df.idxmax()
        st.write('And the date is',maximum_cons_date)
        avg_cons=temp_df.mean()
        st.write('Average daily consumption is',avg_cons,'kWh')
        
    #function to find the total consumption in each day of the week    
    def weekday_weekend(temp_df):    
        mon=temp_df.groupby(['day_of_week']).sum(numeric_only=True)        
        fig, ax = plt.subplots()
        mon.plot.bar(y='total_consumption',ax=ax)
        plt.xticks(ticks=[0,1,2,3,4,5,6],labels=['Mon','Tue','Wed','Thr','Fri','Sat','Sun'])
        plt.xlabel('Day of the week')
        plt.ylabel('Total Consumption (kWh)')
        st.write('The plot below shows the total consumption in each day of the week.')        
        st.pyplot(fig)
        

    min_max_avg(client_data[cl_num])
    weekday_weekend(client_data[cl_num])


with tab3:
    #forecasting
        
    st.subheader("Forecasting")
    st.write('ARIMA model is developed to predict the next 10 days consumption')
    warnings.filterwarnings('ignore')
    #Function to create best arima model by optimizing the arima parameters
    def arima(temp_df):
          
        temp_df=temp_df.set_index('localminute')
        temp_df=temp_df.resample('D').sum()    
        cons_series=temp_df['total_consumption']
        train_data=np.array(cons_series[0:100])    
        test_data=np.array(cons_series[100:110])
        p = range(0, 4)
        d = range(0, 3)
        q = range(0, 4)
        pdq = list(itertools.product(p, d, q))
        
        best_aic = np.inf
        best_order = None
        best_model = None
        
        for order in pdq:
            try:
                model = ARIMA(train_data, order=order)
                model_fit = model.fit(method_kwargs={'maxiter':300})
                if model_fit.aic < best_aic:
                    best_aic = model_fit.aic
                    best_order = order
                    best_model = model_fit
            except:
                continue
        
        st.write(f'Best ARIMA order: {best_order} with AIC: {best_aic}')
        
        forecast=best_model.forecast(10)    
        fig, ax = plt.subplots()
        x = range(1, len(forecast) + 1)
        ax.plot(x,forecast,label="Prediction")
        ax.plot(x,test_data,label="Actual")
        plt.xlabel('Test days')
        plt.ylabel('Total Consumption (kWh)')
        plt.legend() 
        st.write('The plot below shows the model performance for test days.')
        st.pyplot(fig)
        mse=mean_squared_error(test_data,forecast)
        mae=mean_absolute_error(test_data,forecast)
        mape=mean_absolute_percentage_error(test_data,forecast)
        st.write('Mean squared error:',mse)
        st.write('Mean absolute error:',mae)
        st.write('Mean absolute percentage error:',mape)
        
    arima(client_data[cl_num])
