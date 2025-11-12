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

st.title('Dashboard of the assignment i created for student')
tab1, tab2, tab3 = st.tabs(["Summary", "Consumption data", "forecasting"])
temp_df=pd.read_csv (r'C:\Users\suryavp\.spyder-py3\15minute_data_austin.csv',sep=';')
#st.dataframe(temp_df)
#st.write(temp_df.columns)
with tab1:
    #count how many houses' consumption data are in dataframe.
    num_houses=len(temp_df['dataid'].unique())
    st.write('Number of houses in the dataset:',num_houses)
    st.subheader("Columns in the dataset")
    cols = st.columns(4)
    for i, col_name in enumerate(temp_df.columns):
        with cols[i % 4]:
            st.markdown(f"<div style='background-color:#f0f2f6;padding:6px;border-radius:6px;text-align:center;'>{col_name}</div>", unsafe_allow_html=True)
    
    
    # get grid and solar and solar1
    df=temp_df[['dataid','localminute','grid','solar','solar2']]
    #print(df)
    #print datatypes of all columns
    #print(df.dtypes)
    
    #convert localminute datatype into datetime
    df['localminute']=pd.to_datetime(df['localminute'], infer_datetime_format=True)
    #print(df.dtypes) 
    # print unique values in dataid
    data_id=df['dataid'].unique()
    
    #seaprate the data based on dataid and store it in a separate dataframe.
    # create a function to check for NaN values in df columns: grid,solar,solar2
    def check_nan(temp_df):
       #check nan in grid
       check_nan=temp_df['grid'].isnull().values.any()
       #print(check_nan)
       temp_df['grid']=temp_df['grid'].fillna(method='ffill')  
       check_nan=temp_df['grid'].isnull().values.any()
       #print(check_nan)
       #check nan in solar
       check_nan=temp_df['solar'].isnull().values.any()
       #print(check_nan)
       temp_df['solar']=temp_df['solar'].fillna(0)
       check_nan=temp_df['solar'].isnull().values.any()
       #print(check_nan)
       #check nan in solar2
       check_nan=temp_df['solar2'].isnull().values.any()
       #print(check_nan)
       temp_df['solar2']=temp_df['solar2'].fillna(0)
       check_nan=temp_df['solar2'].isnull().values.any()
       #print(check_nan)   
       return temp_df
    # check for missing timestamp.
    def check_timestamp(temp_df):
        #print(temp_df)
        temp_df=temp_df.sort_values(by=['localminute'])
        #print(temp_df)
        temp_df=temp_df.set_index('localminute')  
        #print(temp_df)
        check_nan=temp_df.isnull().values.any()
        #print(check_nan)        
        temp_df=temp_df.reindex(pd.date_range(start=temp_df.index[0], end=temp_df.index[-1], freq='15T'),copy='false',method='ffill') 
        temp_df.index.name = "localminute"
        check_nan=temp_df.isnull().values.any()
        #print(check_nan)
        #print(temp_df)     
        
        return temp_df
    def sum_data(temp_df):
        
        total=temp_df['grid']+temp_df['solar']+temp_df['solar2']
        temp_df['total_consumption']=total    
        return temp_df
    def check_negative_consumption(temp_df):
        negative_rows=temp_df.loc[temp_df['total_consumption']<0]
        #print(negative_rows)
    def resample_data(temp_df):    
        temp_df=temp_df.resample('60T').mean()
        #print(temp_df)    
        check_nan=temp_df['total_consumption'].isnull().values.any()
        #print(check_nan)    
        nan_rows=temp_df[temp_df.isna().any(axis=1)]
        #print(nan_rows)    
        return temp_df
    def extract_time_features(wat):
        wat=wat.reset_index()
        #print(wat)    
        wat['day']=pd.DatetimeIndex(wat['localminute']).day
        wat['month']=pd.DatetimeIndex(wat['localminute']).month
        wat['day_of_week']=pd.DatetimeIndex(wat['localminute']).day_of_week
        wat['hour']=pd.DatetimeIndex(wat['localminute']).hour    
        wat['weekend']=(wat['day_of_week'] > 4).astype(float)
        wat.set_index('localminute')
        #print(wat)    
        return wat
    def extract_cons_features(temp_df):
        
        temp_df['c-1']=temp_df['total_consumption'].shift(1)    
        temp_df['c-2']=temp_df['c-1'].shift(1)
        temp_df['c-3']=temp_df['c-2'].shift(1)    
        temp_df=temp_df.fillna(0)
        #print(temp_df)
        return temp_df
    def check_outliers(temp_df):
        check_nan=temp_df.isnull().values.any()
        #print(check_nan)   
        temp_df['total_consumption']=temp_df['total_consumption'].mask(np.abs(stats.zscore(temp_df['total_consumption'])) > 2,np.nan)
        temp_df['solar']=temp_df['solar'].mask(np.abs(stats.zscore(temp_df['solar'])) > 2,np.nan)
        temp_df['total_consumption']=temp_df['total_consumption'].fillna(method='ffill')
        temp_df['solar']=temp_df['solar'].fillna(method='ffill')
        #print(temp_df)
        
        return temp_df
    
    client_data=list()
    
    for j in range(num_houses):
        print(j)
        #name=str(client_names[j])
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
    #print(client_data)
    cl_num = st.number_input("Select a house (1-"+str(num_houses)+")",value=1)
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
        #print(consumption)
        fig, ax = plt.subplots()
    
        ax.bar(month, consumption, label="Total Consumption")
        ax.bar(month, solar_consumption, label="Solar Consumption")
        
        plt.xlabel('Months')
        plt.ylabel('Consumption (kWh)')
        plt.legend()
        st.subheader("Consumption data")
        st.pyplot(fig)
        
        
        
    monthly_consumption(client_data[cl_num])
    
    def correlation(temp_df):
        cor=temp_df.corr()
        st.subheader("Correlation")
        st.write(cor['total_consumption'][1:])
        st.write('Correlation tells us:')
        st.write('How much does one variable change when the other changes?')
        st.write('If one never changes (all zeros), the question is meaningless, value is undefined and will be NaN')
        
    correlation(client_data[cl_num])
    
    def boxplot(temp_df):
        st.subheader("Boxplot")
    
        fig, ax = plt.subplots()  
        temp_df[['total_consumption', 'solar']].boxplot(ax=ax)
        ax.set_ylabel('Consumption (kWh)')
        st.pyplot(fig)   
    boxplot(client_data[cl_num])
    
    def histogram(temp_df):
        st.subheader("Histogram")
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
    
    def min_max_avg(temp_df):
        st.subheader("Some basic findings")
        temp_df=temp_df.set_index('localminute')
        temp_df=temp_df['total_consumption'].resample('D').sum()
        #print(temp_df)
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
        
    def weekday_weekend(temp_df):    
        mon=temp_df.groupby(['day_of_week']).sum(numeric_only=True)
        #print(mon['total_consumption'])
        fig, ax = plt.subplots()
        mon.plot.bar(y='total_consumption',ax=ax)
        plt.xticks(ticks=[0,1,2,3,4,5,6],labels=['Mon','Tue','Wed','Thr','Fri','Sat','Sun'])
        plt.xlabel('Day of the week')
        plt.ylabel('Total Consumption (kWh)')        
        st.pyplot(fig)
        

    min_max_avg(client_data[cl_num])
    weekday_weekend(client_data[cl_num])
