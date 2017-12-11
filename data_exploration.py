import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
from collections import Counter


train = pd.read_csv('D:/Kaggle Data/MMBOX Churn/churn/train.csv')
train_2 = pd.read_csv('D:/Kaggle Data/MMBOX Churn/churn/train_v2.csv')
test_2 = pd.read_csv('D:/Kaggle Data/MMBOX Churn/churn/sample_submission_v2.csv')
transactions = pd.read_csv('D:/Kaggle Data/MMBOX Churn/churn/transactions.csv')
transactions_2 = pd.read_csv('D:/Kaggle Data/MMBOX Churn/churn/transactions_v2.csv')
members_3 = pd.read_csv('D:/Kaggle Data/MMBOX Churn/churn/members_v3.csv')
user_logs_2 = pd.read_csv('D:/Kaggle Data/MMBOX Churn/churn/user_logs_v2.csv')
user_logs = pd.read_csv('D:/Kaggle Data/MMBOX Churn/churn/user_logs.csv')

print("=========== Train ===========")
print(train.head())
print(train.info())
print("=========== Train 2 ===========")
print(train_2.head())
print(train_2.info())
print("=========== Transaction ===========")
print(transactions.head())
print(transactions.info())
print("=========== Transaction 2 ===========")
print(transactions_2.head())
print(transactions_2.info())
print("=========== Members 3 ===========")
print(members_3.head())
print(members_3.info())
print("=========== User Logs  ===========")
print(user_logs.head())
print(user_logs.info())
print("=========== User Logs 2  ===========")
print(user_logs_2.head())
print(user_logs_2.info())

# Merge training data and members data on members id
training = pd.merge(left = train,right = members,how = 'left',on=['msno'])
training.head()
training.info()

# Changing the format of city and registered_via( except missing values) 
# from float to int and changing blank values with NAN( for city, registered_via and gender)
training['city'] = training.city.apply(lambda x: int(x) if pd.notnull(x) else "NAN")
training['registered_via'] = training.registered_via.apply(lambda x: int(x) if pd.notnull(x) else "NAN")
training['gender']=training['gender'].fillna("NAN")

training['registration_init_time'] = training.registration_init_time.apply(lambda x: datetime.strptime(str(int(x)), "%Y%m%d").date() if pd.notnull(x) else "NAN" )
training['expiration_date'] = training.expiration_date.apply(lambda x: datetime.strptime(str(int(x)), "%Y%m%d").date() if pd.notnull(x) else "NAN")

## Data Exploration in Training Data ##

# City count
plt.figure(figsize=(12,12))
plt.subplot(411)
city_order = training['city'].unique()
city_order=sorted(city_order, key=lambda x: float(x))
sns.countplot(x="city", data=training , order = city_order)
plt.ylabel('Count', fontsize=12)
plt.xlabel('City', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Frequency of City Count", fontsize=12)
plt.show()
city_count = Counter(training['city']).most_common()
print("City Count " +str(city_count))

#Registered Via Count
plt.figure(figsize=(12,12))
plt.subplot(412)
R_V_order = training['registered_via'].unique()
R_V_order = sorted(R_V_order, key=lambda x: str(x))
R_V_order = sorted(R_V_order, key=lambda x: float(x))
#above repetion of commands are very silly, but this was the only way I was able to diplay what I wanted
sns.countplot(x="registered_via", data=training,order = R_V_order)
plt.ylabel('Count', fontsize=12)
plt.xlabel('Registered Via', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Frequency of Registered Via Count", fontsize=12)
plt.show()
RV_count = Counter(training['registered_via']).most_common()
print("Registered Via Count " +str(RV_count))

#Gender count
plt.figure(figsize=(12,12))
plt.subplot(413)
sns.countplot(x="gender", data=training)
plt.ylabel('Count', fontsize=12)
plt.xlabel('Gender', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Frequency of Gender Count", fontsize=12)
plt.show()
gender_count = Counter(training['gender']).most_common()
print("Gender Count " +str(gender_count))


## Registration_init_time Trends exploration ##

#registration_init_time yearly trend
training['registration_init_time_year'] = pd.DatetimeIndex(training['registration_init_time']).year
training['registration_init_time_year'] = training.registration_init_time_year.apply(lambda x: int(x) if pd.notnull(x) else "NAN" )
year_count=training['registration_init_time_year'].value_counts()
#print(year_count)
plt.figure(figsize=(12,12))
plt.subplot(311)
year_order = training['registration_init_time_year'].unique()
year_order=sorted(year_order, key=lambda x: str(x))
year_order = sorted(year_order, key=lambda x: float(x))
sns.barplot(year_count.index, year_count.values,order=year_order)
plt.ylabel('Count', fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Yearly Trend of registration_init_time", fontsize=12)
plt.show()
year_count_2 = Counter(training['registration_init_time_year']).most_common()
print("Yearly Count " +str(year_count_2))

#registration_init_time monthly trend
training['registration_init_time_month'] = pd.DatetimeIndex(training['registration_init_time']).month
training['registration_init_time_month'] = training.registration_init_time_month.apply(lambda x: int(x) if pd.notnull(x) else "NAN" )
month_count=training['registration_init_time_month'].value_counts()
plt.figure(figsize=(12,12))
plt.subplot(312)
month_order = training['registration_init_time_month'].unique()
month_order = sorted(month_order, key=lambda x: str(x))
month_order = sorted(month_order, key=lambda x: float(x))
sns.barplot(month_count.index, month_count.values,order=month_order)
plt.ylabel('Count', fontsize=12)
plt.xlabel('Month', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Monthly Trend of registration_init_time", fontsize=12)
plt.show()
month_count_2 = Counter(training['registration_init_time_month']).most_common()
print("Monthly Count " +str(month_count_2))

#registration_init_time day wise trend
training['registration_init_time_weekday'] = pd.DatetimeIndex(training['registration_init_time']).weekday_name
training['registration_init_time_weekday'] = training.registration_init_time_weekday.apply(lambda x: str(x) if pd.notnull(x) else "NAN" )
day_count=training['registration_init_time_weekday'].value_counts()
plt.figure(figsize=(12,12))
plt.subplot(313)
#day_order = training['registration_init_time_day'].unique()
day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday','NAN']
sns.barplot(day_count.index, day_count.values,order=day_order)
plt.ylabel('Count', fontsize=12)
plt.xlabel('Day', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Day-wise Trend of registration_init_time", fontsize=12)
plt.show()
day_count_2 = Counter(training['registration_init_time_weekday']).most_common()
print("Day-wise Count " +str(day_count_2))

## FOCUS ON BIRTH DATE ##

#Ignore negative birth date and more than 100
training['bd'] = training.bd.apply(lambda x: -99999 if float(x)<=1 else x )
training['bd'] = training.bd.apply(lambda x: -99999 if float(x)>=100 else x )

#Throw NAN and -99999 values
tmp_bd = training[(training.bd != "NAN") & (training.bd != -99999)]
print("Mean of Birth Date = " +str(np.mean(tmp_bd['bd'])))
print("Median of Birth Date = " +str(np.median(tmp_bd['bd'])))
#print("Mode of Birth Date = " +str(np.mode(tmp_bd['bd'])))
plt.figure(figsize=(12,8))
plt.subplot(211)
bd_order_2 = tmp_bd['bd'].unique()
bd_order_2 = sorted(bd_order_2, key=lambda x: float(x))
sns.countplot(x="bd", data=tmp_bd , order = bd_order_2)
plt.ylabel('Count', fontsize=12)
plt.xlabel('BD', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Frequency of BD Count without ouliers and NAN values", fontsize=12)
plt.show()

plt.figure(figsize=(4,12))
plt.subplot(212)
sns.boxplot(y=tmp_bd["bd"],data=tmp_bd)
plt.xlabel('BD', fontsize=12)
plt.title("Box Plot of Birth Date without ouliers and NAN values", fontsize=12)
plt.show()



