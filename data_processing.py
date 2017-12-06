import numpy as np
import pandas as pd

######### LOAD DATA ########

def process_data():

	#test = pd.read_csv('sample_submission_zero.csv')

	print("Loading and processing data ...")

	# Load Train and Train_2
	train = pd.read_csv('train.csv')
	train_2 = pd.read_csv('train_v2.csv')

	#Concatenate train and train_v2
	train= pd.concat((train, train_2), axis=0, ignore_index=True).reset_index(drop=True)
	#Reducing the size int64 --> int 8
	train['is_churn'] = train['is_churn'].astype(np.int8)

	# Merge with members data
	members = pd.read_csv('members_v3.csv')

	#Reducing the size int64 --> int 8
	members['city'] = members['city'].astype(np.int8)
	#Reducing the size int64 --> int 16
	members['bd'] = members['bd'].astype(np.int16)
	#Reducing the size int64 --> int 16
	members['registered_via'] = members['registered_via'].astype(np.int8)
	# Reducing the size of registration_init_time by creating 3 others columns
	members['registration_init_year'] = members['registration_init_time'].apply(lambda x: int(str(x)[:4]))
	members['registration_init_month'] = members['registration_init_time'].apply(lambda x: int(str(x)[4:6]))
	members['registration_init_date'] = members['registration_init_time'].apply(lambda x: int(str(x)[-2:]))

	members['registration_init_year'] = members['registration_init_year'].astype(np.int16)
	members['registration_init_month'] = members['registration_init_month'].astype(np.int8)
	members['registration_init_date'] = members['registration_init_date'].astype(np.int8)

	members = members.drop('registration_init_time', 1)

	df_train = pd.merge(train, members, how='left', on='msno')
	members =[]
	train = []

	#Preprocessing on gender
	gender = {'male':1, 'female':2}
	df_train['gender'] = df_train['gender'].map(gender)
	df_train = df_train.fillna(0)

	# Merge with transactions
	transactions = pd.read_csv('transactions.csv')
	print("Transactions read")
	transactions_2 = pd.read_csv('transactions_v2.csv')
	print("Transaction_2 read")
	df_transactions = pd.concat((transactions, transactions_2), axis=0, ignore_index=True).reset_index(drop=True)
	transactions_2 = []

	 
	df_transactions['payment_method_id'] = df_transactions['payment_method_id'].astype(np.int8)
	df_transactions['payment_plan_days'] = df_transactions['payment_plan_days'].astype(np.int16)
	df_transactions['plan_list_price'] = df_transactions['plan_list_price'].astype(np.int16)
	df_transactions['actual_amount_paid'] = df_transactions['actual_amount_paid'].astype(np.int16)
	df_transactions['is_auto_renew'] = df_transactions['is_auto_renew'].astype(np.int8)
	df_transactions['is_cancel'] = df_transactions['is_cancel'].astype(np.int8)

	df_transactions['transaction_date_year'] = df_transactions['transaction_date'].apply(lambda x: int(str(x)[:4]))
	df_transactions['transaction_date_month'] = df_transactions['transaction_date'].apply(lambda x: int(str(x)[4:6]))
	df_transactions['transaction_date_date'] = df_transactions['transaction_date'].apply(lambda x: int(str(x)[-2:]))

	df_transactions['membership_expire_date_year'] = df_transactions['membership_expire_date'].apply(lambda x: int(str(x)[:4]))
	df_transactions['membership_expire_date_month'] = df_transactions['membership_expire_date'].apply(lambda x: int(str(x)[4:6]))
	df_transactions['membership_expire_date_date'] = df_transactions['membership_expire_date'].apply(lambda x: int(str(x)[-2:]))

	df_transactions['transaction_date_year'] = df_transactions['transaction_date_year'].astype(np.int16)
	df_transactions['transaction_date_month'] = df_transactions['transaction_date_month'].astype(np.int8)
	df_transactions['transaction_date_date'] = df_transactions['transaction_date_date'].astype(np.int8)

	df_transactions['membership_expire_date_year'] = df_transactions['membership_expire_date_year'].astype(np.int16)
	df_transactions['membership_expire_date_month'] = df_transactions['membership_expire_date_month'].astype(np.int8)
	df_transactions['membership_expire_date_date'] = df_transactions['membership_expire_date_date'].astype(np.int8)


	## WARNING ##
	#df_transactions = pd.DataFrame(df_transactions['msno'].value_counts().reset_index())
	#df_transactions.columns = ['msno','trans_count']
	############
	df_train = pd.merge(df_train, df_transactions, how='left', on='msno')
	transactions= []


	# Merge with users_logs
	#df_train = pd.merge(df_train, user_logs, how='left', on='msno')
	user_logs_2 = pd.read_csv('user_logs_v2.csv')
	print("User_logs_2 read")

	user_logs_2.info()
	user_logs_2.head()
	#user_logs = pd.read_csv('user_logs.csv')
	#print("User_logs read")
	#user_logs_2 = pd.DataFrame(user_logs_2['msno'].value_counts().reset_index())
	#user_logs_2.columns = ['msno','logs_count']
	df_train = pd.merge(df_train, user_logs_2, how='left', on='msno')
	user_logs_2 = []
	print("Processing done !")

	return df_train