import pandas as pd 
import numpy as np

user_logs_2 = pd.read_csv('user_logs.csv')

user_logs_2['num_25'] = user_logs_2['num_25'].astype(np.int16)
user_logs_2['num_50'] = user_logs_2['num_50'].astype(np.int16)
user_logs_2['num_75'] = user_logs_2['num_75'].astype(np.int16)
user_logs_2['num_985'] = user_logs_2['num_985'].astype(np.int16)
user_logs_2['num_100'] = user_logs_2['num_100'].astype(np.int16)
user_logs_2['num_unq'] = user_logs_2['num_unq'].astype(np.int16)
user_logs_2['total_secs'] = user_logs_2['total_secs'].astype(np.int32)

# Reducing the size of date column by creating 3 others columns
user_logs_2['date_year'] = user_logs_2['date'].apply(lambda x: int(str(x)[:4]))
user_logs_2['date_month'] = user_logs_2['date'].apply(lambda x: int(str(x)[4:6]))
user_logs_2['date_day'] = user_logs_2['date'].apply(lambda x: int(str(x)[-2:]))

user_logs_2['date_year'] = user_logs_2['date_year'].astype(np.int16)
user_logs_2['date_month'] = user_logs_2['date_month'].astype(np.int8)
user_logs_2['date_day'] = user_logs_2['date_day'].astype(np.int8)

user_logs_2 = user_logs_2.drop('date', 1)


user_logs_2.to_csv('new_user_logs.csv', index = False)
