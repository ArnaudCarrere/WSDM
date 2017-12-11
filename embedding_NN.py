# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as ts
from keras.models import Sequential
from keras.layers import Dense, Activation, Merge, Reshape, Dropout
from keras.layers.embeddings import Embedding
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from data_processing import process_data

#random seeds for stochastic parts of neural network 
np.random.seed(10)

ts.set_random_seed(15)

def build_embedding_network():
    
    models = []
    
    model_city = Sequential()
    model_city.add(Embedding(22, 3, input_length=1))
    model_city.add(Reshape(target_shape=(3,)))
    models.append(model_city)
    
    model_bd = Sequential()
    model_bd.add(Embedding(228, 10, input_length=1))
    model_bd.add(Reshape(target_sape=(2,)))
    models.append(model_bd)
    
    model_gender = Sequential()
    model_gender.add(Embedding(3, 2, input_length=1))
    model_gender.add(Reshape(target_shape=(5,)))
    models.append(model_gender)
    
    model_registered_via = Sequential()
    model_registered_via.add(Embedding(6, 3, input_length=1))
    model_registered_via.add(Reshape(target_shape=(7,)))
    models.append(model_registered_via)
    
    model_registration_init_year = Sequential()
    model_registration_init_year.add(Embedding(15, 7, input_length=1))
    model_registration_init_year.add(Reshape(target_shape=(2,)))
    models.append(model_registration_init_year)
    
    model_registration_init_month = Sequential()
    model_registration_init_month.add(Embedding(13, 5, input_length=1))
    model_registration_init_month.add(Reshape(target_shape=(2,)))
    models.append(model_registration_init_month)
    
    model_registration_init_date = Sequential()
    model_registration_init_date.add(Embedding(32, 13, input_length=1))
    model_registration_init_date.add(Reshape(target_shape=(5,)))
    models.append(model_registration_init_date)
    
    model_trans_count = Sequential()
    model_trans_count.add(Embedding(86, 20, input_length=1))
    model_trans_count.add(Reshape(target_shape=(2,)))
    models.append(model_trans_count)

    #model_logs_count = Sequential()
    #model_logs_count.add(Embedding(32, 14, input_length=1))
    #model_logs_count.add(Reshape(target_shape=(8,)))
    #models.append(model_logs_count)
    
    model_rest = Sequential()
    model_rest.add(Dense(16, input_dim=24))
    models.append(model_rest)

    model = Sequential()
    model.add(Merge(models, mode='concat'))
    model.add(Dense(80))
    model.add(Activation('relu'))
    model.add(Dropout(.35))
    model.add(Dense(20))
    model.add(Activation('relu'))
    model.add(Dropout(.15))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dropout(.15))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam')
    
    return model

#converting data to list format to match the network structure
def preproc(X_train, X_val, X_test):

    input_list_train = []
    input_list_val = []
    input_list_test = []
    
    #the cols to be embedded: rescaling to range [0, # values)
    for c in embed_cols:
        raw_vals = np.unique(X_train[c])
        val_map = {}
        for i in range(len(raw_vals)):
            val_map[raw_vals[i]] = i       
        input_list_train.append(X_train[c].map(val_map).values)
        input_list_val.append(X_val[c].map(val_map).fillna(0).values)
        input_list_test.append(X_test[c].map(val_map).fillna(0).values)
     
    #the rest of the columns
    other_cols = [c for c in X_train.columns if (not c in embed_cols)]
    input_list_train.append(X_train[other_cols].values)
    input_list_val.append(X_val[other_cols].values)
    input_list_test.append(X_test[other_cols].values)
    
    return input_list_train, input_list_val, input_list_test 

def logloss(y, pred):
    N = len(y)
    score = 0
    for i in range (N):
        score += y[i]*log(pred[i])+(1-y[i])*log(1-pred[i])
    return -score/N

## Network training ##


# Loading data
df_train = process_data()


print("First steps of the neural network... ")

cols = [c for c in df_train.columns if c not in ['is_churn','msno']]

X_train, y_train, X_test, y_test = train_test_split(df_train[cols], df_train['is_churn'], test_size=0.30, random_state=242)

print(" X_train = ", X_train)
print("y_train = ", y_train)
print(" X_test = ", X_test)
print("y_test = ", y_test)

col_vals_dict = {c: list(X_train[c].unique()) for c in X_train.columns}

embed_cols = []
for c in col_vals_dict:
    if len(col_vals_dict[c])>2:
        embed_cols.append(c)
        print(c + ': %d values' % len(col_vals_dict[c])) #look at value counts to know the embedding dimensions

print('\n')

K = 6
runs_per_fold = 3
n_epochs = 15

y_preds = np.zeros((np.shape(X_test)[0],K))

kfold = StratifiedKFold(n_splits = K, random_state = 231, shuffle = True)    

print("Training ....")

for i, (f_ind, outf_ind) in enumerate(kfold.split(X_train, y_train)):

    X_train_f, X_val_f = X_train.loc[f_ind].copy(), X_train.loc[outf_ind].copy()
    y_train_f, y_val_f = y_train[f_ind], y_train[outf_ind]
    
    X_test_f = X_test.copy()
    
    ###### WARNING #######

    #upsampling adapted from kernel: 
    #https://www.kaggle.com/ogrellier/xgb-classifier-upsampling-lb-0-283
    pos = (pd.Series(y_train_f == 1))
    
    # Add positive examples
    X_train_f = pd.concat([X_train_f, X_train_f.loc[pos]], axis=0)
    y_train_f = pd.concat([y_train_f, y_train_f.loc[pos]], axis=0)

    ######################

    # Shuffle data
    idx = np.arange(len(X_train_f))
    np.random.shuffle(idx)
    X_train_f = X_train_f.iloc[idx]
    y_train_f = y_train_f.iloc[idx]
    
    #preprocessing
    proc_X_train_f, proc_X_val_f, proc_X_test_f = preproc(X_train_f, X_val_f, X_test_f)
    
    #track of prediction for cv scores
    val_preds = 0
    
    for j in range(runs_per_fold):
    
        NN = build_embedding_network()
        NN.fit(proc_X_train_f, y_train_f.values, epochs=n_epochs, batch_size=4096, verbose=0)
   
        val_preds += NN.predict(proc_X_val_f)[:,0] / runs_per_fold
        y_preds[:,i] += NN.predict(proc_X_test_f)[:,0] / runs_per_fold

    cv_logloss = logloss(y_val_f.values, val_preds)
    print ('\nFold %i prediction cv logloss: %.5f\n' %(i,cv_logloss))


print(" Training done !")
### END TRAINING ###

### Final prediction ###
print(" Predicting ....")
# Load and process data #
test_2 = pd.read_csv('sample_submission_v2.csv')
#Reducing the size int64 --> int 8
test_2['is_churn'] = test_2['is_churn'].astype(np.int8)
test_2['gender'] = test['gender'].map(gender)
test_2 = test.fillna(0)
print("Sample_submission read")

# Merge data for submission
df_test = test_2
df_test = pd.merge(df_test, members, how='left', on='msno')
df_test = pd.merge(df_test, user_logs, how='left', on='msno')
df_test = pd.merge(df_test, user_logs_2, how='left', on='msno')
df_test = pd.merge(df_test, transactions, how='left', on='msno')
df_test = pd.merge(df_test, transactions_2, how='left', on='msno')

#Predict
pred_final = NN.predict(df_test)[:,0]
## Given an interval, values outside the interval are clipped to the interval edges
pred_final = pred_final.clip(0.+1e-15, 1-1e-15)

df_sub = pd.DataFrame({'msno' : df_test.id, 
                       'is_churn' : pred_final},
                       columns = ['msno','is_churn'])
df_sub.to_csv('submission.csv', index=False)


