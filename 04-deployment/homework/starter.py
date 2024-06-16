#!/usr/bin/env python
# coding: utf-8



import pickle
import pandas as pd
import sys
from statistics import mean 

# year = 2023
# month = 3

year = int(sys.argv[1])
month = int(sys.argv[2])

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# Q1
# import numpy as np
# np.std(y_pred)


df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


# df['ride_id'] = (df.tpep_pickup_datetime.dt.year.map("{:04}".format).astype('int')/df.tpep_pickup_datetime.dt.month.map("{:02}".format).astype('int')).astype('int').astype('string')\
#       +'_' + df.index.astype('str')

df_result = pd.concat([df['ride_id'].reset_index(drop=True),pd.Series(y_pred,name='y_pred')], axis=1)

df_result.to_parquet(
    'df_result.parquet',
    engine='pyarrow',
    compression=None,
    index=False
)
print(sum(y_pred)/len(y_pred))
print(mean(y_pred))