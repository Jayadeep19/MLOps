
import pickle
import pandas as pd
#import os
import sys


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    print(f"Reading data from {filename}")
    df = pd.read_parquet(filename, engine='pyarrow')

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


def load_model(path):
    print(f"Loading model from {path}")
    with open(path, 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model

def get_predictions(dv, model, df):
    print("Generating predictions")
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    #write the predictions to the dataframe
    print("Writing predictions to the dataframe")
    year = df["tpep_pickup_datetime"].dt.year.values[0]
    month = df["tpep_pickup_datetime"].dt.month.values[0]
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_results = pd.DataFrame({
    'ride_id': df.ride_id,
    'predicted_duration': y_pred
    })
    return df_results


def run():

    year = int(sys.argv[1]) #2023
    month = int(sys.argv[2]) #3
    #input = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input =  "yellow_tripdata_2023-03.parquet"  #f"yellow_tripdata_{year:04d}-{month:02d}.parquet"
    df = read_data(input)
    dv, model = load_model('model.bin')
    df_results = get_predictions(dv, model, df)
    print("The mean of the predicted durations is: ", df_results['predicted_duration'].mean())

if __name__ == '__main__':
    run()
    #print("Homework completed successfully!")
