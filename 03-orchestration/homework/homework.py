import pandas as pd
import argparse
import pickle

from pathlib import Path
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression

import mlflow
from prefect import task, flow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("orchestration-experiment-homework")

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)

@task(log_prints=True)
def load_data(year, month):
    """
    Load the data from the CSV file.
    """
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    print(f"The number of rows in the intial loaded dataframe: {len(df)}")
    return df


@task(log_prints=True)
def prepare_features(df: pd.DataFrame):
    """
    Prepare the features for the model.
    """
    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    print(f"the training data has {df.shape[0]} rows and {df.shape[1]} columns after cleaning.")

    return df

@task
def create_X(df, dv=None):
    """
    Create the feature matrix X and the DictVectorizer.
    """
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv

@task(log_prints=True)
def train_model(X_train, y_train, X_val, y_val, dv):
    """
    Train the model using XGBoost.
    """
    with mlflow.start_run() as run:
        mlflow.set_tag("developer", "jayadeep")
        mlflow.set_tag("model_type", "linear_regression")
        mlflow.set_tag("chapter", "03-orchestration-hw")
        mlflow.log_param("num_features", X_train.shape[1])
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)

        print(f"The intercept of the model is: {model.intercept_}")
        print(f"RMSE: {rmse}")
        mlflow.log_metric("rmse", rmse)

        mlflow.sklearn.log_model(sk_model=model, artifact_path="lr_model")
        
        with open("models/dict_vectorizer.pkl", "wb") as f:
            pickle.dump(dv, f)
        mlflow.log_artifact("models/dict_vectorizer.pkl", "dict_vectorizer")
        runid = run.info.run_id
    return model, dv, runid

@flow(log_prints=True)
def main(year, month):
    """
    Main function to run the workflow.
    """
    df_train = load_data(year, month)
    df_train = prepare_features(df_train)

    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1
    
    df_val = load_data(next_year, next_month)
    df_val = prepare_features(df_val)
    
    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    y_train = df_train.duration.values
    y_val = df_val.duration.values

    #take the runid from mlflow and register the model
    model, dv, runid = train_model(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, dv=dv)
    print(f"Model trained and saved with run ID: {runid}")
    ## Register the model in MLflow
    # mlflow.register_model(f"runs:/{runid}/models_mlflow", "duration_prediction_model")

if __name__ == "__main__":
    
    arg = argparse.ArgumentParser(description="Duration Prediction Model Training")
    arg.add_argument("--year", type=int, required=True, help="Year of the data")
    arg.add_argument("--month", type=int, required=True, help="Month of the data")
    args = arg.parse_args()
    year = args.year
    month = args.month
    print(f"Training model for year: {year}, month: {month}")
    # Call the main function with the provided year and month   
    main(year, month)