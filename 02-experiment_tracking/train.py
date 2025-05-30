import os
import pickle
import click

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error


import mlflow


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment-homework02")
mlflow.sklearn.autolog()


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    with mlflow.start_run():

        mlflow.set_tag("developer","jayadeep")
        mlflow.log_param("train-data-path", "./green_tripdata_2021-01.csv")
        mlflow.log_param("valid-data-path", "./green_tripdata_2021-02.csv")
        mlflow.log_param("test-data-path", "./green_tripdata_2021-03.csv")
        params = {"max_depth":10,
                  "random_state":0}

        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

    min_split = rf.get_params()['min_samples_split']
    print("min_samples_split:", min_split)

if __name__ == '__main__':
    run_train()