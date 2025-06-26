import pickle
import pandas as pd
import datetime
import time

import prefect
from evidently import Report
from evidently import DataDefinition
from evidently import Dataset
from evidently.metrics import ValueDrift, DriftedColumnsCount, MissingValueCount
import psycopg

#-----------------------------data loading and model loading variables-------------------------------
# with open("model/model.pkl", "rb") as f_out:
#     model = pickle.load(f_out)

ref_data = pd.read_parquet("05-monitoring/data/reference_train.parquet")
curr_data = pd.read_parquet("05-monitoring/data/current_val.parquet")

categorical = ['PULocationID', 'DOLocationID']
numerical = ['trip_distance']

begin_date = datetime.datetime(2024, 3, 1, 0, 0)

#-----------------------------database variables-------------------------------
CONNECTION_STRING = "host = localhost port = 5432 user = postgres password = naga"
CONNECTION_STRING_DB = CONNECTION_STRING + " dbname = test"

create_table_statement= """drop table if exists evidently_report;
 create table evidently_report (
    timestamp timestamp,
    value_drift float,
    drifted_columns_count float,
    missing_value_count float)"""

#Wait for 10 seconds before sending the new data to db
SEND_TIMEOUT = 10
#-----------------------------Report generation variables-------------------------------
report = Report(metrics=[
    ValueDrift(column='prediction'),
    DriftedColumnsCount(),
    MissingValueCount(column = 'prediction')
])

data_definition = DataDefinition(
    numerical_columns=numerical + ['prediction'],
    categorical_columns= categorical)

#------------------function definitions for prefect flow and tasks---------------------------
def create_db():
    """Create a PostgreSQL database named 'test' if it does not exist.
       Else, Create a table named 'evidently_report' in the 'test' database.
    """
    with psycopg.connect(
        CONNECTION_STRING, autocommit= True) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname = 'test'")
        if len(res.fetchall()) == 0:
            conn.execute("CREATE DATABASE test")
            print("Database 'test' created successfully.")
        with psycopg.connect(CONNECTION_STRING_DB) as conn:
            conn.execute(create_table_statement)
            print("Table 'evidently_report' created successfully.")

def calculate_metric_postgresql(i):
    """Calculate the Evidently metrics and store the results in a PostgreSQL database.
       - The data is fed in a production simulated nature to generat the metrics for the report. This is hepful for
         practicing model monitoring in online model deployment scenarios.
       - The data usually is divided into per day batches and we use the loaded model to generate predictions. However,
            for simplicity, I have prepared the current and reference data in the 'data' folder beforehand.
       - In this function now, I will use this data to generate the Evidently report and store the results in a PostgreSQL database. 
    """
    #create the filter to give per day data to the report.
    print("Loading data for day:", (begin_date+ datetime.timedelta(i)))
    curr_data_temp = curr_data[(curr_data.lpep_pickup_datetime >= (begin_date+ datetime.timedelta(i))) &
                                (curr_data.lpep_pickup_datetime < (begin_date + datetime.timedelta(i+1)))] 
    ref_data_temp = ref_data[(ref_data.lpep_pickup_datetime >= (begin_date+ datetime.timedelta(i))) &
                                (ref_data.lpep_pickup_datetime < (begin_date + datetime.timedelta(i+1)))] 
    
    
    current_data = Dataset.from_pandas(curr_data_temp, data_definition=data_definition)
    reference_data = Dataset.from_pandas(ref_data_temp, data_definition=data_definition)

    # Run the report on the training and validation data
    # The training data is used as the reference data, and the validation data is the current data
    print("Generating Evidently report for day:", i+1)
    run = report.run(reference_data=reference_data, current_data=current_data)

    result = run.dict()

    #load the results into the PostgreSQL database
    with psycopg.connect(CONNECTION_STRING_DB) as conn:
        with conn.cursor() as cursor:
            # Insert the results into the table
            cursor.execute(
                "INSERT INTO evidently_report (timestamp, value_drift, drifted_columns_count, missing_value_count) "
                "VALUES (%s, %s, %s, %s)",
                (
                    (begin_date + datetime.timedelta(i)),
                    result['metrics'][0]['value'],
                    result['metrics'][1]['value']["count"],
                    result['metrics'][2]['value']['share']
                )
            )
            print("Metrics for day", i+1, "inserted successfully into the database.")

    # Return the generated report
    return None

def main():
    # Create the database and table
    create_db()
    # Loop through the days to generate the report loop 14 times as the number of days in both datasets are different and we take least number of days
    last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
    for i in range(1, 14):
        print(f"Calculating metrics for day {i+1}")
        calculate_metric_postgresql(i)
        
        #wait for 10 seconds before the data is sent again.
        new_send = datetime.datetime.now()
        seconds_elapsed = (new_send - last_send).total_seconds()
        if seconds_elapsed < SEND_TIMEOUT:
            time.sleep(SEND_TIMEOUT - seconds_elapsed)
        while last_send < new_send:
            last_send = last_send + datetime.timedelta(seconds=10)
    print("Evidently report generation completed successfully and db is populated.")


if __name__ == "__main__":
    main()