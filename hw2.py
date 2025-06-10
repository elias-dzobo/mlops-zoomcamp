import pandas as pd 
import mlflow
from prefect import task, flow
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer

@task()
def read_data(file_path):
    """
    Reads a Parquet file and returns a DataFrame.
    
    Parameters:
    file_path (str): The path to the Parquet file.
    
    Returns:
    pd.DataFrame: The DataFrame containing the data from the Parquet file.
    """
    df = pd.read_parquet(file_path)
    print(f"Data read has {len(df)} records")

    return df 

@task()
def read_dataframe(filename):
    df = read_data(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    print(f"Dataframe has {len(df)} records after filtering")
    
    return df

@task()
def transform_data(df):
    """
    Transforms the training data by extracting features and target variable.
    
    Returns:
    X (pd.DataFrame): Features for training.
    y (pd.Series): Target variable for training.
    """
    y_col = 'duration'


    dv = DictVectorizer()
    train_dicts = df[['PULocationID', 'DOLocationID']].to_dict(orient='records')
    X = dv.fit_transform(train_dicts)

    y = df[y_col].values

    return X, y, dv

@task()
def train_model(df):
    """
    Trains a linear regression model on the provided features and target variable.
    
    Parameters:
    X (pd.DataFrame): Features for training.
    y (pd.Series): Target variable for training.
    
    Returns:
    LinearRegression: The trained linear regression model.
    """
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("nyc_taxi")

    with mlflow.start_run():
        X, y, dv = transform_data(df)
        model = LinearRegression()
        model.fit(X, y)

        # Log parameters and metrics
        mlflow.log_param("model_type", "LinearRegression")
        
        # Log artifacts (vectorizer and model)
        mlflow.sklearn.log_model(dv, "dict_vectorizer")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="fare_model",
            registered_model_name="taxi_fare_model"  # Register in model registry
        )

    print(f"The intercept of the model is {model.intercept_}")
    
    return model, dv

@flow()
def main():
    # Example usage
    file_path = "/Users/eliasdzobo/Desktop/2025/mlops-2025/data/yellow_tripdata_2023-03.parquet"
    df = read_dataframe(file_path)

    model, dv = train_model(df)
    print("Model training completed.")
    # Save the model and DictVectorizer if needed
    # You can use joblib or pickle to save the model and dv
    import pickle
    pickle.dump(model, open('model.pkl', 'wb'))
    pickle.dump(dv, open('dict_vectorizer.pkl', 'wb'))


if __name__ == "__main__":
    main()
