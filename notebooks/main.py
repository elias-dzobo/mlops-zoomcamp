# %%
import pandas as pd 

# %%
df = pd.read_parquet('../data/yellow_tripdata_2023-01.parquet')

df.head()

# %%
df.columns

# %%
df['duration'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']

df['duration'].head()

# %%
df['duration_minutes'] = df['duration'].dt.total_seconds() / 60
df['duration_minutes'].head()

# %%
df['duration_minutes'].std()

# %%
len(df)

# %%
mask = (df['duration_minutes'] >= 1) & (df['duration_minutes'] <= 60)

nonoutlier_df = df.loc[mask]

len(nonoutlier_df)

# %%
100 - (((len(df) - len(nonoutlier_df)) / len(df)) * 100)

# %%
from sklearn.feature_extraction import DictVectorizer



# Convert the DataFrame to a list of dictionaries
df_dict_list = nonoutlier_df[['PULocationID', 'DOLocationID']].astype(str).to_dict(orient='records')

# Initialize the dictionary vectorizer
dict_vectorizer = DictVectorizer(sparse=False)

# Fit and transform the data
feature_matrix = dict_vectorizer.fit_transform(df_dict_list)

# Display the feature matrix
feature_matrix.shape[1]


# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

model = LinearRegression()

model.fit(feature_matrix, nonoutlier_df['duration_minutes'])

y_pred = model.predict(feature_matrix)

rmse = root_mean_squared_error(nonoutlier_df['duration_minutes'], y_pred)
rmse

# %%



