import numpy as np
import pandas as pd
import pickle

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from typing import List, Tuple


data =  fetch_california_housing(as_frame=True)

df = pd.DataFrame(data['data'])
columns = df.columns.tolist()
df['label'] = data['target']
df.head()


shuffled_data = df.sample(frac=1)

split = int(len(shuffled_data) * .7)
train_df = df[:split]
test_df = df[split:]


def extract_X_and_y_from_df(
    df: pd.DataFrame,
    y_column: str,
    X_columns: list[str] = [],
) -> Tuple[dict, list]:
    X = {}
    for X_column in X_columns:
        X[X_column] = df[X_column].tolist()
    y = df[y_column].tolist()
    return X, y

X_train, y_train = extract_X_and_y_from_df(train_df, 'label', ['AveRooms'])
X_test, y_test = extract_X_and_y_from_df(test_df, 'label', ['AveRooms'])

X_train_arr = np.array(X_train['AveRooms'])
X_train_arr = X_train_arr.reshape(X_train_arr.shape[0],1)
X_test_arr = np.array(X_test['AveRooms'])
X_test_arr = X_test_arr.reshape(X_test_arr.shape[0],1)

model = LinearRegression()
model.fit(X_train_arr, y_train)

y_pred = model.predict(X_test_arr)

pickled_model = {'model': model}
pickle.dump(pickled_model, open( 'model_file' + ".p", "wb" ))
