from flask import Flask, render_template, request
import os
import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tabulate import tabulate
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Directory path where models are stored
datasets_directory = "datasets"
models_directory = "models"

def load_saved_model(model_name):
    global model_flag

    model_path = os.path.join( 'models',model_name)
    if 'pkl' in model_path:
        model = joblib.load(model_path)
        model_flag = 0
    else:
        model = load_model(model_path)
        model_flag = 1

    return model

def compare_close_values(group):
    group['target'] = (group['close'].shift(-1) > group['close']).astype(int)
    return group

def dataset_processing (dataset_loc):

    df = pd.read_csv(dataset_loc)
    if 'FINAL_FROM_DF.csv' in dataset_loc:
        df['TARGET'] = (df['CLOSE'] > df['PREVCLOSE']).astype(int)
        features = df[['OPEN', 'HIGH', 'LOW', 'CLOSE', 'LAST', 'PREVCLOSE', 'TOTTRDQTY', 'TOTTRDVAL']]
        target = df['TARGET']
        if model_flag ==1:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_features = scaler.fit_transform(features)

            _, X_test, _, _ = train_test_split(scaled_features, target, test_size=0.2, random_state=42)
        else:
            _, X_test, _, _ = train_test_split(features, target, test_size=0.2, random_state=42)
        if model_flag == 1:
            X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

        # X_test = X_test.reshape(-1, X_test.shape[-1])
        # X_test =pd.DataFrame(X_test)
    elif 'all_stocks_5yr' in dataset_loc:

        df = df.sort_values(by=['Name', 'date'])

        df['target'] = 0
        df = df.groupby('Name', group_keys=False).apply(compare_close_values)
        df = df.reset_index(drop=True)

        df = df.sample(frac=1).reset_index(drop=True)
        features = df[['open', 'high', 'low', 'close', 'volume']]
        target = df['target']

        df.dropna(inplace=True)
        df = df.reset_index(drop=True)
        _, X_test, _, _ = train_test_split(features, target, test_size=0.2, random_state=42)
        X_test.dropna(inplace=True)

    return df, X_test


def perform_testing(model_name, dataset_name):
    model = load_saved_model(model_name)
    df, X_test = dataset_processing (dataset_name)
  
    predictions = model.predict(X_test)
    threshold = 0.5
    mapped_predictions = [1 if val > threshold else 0 for val in predictions.flatten()]

    indices = np.arange(len(X_test)) 

    if 'FINAL_FROM_DF' in dataset_name:
        symbols = [df.loc[index, 'SYMBOL'] for index in indices]
        time_stamps = [df.loc[index, 'TIMESTAMP'] for index in indices]
    elif 'all_stocks_5yr.csv' in dataset_name:
        symbols = [df.loc[index, 'Name'] for index in indices]
        time_stamps = [df.loc[index, 'date'] for index in indices]    

    symbols = pd.Series(symbols)

    results_df = pd.DataFrame({'SYMBOL': symbols, 'Prediction': mapped_predictions, 'TIME_STAMP' : time_stamps})
    results_df['Prediction'] = results_df['Prediction'].map({0: 'sell', 1: 'buy'})
    results_df = results_df.sample(frac=1).reset_index(drop=True)
    return (tabulate(results_df.head(15), headers='keys', tablefmt='html'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test', methods=['POST'])
def test():
    model_name = request.form['modelName']
    dataset_file = request.files['datasetLocation']

    dataset_filename = dataset_file.filename
    dataset_file.save('uploads/' + dataset_filename)
    dataset_location = os.path.join('uploads', dataset_filename)


    # if 'FINAL_FROM_DF' in dataset_filename:
    results = perform_testing(model_name, dataset_location)

 
    # os.remove(dataset_location + dataset_filename)

    return render_template('index.html', table_html=results)

if __name__ == '__main__':
    app.run(debug=True)

