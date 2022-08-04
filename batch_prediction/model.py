import argparse
import joblib 
import pandas as pd


def load_model():
    return joblib.load("assets/model/model.joblib")

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file')
    
    args = parser.parse_arg()
    return args.__dict__

def prediction(data: pd.DataFrame):
    mdl = load_model()
    proba_prediction = mdl.pipeline_prediction(data)
    
    return (proba_prediction[:,0], proba_prediction[:,1])

if __name__ == "__main__":
    
    args = parse_arguments()
    
    data_for_prediction = pd.read_csv(args['csv_file'])
    proba_zero, proba_one = prediction(data_for_prediction)
    
    data_for_prediction['proba0'] = proba_zero
    data_for_prediction['proba1'] = proba_one
    
    data_for_prediction[['proba0', 'proba1']].to_csv('assets/model/batch_prediction.csv')
    