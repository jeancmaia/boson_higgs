import joblib 
import pandas as pd


def load_model():
    return joblib.load("assets/model/model.joblib")
    
def get_model_response(input):
    X = pd.json_normalize(input.__dict__)
    mdl = load_model()
    
    proba_list = mdl.pipeline_prediction(X)
    threshold = mdl.threshold
    
    return dict(proba_0=proba_list[0], proba_1=proba_list[1], recommended_threshold=threshold)
    