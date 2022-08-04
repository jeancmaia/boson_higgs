"""
Base method for feature manipulation on the raw_data dataset. You should use this method 
for cumbersome feature creation steps, such as crafting new columns from actual covariates. 
An ordinary action that might declare as any classical transformer of scikit-sklearn, is that
to say null-imputation and scaling, must be implemented in this fashion.
"""
import dask.dataframe as dd
import dask.array as np

COLUMNS_TO_CLIP = {23: [0.9, 100]}
COLUMNS_TO_LOG2 = ['23_clip']
COLUMNS_TO_LOG10 = [0, 3, 5, 9, 13, 17, 21, 22, 24, 25, 26, 27]
DEFAULT_FEATURES = list(range(28)) # 28 features

def _features_push_pull(features: np.array, old_feature: str, new_features: str):
    features.remove(old_feature)
    features.append(new_features)
    return features
    
    
def _extract_last_name(str_series: dd.Series):
    """Sample method to extract the last name of a name. It splits a string by the spaces and
    returns the last position.  
    
    Args:
        str_series (pd.Series): pandas column with names to extract the last name.
    """
    return str_series.split(' ')[-1]

def _columns_clip(X: dd.DataFrame, features: np.array):
    for key, value in COLUMNS_TO_CLIP.items():
        X[str(key)+'_clip'] = X[key].clip(value[0], value[1])
        features = _features_push_pull(features, key, str(key)+'_clip')
    return X, features
        
def _columns_log2(X: dd.DataFrame, features: np.array):
    for feature in COLUMNS_TO_LOG2:
        X[str(feature)+'_log'] = np.log2(X[feature])
        features = _features_push_pull(features, feature, str(feature)+'_log')
    return X, features
    
def _columns_log10(X: dd.DataFrame, features: np.array):
    for feature in COLUMNS_TO_LOG10:
        X[str(feature)+'_log'] = np.log10(X[feature])
        features = _features_push_pull(features, feature, str(feature)+'_log')
    return X, features

def _create_twenty_eight(X: dd.DataFrame, features:np.array):
    X['28_log'] = X['27_log'] - X['26_log'] 
    features = _features_push_pull(features, '27_log', '28_log')
    return X, features

def run(X: dd.DataFrame, features:np.array):
    X, features = _columns_clip(X, features)
    X, features = _columns_log2(X, features)
    X, features = _columns_log10(X, features)
    X, features = _create_twenty_eight(X, features)
    return X, features

if __name__ == '__main__':
    target_features = ['target'] + DEFAULT_FEATURES
    
    data = dd.read_csv('assets/raw_data.csv', 
                    names = target_features,
                    dtype={'target': int},
                    blocksize=2e8
                    )
    data, features = run(data, DEFAULT_FEATURES)
    
    new_target_features = ['target'] + features
    data[new_target_features].to_csv('assets/cleaned_data/data-*.csv', index=False)