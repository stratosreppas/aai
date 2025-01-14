from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import KFold

def import_dataset(dataset_id, random_state=42, splits=4):
    datasets = ['Australian', 'BreastCancer', 'Heart', 'Ionosphere', "MONK's", 'Parkinsons', 'Sonar', 'Wholesale']
    if dataset_id not in datasets:
        raise ValueError('Invalid dataset ID. Please choose from the following: {}'.format(datasets))
    
    # fetch dataset
    print(f'Loading {dataset_id} dataset...')
    if dataset_id == 'Australian':
        dataset = fetch_ucirepo(id=143)
    elif dataset_id == 'BreastCancer':  
        dataset = fetch_ucirepo(id=15)
        dataset.data.features = dataset.data.features.fillna(dataset.data.features.mean())
        dataset.data.targets = dataset.data.targets.replace({2: -1, 4: 1})
    elif dataset_id == 'Heart':
        dataset = fetch_ucirepo(id=145)
        dataset.data.targets = dataset.data.targets.replace({1: -1, 2: 1})
    elif dataset_id == 'Ionosphere':
        dataset = fetch_ucirepo(id=52)
        dataset.data.targets = dataset.data.targets.replace({'b': -1, 'g': 1})
    elif dataset_id == "MONK's":
        dataset = fetch_ucirepo(id=70)
        dataset.data.targets = dataset.data.targets.replace({0: -1})
    elif dataset_id == 'Parkinsons':
        pass
    elif dataset_id == 'Sonar':
        dataset = fetch_ucirepo(id=151)
        # drop any rows with missing values
        dataset.data.features = dataset.data.features.dropna()
        dataset.data.targets = dataset.data.targets.replace({'R': -1, 'M': 1})
    elif dataset_id == 'Wholesale':
        dataset = fetch_ucirepo(id=292)
        # set the channel as the target variable 
        dataset.data.targets = dataset.data.features['Channel'].to_frame()
        dataset.data.features = dataset.data.features.drop(columns='Channel')      
        dataset.data.targets = dataset.data.targets.replace({1: -1, 2: 1})    
    # data (as pandas dataframes) 
    X = dataset.data.features 
    y = dataset.data.targets

    # metadata 
    print(dataset.metadata) 
    print(dataset.variables)
    
    print(X.shape, y.shape)
    return X, y


def kfolds(X, y, splits=4):
    
    # perform 4-fold cross-validation split
    kf = KFold(n_splits=splits, shuffle=True)
    
    splits = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        splits.append((X_train, X_test, y_train, y_test))
    
    return splits