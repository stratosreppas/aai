from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from datasets import import_dataset, kfolds
from noise import flip_labels
from OCTSVM import OCTSVM
from OCT.tree.oct import optimalDecisionTreeClassifier
import numpy as np
import csv

datasets = ['Australian', 'BreastCancer', 'Heart', 'Ionosphere', "MONK's", 'Sonar', 'Wholesale']
tree_types = ['OCTSVM']
flip_percentages = [0, 20, 30, 40]
max_depth = 3
repetitions = 4
folds=4

results = {}

def experiment(model, X_train, X_test, y_train, y_test):
    accuracy = 0
    try: 
        print('Fitting model...')
        model.fit(np.array(X_train), (np.array(y_train)))1
        print('Predicting labels...')
        labels = model.predict(np.array(X_test))
        print(labels)
        accuracy = accuracy_score(y_test, labels)
    except Exception as e:
        print(e)
        return -1     
    
    return accuracy

def save_result(dataset, tree_type, percentage, parameters, fold, accuracy):
    with open('results.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([dataset, tree_type, percentage, parameters, fold, accuracy])

if __name__ == '__main__':
    # Initialize the CSV file with headers
    with open('results.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'tree_type', 'flip_percentage', 'parameters', 'fold', 'accuracy'])

    for dataset in datasets:
        X, y = import_dataset(dataset)
        print(y)
        print(y.shape)
        for tree_type in tree_types:    
            for percentage in flip_percentages:
                for fold, (X_train, X_test, y_train, y_test) in enumerate(kfolds(X, y, folds)):
                    y_train = flip_labels(y_train, percentage)
                    for i in range(-2, 2):
                        c1 = 10**i
                        
                        if tree_type == 'CART':
                            tree = DecisionTreeClassifier(criterion='gini', max_depth=max_depth, random_state=0, ccp_alpha=c1)
                        
                        if tree_type == 'OCT':
                            tree = optimalDecisionTreeClassifier(max_depth=max_depth, min_samples_split=2, alpha=c1)
                        
                        if tree_type == 'OCTSVM':
                            for j in range(-2, 1):
                                c2 = 10**j
                                for k in range(-2, 1):
                                    c3 = 10**k
                                    tree = OCTSVM(max_depth=max_depth-1, c1=c1, c2=c2, c3=c3)
                                    print(f'Running experiment for {dataset}, {tree_type}, {percentage}, {c1}, {c2}, {c3}, {fold+1}')
                                    accuracy = experiment(tree, X_train, X_test, y_train, y_test)
                                    results[(dataset, tree_type, percentage, (c1, c2, c3), fold+1)] = accuracy
                                    print('Accuracy:', accuracy)
                                    save_result(dataset, tree_type, percentage, (c1, c2, c3), fold+1, accuracy)
                        
                        else:            
                            print(f'Running experiment for {dataset}, {tree_type}, {percentage}, {c1}, {fold+1}')                                                           
                            accuracy = experiment(tree, X_train, X_test, y_train, y_test)
                            results[(dataset, tree_type, percentage, (c1), fold+1)] = accuracy
                            print('Accuracy:', accuracy)
                            save_result(dataset, tree_type, percentage, (c1), fold+1, accuracy)
    print(results)