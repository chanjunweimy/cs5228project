import lightgbm as lgb
import numpy as np
import pandas as pd
from functools import partial
from hyperopt import fmin
from hyperopt import hp
from hyperopt import STATUS_OK
from hyperopt import space_eval
from hyperopt import Trials
from hyperopt import tpe
from lightgbm import LGBMClassifier
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm

NUM_OF_COL = 40


def load_features(col):
    return pd.read_parquet("filtered_features/filtered_feature{}.gzip".format(col))


def load_test_features(col):
    return pd.read_parquet("filtered_test_features/filtered_test_feature{}.gzip".format(col))


def select_train(selected_features):
    train = []
    for col in tqdm(range(NUM_OF_COL)):
        if selected_features[col] == False:
            continue
        tf = load_features(col)
        train.append(tf)
    train = np.concatenate(train,  axis=1)
    return train


def select_test(selected_features):
    test = []
    for col in range(NUM_OF_COL):
        if selected_features[col] == False:
            continue
        tf = load_test_features(col)
        tf = tf.replace([np.inf], np.finfo('float32').max).replace(
            [np.inf, -np.inf], np.finfo('float32').min).fillna(0)
        test.append(tf)
    test = np.concatenate(test,  axis=1)
    return test


def calculate_auc(X_train, X_valid, y_train, y_valid):
    model = LGBMClassifier()
    model.fit(X_train, y_train)
    pred = model.predict_proba(X_valid)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_valid, pred)
    auc_score = metrics.auc(fpr, tpr)
    return auc_score


def show_index(selected_features):
    return np.where(np.array(selected_features) == False)


def get_extra_tree(train, y):
    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(train, y)
    return clf


def get_feature_model(train, y, threshold=None):
    clf = get_extra_tree(train, y)
    # print(clf.feature_importances_)
    modelSelection = SelectFromModel(
        clf, prefit=True, max_features=8000, threshold=threshold)
    return modelSelection


# Hyperopt util functions
ITER = 50
STOP_ROUND = 5
MAX_EVALS = 500


def param_objective(params, train_set, valid_set):
    """Objective function for Gradient Boosting Machine Hyperparameter Tuning"""
    params['num_leaves'] = int(params['num_leaves'])
    params['subsample_for_bin'] = int(params['subsample_for_bin'])
    params['min_child_samples'] = int(params['min_child_samples'])
    # Use early stopping and evalute based on ROC AUC
    bst = lgb.train(params, train_set, ITER, valid_sets=valid_set,
                    early_stopping_rounds=STOP_ROUND)
    bst.save_model('model.txt', num_iteration=bst.best_iteration)

    # Extract the best score
    best_score = bst.best_score['valid_0']['auc']

    # Loss must be minimized
    loss = 1 - best_score

    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'status': STATUS_OK}


def find_best_params(train_set, valid_set):

    # Define the search space
    space = {
        'boosting_type': 'dart',
        'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
        'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
        'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
        'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
        'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
        'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
        'metric': 'auc'
    }
    bayes_trials = Trials()
    # Optimize
    objective = partial(
        param_objective, train_set=train_set, valid_set=valid_set)
    bestDict = fmin(fn=objective, space=space, algo=tpe.suggest,
                    max_evals=MAX_EVALS, trials=bayes_trials)
    best = space_eval(space, bestDict)
    best['num_leaves'] = int(best['num_leaves'])
    best['subsample_for_bin'] = int(best['subsample_for_bin'])
    best['min_child_samples'] = int(best['min_child_samples'])
    best['metric'] = 'auc'
    return best


def main():
    df_train = pd.read_csv('train_kaggle.csv')
    df_test = pd.read_csv('sample_solution.csv')
    y = df_train['label'].values

    # only col 18 is False
    selected_features = [True,  True,  True,  True,  True,  True,  True,  True,  True,
                         True, True,  True,  True,  True,  True,  True, True,  True,
                         False,  True,  True,  True,  True,  True,  True, True,  True,
                         True,  True,  True,  True,  True,  True,  True, True,  True,
                         True,  True,  True,  True]
    print('droping index:')
    print(show_index(selected_features))
    print('loading training data')
    XTrain = select_train(selected_features)
    print('loading testing data')
    XTest = select_test(selected_features)

    print('selecting features')
    modelSelection = get_feature_model(XTrain, y)
    XTrain = modelSelection.transform(XTrain)
    XTest = modelSelection.transform(XTest)

    print('getting validation sets')
    X_train, X_valid, y_train, y_valid = train_test_split(
        XTrain, y, test_size=0.2, random_state=42)

    print('validating model')
    calculate_auc(X_train, X_valid, y_train, y_valid)

    print('tuning')
    train_set = lgb.Dataset(X_train, y_train)
    valid_set = lgb.Dataset(X_valid, y_valid)
    best = find_best_params(train_set, valid_set)
    bestModel = lgb.train(
        best, train_set, ITER, valid_sets=valid_set, early_stopping_rounds=STOP_ROUND)

    print('training')
    best['boosting_type'] = 'dart'
    gb_clf = LGBMClassifier(boosting_type=best['boosting_type'],
                            num_leaves=best['num_leaves'],
                            learning_rate=best['learning_rate'],
                            subsample_for_bin=best['subsample_for_bin'],
                            min_child_samples=best['min_child_samples'],
                            reg_alpha=best['reg_alpha'],
                            reg_lambda=best['reg_lambda'],
                            colsample_bytree=best['colsample_bytree'])
    gb_clf.fit(XTrain, y)
    probs = gb_clf.predict_proba(XTest, num_iteration=bestModel.best_iteration)
    YTest = probs[:, 1]
    df_test['Predicted'] = YTest

    file_name = 'test.csv'
    print('saved result to ' + file_name)
    df_test.to_csv(file_name, index=False)

if __name__ == "__main__":
    main()
