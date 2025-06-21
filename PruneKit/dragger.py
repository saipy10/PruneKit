import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score


def LinPruneReg(dataframe, target_column_name):
    X = dataframe.drop(columns=[target_column_name])
    y = dataframe[target_column_name]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    draggers = []
    
    is_further_removal_needed = True
    
    while is_further_removal_needed:
        # Remove already dropped features
        X_train_temp = X_train.drop(columns=draggers)
        X_test_temp = X_test.drop(columns=draggers)

        # Baseline model
        base_model = LinearRegression()
        base_model.fit(X_train_temp, y_train)
        base_predictions = base_model.predict(X_test_temp)
        baseline_score = mean_squared_error(y_test, base_predictions)

        feature_to_be_removed = ""
        min_score = baseline_score

        for column in X_train_temp.columns:
            # Drop one more column
            X_train_iter = X_train_temp.drop(columns=[column])
            X_test_iter = X_test_temp.drop(columns=[column])

            model_iter = LinearRegression()
            model_iter.fit(X_train_iter, y_train)
            predictions_iter = model_iter.predict(X_test_iter)
            current_score = mean_squared_error(y_test, predictions_iter)

            if current_score < min_score:
                min_score = current_score
                feature_to_be_removed = column

        if min_score < baseline_score:
            draggers.append(feature_to_be_removed)
        else:
            is_further_removal_needed = False

    return draggers


def LinPruneCat(dataframe, target_column_name):
    X = dataframe.drop(columns=[target_column_name])
    y = dataframe[target_column_name]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    draggers = []
    
    is_further_removal_needed = True
    
    while is_further_removal_needed:
        # Remove already dropped features
        X_train_temp = X_train.drop(columns=draggers)
        X_test_temp = X_test.drop(columns=draggers)

        # Baseline model
        base_model = LogisticRegression(max_iter=10000)
        base_model.fit(X_train_temp, y_train)
        base_predictions = base_model.predict(X_test_temp)
        baseline_score = accuracy_score(y_test, base_predictions)

        feature_to_be_removed = ""
        max_score = baseline_score

        for column in X_train_temp.columns:
            # Drop one more column
            X_train_iter = X_train_temp.drop(columns=[column])
            X_test_iter = X_test_temp.drop(columns=[column])

            model_iter = LogisticRegression()
            model_iter.fit(X_train_iter, y_train)
            predictions_iter = model_iter.predict(X_test_iter)
            current_score = accuracy_score(y_test, predictions_iter)

            if current_score > max_score:
                max_score = current_score
                feature_to_be_removed = column

        if max_score > baseline_score:
            draggers.append(feature_to_be_removed)
        else:
            is_further_removal_needed = False

    return draggers