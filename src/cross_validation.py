from sklearn.model_selection import GridSearchCV

def gridsearchcv(estimator, param_grid, scoring):
    return GridSearchCV(estimator=estimator, param_grid=param_grid, scoring=scoring)