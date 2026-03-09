from sklearn.metrics import confusion_matrix
import numpy as np
import data, processing, cross_validation
import models 

# this is the main function

def main():

    # Transform the CSV file into a pandas df
    path = '/Users/afonsolopes/SVM-classification/OJ.csv'
    oj = data.csv_to_pd(path)

    # Define the predictors and the response 
    X = oj.drop(columns="Purchase")
    y = oj["Purchase"]

    # Separate the predictors into categorical and numerical
    categorical = ["StoreID", "SpecialCH", "SpecialMM", "Store7", "STORE"]
    numeric = [
        "WeekofPurchase", "PriceCH", "PriceMM", "DiscCH", "DiscMM",
        "LoyalCH", "SalePriceMM", "SalePriceCH", "PriceDiff", "PctDiscMM",
        "PctDiscCH", "ListPriceDiff",
    ]

    # Split the dataset into a training and test set
    X_train, X_test, y_train, y_test = data.make_splits(
        X = X,
        y = y,
        test_size = 0.2,
        random_state = 42,
        shuffle = True
    )

    # Aplly the transformations to the columns and corresponding training set
    ct = processing.column_transformer(categorical_predictors=categorical, numerical_predictors=numeric)
    X_train_transformed = ct.fit_transform(X_train)
    X_test_transformed = ct.transform(X_test)

    # Fit the SVC 
    clf = models.svc(C = 0.05, kernel = "linear")
    clf.fit(X_train_transformed, y_train)

    # Prediction and confusion matrix
    y_pred = clf.predict(X_test_transformed)
    m = confusion_matrix(y_test, y_pred)
    
    # Perform CV to find the best value of C
    cv = cross_validation.GridSearchCV(
        estimator = models.svc(C=1, kernel='linear'),
        param_grid={"C": [0.001, 0.01, 0.5, 1, 2, 5, 10]},
        scoring="accuracy"
    )
    cv.fit(X_train_transformed, y_train)
    print(cv.best_params_)

    # Make the fit and prediction with a radial kernel

    # Fit the SVC 
    clf = models.svc(C = 1, kernel = "rbf", gamma='scale')
    clf.fit(X_train_transformed, y_train)

    # Prediction and confusion matrix
    y_pred = clf.predict(X_test_transformed)
    m = confusion_matrix(y_test, y_pred)

     # Fit the SVC 
    clf = models.svc(C = 1, kernel = "poly", degree=2)
    clf.fit(X_train_transformed, y_train)

    # Prediction and confusion matrix
    y_pred = clf.predict(X_test_transformed)
    m = confusion_matrix(y_test, y_pred)
    print(m)
    print(np.mean(y_pred == y_test))







if __name__ == "__main__":
    main()