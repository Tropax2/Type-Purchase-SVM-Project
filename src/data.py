import pandas as pd 
from sklearn.model_selection import train_test_split

# import the csv file into a pandas pd and remove any rows with null entries 
def csv_to_pd(path : str) -> pd.DataFrame:
    oj = pd.read_csv(path)
    oj = oj.dropna()

    return oj

# split the dataset into training and test set
def make_splits(X, y, test_size, random_state, shuffle):

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle
    )
    

