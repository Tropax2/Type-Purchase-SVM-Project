# Orange Juice Purchase Classification
This project compares support vector machine classifiers to predict whether a customer purchases Citrus Hill (`CH`) or Minute Maid (`MM`) orange juice based on pricing, discounts, store information, and customer loyalty variables.

## Project structure

- `src/` - source code for models and evaluations;
- `report_and_results/` - detailed report of the results obtained;

## Dataset
The dataset contains 1,070 purchase observations where the response variable is `Purchase`, indicating whether the customer bought Citrus Hill (`CH`) or Minute Maid (`MM`) orange juice.

Before modeling, the data is processed as follows:

- The CSV file is loaded into a pandas DataFrame.
- Rows with missing values are removed.
- Numerical features are standardized using `StandardScaler`.
- Categorical features are one-hot encoded using `OneHotEncoder`.
- All preprocessing is handled through a `ColumnTransformer`.
- Transformations are fit on the training set and then applied to the test set to avoid data leakage.

## Results summary
Three support vector-based models were evaluated on this classification task:

### Linear Support Vector Classifier
A linear support vector classifier with `C = 0.01` achieved about **80% accuracy**.

Confusion matrix:

|                | Predicted Negative | Predicted Positive |
|----------------|--------------------|--------------------|
| Actual Negative | 115               | 15                 |
| Actual Positive | 26                | 58                 |

Cross-validation was also performed over the values `C = 0.001, 0.01, 0.5, 1, 2, 5, 10`. The best result was obtained with `C = 0.5`, though without a meaningful improvement in performance.

### Radial Kernel SVM
The radial kernel SVM achieved the best overall performance, with **81.7% accuracy**.

Confusion matrix:

|                | Predicted Negative | Predicted Positive |
|----------------|--------------------|--------------------|
| Actual Negative | 116               | 14                 |
| Actual Positive | 25                | 59                 |

### Polynomial Kernel SVM
The polynomial kernel SVM with degree 2 performed worse than the other models, achieving **78.9% accuracy**.

Confusion matrix:

|                | Predicted Negative | Predicted Positive |
|----------------|--------------------|--------------------|
| Actual Negative | 117               | 13                 |
| Actual Positive | 32                | 52                 |

### Conclusion
Support vector methods performed well overall, with classification accuracy close to 80% across all tested models. The radial kernel SVM produced the best result, although the improvement over the linear classifier was small.

This suggests that the relationship between the predictors and the purchase decision may include some nonlinearity, but not one that is well captured by a degree-2 polynomial kernel. Because the radial model only slightly outperformed the linear model, the linear SVC may still be a strong option when simplicity and interpretability are preferred.

## How to run


### 1) Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\Activate.ps1 # Windows PowerShell
```

### 2) Install Dependencies
```bash 
pip install -r requirements.txt
```

### 3) Run the Project 
```bash
python src/main.py
```