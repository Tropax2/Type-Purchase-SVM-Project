## The Dataset

The data contains 1070 purchases where the customer either purchased Citrus Hill or Minute Mais Orange Juice. A number of characteristics of the customer and product are recorded:

- `Purchase`: A factor with levels 'CH' and 'MM' indicating whether the customer purchased Citrus Hill or Minute Maid Orange Juice.
- `WeekofPurchase`: Week of purchase.
- `StoreId`: Store ID.
- `PriceCH`: Price charged for CH.
- `PriceMM`: Price charged for MM.
- `DiscCH`: Discount offered for CH.
- `DiscMM`: Discount offered for MM.
- `SpecialCH`: Indicator of special on CH.
- `SpecialMM`: Indicator of special on MM.
- `LoyalCH`: Customer brand loyalty for CH.
- `SalePriceMM`: Sale price for MM.
- `SalePriceCH`: Sale price for CH.
- `PriceDiff`: Sale price of MM less sale price of CH.
- `Store7`: A factor with levels 'No' and 'Yes' indicating whether the sale is at Store 7.
- `PctDiscMM`: Percentage discount for MM.
- `PctDiscCH`: Percentage discount for CH.
- `ListPriceDiff`: List price of MM less list price of CH.
- `Store`: Which of the 5 possible stores the sale occurred at.

The target variable is `Purchase`, while the remaining are predictor variables.

The data comes from Stine, Robert A., Foster, Dean P., Waterman, Richard P. Business Analysis Using Regression (1998). Published by Springer.

## Data Processing

The CSV is loaded into a pandas DataFrame and rows with missing values are dropped. Numerical predictors are standardised using `StandardScaler`, while categorical predictors are one-hot encoded using `OneHotEncoder`. The transformations were applied using `ColumnTransformer`. The preprocessing steps fit and transform the training set, and are then applied to the test set to avoid data leakage.

## Models Used and Results Obtained

### Support Vector Classifier

We first fit a classical support vector classifier with a regularization parameter `C = 0.01`. The method predicted correctly around 80% of the observations with the following confusion matrix:

|                | Predicted Negative | Predicted Positive |
|----------------|--------------------|--------------------|
| Actual Negative | 115               | 15                 |
| Actual Positive | 26                | 58                 |

Following this, we performed cross-validation, with the values o `C` `0.001, 0.01, 0.5, 1, 2, 5, 10`; and the best result comes for `C = 0.5`, but with no significant improvement.

We then fit a support vector machine with radial kernel and the default value of `gamma` and with `C = 1`. The results obtained improved slightly, now with a 81.7% correct classification rate and the following confusion matrix:

|                | Predicted Negative | Predicted Positive |
|----------------|--------------------|--------------------|
| Actual Negative | 116               | 14                 |
| Actual Positive | 25                | 59                 |

Lastly, we did the same but with a polynomial kernel of degree 2, and the results dropped to a 78.9% correct classification rate with the following confusion matrix:

|                | Predicted Negative | Predicted Positive |
|----------------|--------------------|--------------------|
| Actual Negative | 117               | 13                 |
| Actual Positive | 32                | 52                 |

## Conclusion 

Overall, the support vector methods produced solid predictive performance on this dataset, with classification accuracy close to 80% across all specifications. Among the models considered, the radial kernel SVM delivered the best result, achieving an accuracy of 81.7%, although the improvement over the linear support vector classifier was small. Cross-validation on the regularization parameter C lead to no improvement in performance. 

The results also show that model choice matters: while the radial kernel slightly improved predictive accuracy, the polynomial kernel of degree 2 performed worse than both the linear and radial versions. This suggests that the boundary obtained from the predictors to the `Purchase` variable may contain some nonlinearity, but not one that is well captured by a polynomial.

In conclusion, the radial SVM would be the preferred model among those tested, as it achieved the highest classification rate while remaining only slightly more complex than the linear alternative. At the same time, because the gain over the linear classifier is small, the linear SVC may still be attractive if interpretability and simplicity are priorities.
