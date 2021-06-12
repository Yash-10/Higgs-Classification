## Boosted decision tree implementation

- `xgboost_train_low_level.txt` and `xgboost_train_high_level.txt` are the saved XGBoost model trained with low level and high level features with the following parameters:

```
{'colsample_bytree': 0.8,
 'eta': 0.01,
 'eval_metric': 'rmse',
 'max_depth': 12,
 'min_child_weight': 13,
 'objective': 'binary:logistic',
 'subsample': 0.8}
 ```
 - Training time was ~1hr 5min for low level features and ~15-20 min for high-level features. The number of trees used is 200.

 **Results**

Low level features:
- AUC = 0.73
- RMSE = 0.465280

High level features:
- AUC = 0.79
- RMSE = 0.449697
