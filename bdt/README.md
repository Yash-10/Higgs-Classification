## Boosted decision tree implementation

- `xgboost_train_low_level.txt`, `xgboost_train_high_level.txt`, and `xgboost_train_full.txt` (inside the folder `bdt_models/`, are the saved XGBoost models trained with low level, high level and all features respectively with the following parameters:

```
{'colsample_bytree': 0.8,
 'eta': 0.01,
 'eval_metric': 'rmse',
 'max_depth': 12,
 'min_child_weight': 13,
 'objective': 'binary:logistic',
 'subsample': 0.8}
 ```
 - Training time was ~1hr 5min for low level features, ~15-20 min for high-level features, and slightly more than 1hr for all features. The number of trees used is 200 in all the cases.

The trained models (in `.txt` format) can be directly loaded using [XGBoost](https://xgboost.readthedocs.io/en/latest/), for example:

```py
model = xgb.Booster()
model.load_model("xgboost_train_full.txt")
```
and then it can be used for prediction:

```py
model.predict(<test_data>)
```

---

 **Results**

Low level features:
- AUC = 0.73
- RMSE = 0.465280

![Low level ROC curve](images/low_level_ROC.png)

High level features:
- AUC = 0.79
- RMSE = 0.449697

![High level ROC curve](images/high_level_ROC.png)

All features:
- AUC = 0.82
- RMSE: 0.437461

![All features](images/all_ROC.png)