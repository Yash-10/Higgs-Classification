"""Helpful functionalities."""

import numpy as np
import xgboost as xgb


def prepare_df(df, which_feats="full"):
  """Returns a new dataset with only the required features.

  Parameters:
  -----------
  df: ~pd.DataFrame
    Dataset, as is.
  which_feats: ~str
    Which features to use for training the model, defaults to "full".
    Other options: "high_level", or "low_level".
  
  Returns:
  --------
  new_df: The dataset with the required features.

  """
  cols = list(df.columns)
  cols = list(map(lambda name: name.strip(), cols))
  if which_feats == "low_level":
    return df[["label"] + cols[1:22]]
  elif which_feats == "high_level":
    return df[["label"] + cols[22:]]
  else:
    return df


def split_data(df, which_feats="full", test_frac=0.3):
  """Split data into train and test sets.

  Parameters
  ----------
  df: ~pd.DataFrame
    Dataset (with labels and features) output from `prepare_df`.
  which_feats: ~str
    Which features to use while training, defaults to "full_level".
  
  Returns
  -------
  train, test: The train and test data.

  """
  num_rows = len(df.index)
  train_df = df[:int((1-test_frac)*num_rows)]
  test_df = df[int(test_frac*num_rows):]

  feature_names = df.columns[1:]

  train = xgb.DMatrix(data=train_df[feature_names], label=train_df.label.cat.codes,
                      feature_names=feature_names)
  test = xgb.DMatrix(data=test_df[feature_names], label=test_df.label.cat.codes,
                     feature_names=feature_names)
  
  return train, test


def train_bdt(train_data, test_data, params, num_trees=100, which_feats="full"):
  """Train the model.

  Parameters
  ----------
  train_data, test_data: ~xgboost.core.DMatrix
    The train and test data output by `split_data`
  params: ~dict
    Dictionary consisting of parameters for the model.
  num_trees: ~float
    Number of trees to use in the model, defaults to 100.
  which_feats: ~str
    Which features to use while training, defaults to "full_level".
  
  Returns
  -------
  booster: The trained model.
  predictions: Predictions from the trained model.
  evaluation: Some statistics on evaluation of the model.

  Notes
  -----
  The model is saved at `model_path` after training is done.

  """
  num_trees = num_trees  # Number of trees to make

  # Training
  booster = xgb.train(params, train_data, num_boost_round=num_trees)
  model_path = f"/content/xgboost_train_{which_feats}.txt"
  booster.save_model(fname=model_path)
  evaluation = booster.eval(test_data)
  predictions = booster.predict(test_data)

  return booster, predictions, evaluation


def plot_predictions(predictions, test):
  """Plot a histogram of predictions by the model, both combined and class wise.

  Parameters
  ----------
  predictions: ~np.array
    Predictions from the model
  test: ~xgboost.core.DMatrix
    The testing dataset

  Returns
  -------
  Prediction plot

  """
  import matplotlib.pyplot as plt

  fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(30, 10));

  hist_style_all = dict(histtype="step", color="darkgreen", label="All events")
  hist_style_sig = dict(histtype="step", color="midnightblue", label="signal")
  hist_style_bkg = dict(histtype="step", color="firebrick", label="background")

  ax1.hist(predictions, bins=np.linspace(0, 1, 50), **hist_style_all);

  ax1.set_xlabel("Prediction from BDT", fontsize=16);
  ax1.set_ylabel("Events", fontsize=16);
  ax1.set_title("All events (combined)");
  ax1.legend(frameon=False);

  ax2.hist(predictions[test.get_label().astype(bool)], bins=np.linspace(0, 1, 50),
          **hist_style_sig);
  ax2.hist(predictions[~(test.get_label().astype(bool))], bins=np.linspace(0, 1, 50),
          **hist_style_bkg);

  ax2.set_xlabel("Prediction from BDT", fontsize=16);
  ax2.set_ylabel("Events", fontsize=16);
  ax2.set_title("Both events (individually)")
  ax2.legend(frameon=False);


def plot_roc(predictions, test):
  import matplotlib.pyplot as plt
  try:
    from plot_metric.functions import BinaryClassification
  except ImportError:
    print("Error while import! Cannot import `BinaryClassification from` `plot_metric.functions`. "
          "Use 'pip install plot-metric' to solve this error.")
  
  # Visualisation with plot_metric
  bc = BinaryClassification(test.get_label(), predictions, labels=["Class 1", "Class 2"])

  # Figures
  plt.figure(figsize=(10,10))
  bc.plot_roc_curve()
  plt.show()
