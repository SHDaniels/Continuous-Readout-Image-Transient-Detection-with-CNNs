import numpy as np
import sys
import pylab as plt
import pywt
import pandas as pd

from argparse import ArgumentParser

from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from tqdm import tqdm
import matplotlib.cm as cm
import seaborn as sns

import crw_config as config
from utils import PR, predict_weighted, aggregate_rf_cv_weighted
from showlcv import plotlcv
import joblib

#from unidip import UniDip
#import unidip.dip as dip


def rffeatures_train(choosechip="05", runsdir=None,
                     thisfilter=None, run=None,
                     datetime=None, extension=None,
                     verbose=None, diagnostic_plots=None, savemodel=True):

  print("chip", choosechip)
  allchips = ["05"]#["04","05","06"]
  if choosechip is None:
    choosechip = config.choosechip
  if runsdir is None:
    runsdir = config.runsdir
  if thisfilter is None:
    thisfilter = config.thisfilter
  if run is None:
    run = config.run
  if datetime is None:
    datetime = config.datetime
  if verbose is None:
    verbose = config.VERBOSE
  if diagnostic_plots is None:
    diagnostic_plots = config.DIAGNOSTICPLOTS
  
  filedetails, lcvarrayfile, chipfilename, featuresfile, \
    predictionfile, rootdir = config.makenames(datetime, run, thisfilter, choosechip)
    
  NCl=  config.NCl
  ap_column = config.ap_column
  
  extension = config.extension
  building_trainingset = config.BUILDINGTRAININGSET
  wavelets = config.wavelets  
  wavelet = config.wavelet
  #'haar'  # Start with 'haar', then try 'db4'
  level = config.waveletlevel
  #5        # Choose an appropriate level (e.g., 5-8)    

  rfthreshold = config.rfthreshold

  
  labelsfile  = f"{rootdir}crwlabels_.csv"
  if wavelets:
    featuresfile = featuresfile.replace('crwfeatures', 'crwfeatures_wwvt')
    labelsfile  = f"{rootdir}crwlabels_wwvt_.csv"

  print(f"reading features from {featuresfile}")
  features = []
  features = [np.load(open(featuresfile.replace("05.", chip+"."), "rb"))
              for chip in allchips]
  labelsdf = pd.read_csv(labelsfile, index_col=0)
  labels = np.array([labelsdf.iloc[i]["0"] for i in range(labelsdf.shape[0])])
  label2num_dict = {l[0]:i for i,l in enumerate(labelsdf.values) if l[0] not in ["Hartigan's"] }
  label2num_dict["H"] = 2

  Nlc = len(features)
  Nfeatures = len(labels)


  tags = config.tags
  alltags = []
  for chips in allchips:
    print(filedetails.replace("_05", "_"+chips) )
    if filedetails.replace("_05", "_"+chips) in tags:
      print(len(tags[filedetails.replace("_05", "_"+chips)]))
      alltags.append(tags[filedetails.replace("_05", "_"+chips)])
  

  features = np.vstack([features[i] for i in range(len(allchips))])
  print("here", features.shape)
  message = f"this code expects {Nfeatures} with wavelets set to {wavelets}"
  assert features.shape[1]== Nfeatures, message

  features_scaled = preprocessing.scale(features, axis=0)
  features_scaled.std(axis=0)
  if diagnostic_plots:
    fig, axs = plt.subplots(2,1,figsize=(7, 7))
    axs = axs.ravel()
    axs[0].boxplot(features[tag > 0])
    axs[0].set_xticks(range(1, features.shape[1]+1),
                      labels)
    axs[0].set_xlabel('Feature Index')
    axs[0].set_ylabel('Value')
    axs[0].tick_params(axis='x', labelrotation=45)
    plt.grid(True)

    axs[1].boxplot(features_scaled[tag > 0])
    axs[1].set_xticks(range(1, features_scaled.shape[1]+1),
                      labels)
    axs[1].set_xlabel('Feature Index')
    axs[1].set_ylabel('Scaled Value')
    axs[1].tick_params(axis='x', labelrotation=45)  
    plt.grid(True)
    #plt.show()

  print(lcvarrayfile)
  #f = lcvarrayfile.replace('crwdata', 'crwrejected')
  #alltags = {lcvarrayfile.split("crwdata")[-1] : np.load(open(f, "rb"))}
  tag = np.hstack([t for t in alltags])
  print(tags, alltags)


  models, weights, skf = aggregate_rf_cv_weighted(features, tag)
  y_pred_weighted = predict_weighted(models, weights, features,
                                     aggregation='soft')

  if savemodel:
    #save the models
    print("saving model as ../models/crw_rf_models.joblib and ../models/crw_rf_weights.npy")
    import joblib
    # Assuming 'models' is the list of trained Random Forest models
    joblib.dump(models, '../models/crw_rf_models.joblib')
    np.save(f"../models/crw_rf_weights.npy", weights)
  

    print("Random Forest models saved successfully to 'crw_rf_models.joblib'")
    print("Random Forest weights  saved successfully to 'crw_rf_weights.joblib'")  

  finalpred = y_pred_weighted[:,1]>0.9
  # Initialize lists to store metrics for each fold
  accuracies = []
  precisions = []
  recalls = []
  f1_scores = []
  fold_importances = []

  # Iterate through the folds generated by StratifiedKFold
  for fold, (train_index, test_index) in enumerate(skf.split(features, tag)):
    print(f"\n--- Fold {fold + 1}/{skf.n_splits} ---")

    # Split data into training and testing sets for the current fold
    X_train_fold, X_test_fold = features[train_index], features[test_index]
    y_train_fold, y_test_fold = tag[train_index], tag[test_index]

    # Instantiate a RandomForestClassifier for the current fold
    model_fold = RandomForestClassifier(random_state=42,
                                        class_weight='balanced_subsample',
                                        max_depth=3)
    
    # Train the model
    model_fold.fit(X_train_fold, y_train_fold)


    fold_importances.append(model_fold.feature_importances_)
    
    
    # Make predictions on the test data
    y_pred_fold = model_fold.predict(X_test_fold)
    y_pred_train_fold= model_fold.predict(X_train_fold)    
    

    # Calculate metrics for the current fold
    accuracy_fold = accuracy_score(y_test_fold, y_pred_fold)
    precision_fold = precision_score(y_test_fold, y_pred_fold, zero_division=0)
    recall_fold = recall_score(y_test_fold, y_pred_fold, zero_division=0)
    f1_fold = f1_score(y_test_fold, y_pred_fold, zero_division=0)
    print(f"overfit train: {accuracy_score(y_train_fold, y_pred_train_fold):.2f} "
          f"test: {accuracy_fold}:.2f")

    # Append metrics to their respective lists
    accuracies.append(accuracy_fold)
    precisions.append(precision_fold)
    recalls.append(recall_fold)
    f1_scores.append(f1_fold)

    print(f"  Accuracy: {accuracy_fold:.4f}")
    print(f"  Precision: {precision_fold:.4f}")
    print(f"  Recall: {recall_fold:.4f}")
    print(f"  F1-Score: {f1_fold:.4f}")

  avg_importances = np.mean(fold_importances, axis=0)
  std_importances = np.std(fold_importances, axis=0)
  
  # Sort features by importance
  indices = np.argsort(avg_importances)[::-1]
  
  # Create plot
  plt.figure(figsize=(12, 8))
  
  # Plot feature importances with error bars
  positions = np.arange(len(indices))
  plt.bar(positions, avg_importances[indices], 
          yerr=std_importances[indices], capsize=5, alpha=0.7,
          color='skyblue', edgecolor='navy')
  
  plt.xticks(positions, [labels[i] for i in indices], rotation=45, ha='right')
  plt.ylabel('Feature Importance')
  plt.title('Aggregated Feature Importance from 5-Fold Random Forest')
  plt.grid(axis='y', alpha=0.3)
  plt.tight_layout()
  plt.show()
    
    # Make predictions on the test data
  print("\nCross-validation complete. Metrics collected for all folds.")

  # Convert lists to NumPy arrays for easier calculation
  accuracies = np.array(accuracies)
  precisions = np.array(precisions)
  recalls = np.array(recalls)
  f1_scores = np.array(f1_scores)

  # Calculate mean and standard deviation for each metric
  mean_accuracy = np.mean(accuracies)
  std_accuracy = np.std(accuracies)

  mean_precision = np.mean(precisions)
  std_precision = np.std(precisions)

  mean_recall = np.mean(recalls)
  std_recall = np.std(recalls)

  mean_f1 = np.mean(f1_scores)
  std_f1 = np.std(f1_scores)

  # Report the aggregated metrics
  print("\n--- Aggregated Cross-Validation Metrics ---")
  print(f"  Mean Accuracy: {mean_accuracy:.4f} (Std: {std_accuracy:.4f})")
  print(f"  Mean Precision: {mean_precision:.4f} (Std: {std_precision:.4f})")
  print(f"  Mean Recall: {mean_recall:.4f} (Std: {std_recall:.4f})")
  print(f"  Mean F1-Score: {mean_f1:.4f} (Std: {std_f1:.4f})")


  # Get probabilities for the positive class (class 1)
  y_probs = y_pred_weighted[:, 1]
  
  fpr, tpr, thresholds = roc_curve(tag, y_probs)
  
  auc_score = auc(tpr, fpr)

  if diagnostic_plots:
    # Plot the Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='b', label=f'ROC Curve (AUC = {auc_score:.2f})')
    plt.xlabel("False Positives")
    plt.ylabel("True Positives")
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

  if savemodel:
    # Plot the Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='b', label=f'ROC Curve (AUC = {auc_score:.2f})')
    plt.xlabel("False Positives")
    plt.ylabel("True Positives")
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{rootdir}/plots/rf_cmplot{filedetails}.pdf")
  
  y_pred_binary = (y_pred_weighted[:, 1] > rfthreshold).astype(int)
  if diagnostic_plots:
    cm = confusion_matrix(tag, y_pred_binary)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show() 

  print(f"wrong classifications: ")
  np.save("tmp",  y_pred_binary)
  for i in np.where(~( y_pred_binary == tag))[0]:
    print(i)
  return  y_pred_binary



  
if __name__ == "__main__":

  rffeatures_train()
  




