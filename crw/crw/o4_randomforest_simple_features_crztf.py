import matplotlib
matplotlib.use('WebAgg')
matplotlib.rcParams['webagg.open_in_browser'] = False
import matplotlib.pyplot as plt

import numpy as np
import sys
import gc

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
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import matplotlib.cm as cm
import seaborn as sns
import corner
import crw_config as config
from utils import PR, predict_weighted, aggregate_rf_cv_weighted, getfromlcvname 
from showlcv import plotlcv
import joblib

REVIEWLABELS = False
TEST = False
#plt.ion()

#from unidip import UniDip
#import unidip.dip as dip


def rffeatures_predict(choosechip=None, runsdir=None,
                       thisfilter=None, run=None,
                       datetime=None, extension=None, 
                       lightcurve=None, verbose=None):
  
  if verbose is None:
      verbose = config.VERBOSE

  if not lightcurve is None: 
    datetime, run, thisfilter, choosechip, seq, x, y, a = getfromlcvname(lightcurve)

  else:
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

  filedetails, lcvarrayfile, chipfilename, featuresfile, predictionfile, rootdir = config.makenames(datetime, run, thisfilter, choosechip)
  rfthreshold = config.rfthreshold
  
  NCl =  config.NCl
  ap_column = config.ap_column  
  extension = config.extension
  building_trainingset = config.BUILDINGTRAININGSET
  wavelets = config.wavelets  
  wavelet = config.wavelet
  #'haar'  # Start with 'haar', then try 'db4'
  level = config.waveletlevel
  #5        # Choose an appropriate level (e.g., 5-8)    

  labelsfile  = f"{rootdir}crwlabels_.csv"
  if wavelets:
    featuresfile = featuresfile.replace('crwfeatures', 'crwfeatures_wwvt')
    labelsfile  = f"{rootdir}crwlabels_wwvt_.csv"      

  print(f"reading features from {featuresfile}, labels from {labelsfile}")
  features = np.load(open(featuresfile, "rb"))
  labelsdf = pd.read_csv(labelsfile, index_col=0)
  labels = np.array([labelsdf.iloc[i]["0"] for i in range(labelsdf.shape[0])])
  
  Nlc = len(features)
  Nfeatures = len(labels)

  label2num_dict = {l[0]:i for i,l in enumerate(labelsdf.values) if l[0] not in ["Hartigan's"] }
  label2num_dict["H"] = 2

  message = f"this code expects {Nfeatures} with wavelets set to {wavelets}"
  assert features.shape[1]== Nfeatures, message

  rfmodels = joblib.load(config.ROOT + "/models/crw_rf_models.joblib")
  rfweights = np.load(open(config.ROOT + "/models/crw_rf_weights.npy", "rb"))
  y_pred_weighted = predict_weighted(rfmodels, rfweights, features, aggregation='SOFT')
  predictionsRF = (y_pred_weighted[:,1] > rfthreshold).astype(int)

  np.save(predictionfile, predictionsRF)

#  tags = config.tags
#  if  filedetails in tags:
#    tag = tags[filedetails]
#    print(tag.shape)
#    print(predictionsRF.shape)
#    print(classification_report(tag, predictionsRF, target_names=("Valid", "Invalid")))
  print(f"saved predictions in crwprediction{filedetails}.npy")
  return predictionsRF


      
def rffeatures_fromfile_train(filename=None, csv=None,
                     verbose=None, diagnostic_plots=None, savemodel=False):

  path = config.RFPATH
  if filename: 
    lcvsfile = filename
    tagfile = filename.replace(".npy", f"_reviewed.npy")
    lcvslist = np.load(path + "/" + lcvsfile)
    tags = np.load(path + "/" + tagfile)
    tagselect = np.zeros(len(tags), bool) * False

  elif csv:
    csvfile = csv
    labelsdf = pd.read_csv(path + "/" + csv, index_col=0).dropna()
#    select = labelsdf["majority"]
#    print(labelsdf[select].head())
    lcvslist = np.array(labelsdf.index)#[select]
    #tags = (labelsdf.loc[select, ["0","1","2"]].sum(axis=1)>0).astype(int)
    tags = labelsdf["label"].astype(int) #> 0
    tagselect = np.zeros(len(tags), bool) * False
    print(f"labels and counts {list(zip(np.unique(tags, return_counts=True)))}")

  NCl =  config.NCl
  ap_column = config.ap_column
  
  extension = config.extension
  building_trainingset = config.BUILDINGTRAININGSET
  wavelets = config.wavelets  
  wavelet = config.wavelet
  #'haar'  # Start with 'haar', then try 'db4'
  level = config.waveletlevel
  #5        # Choose an appropriate level (e.g., 5-8)    

  rfthreshold = config.rfthreshold
  Nlc = 0  
  features = []
  labels = []
  
  rootdir = config.ROOT
  if wavelets:
      labelsfile  = f"{rootdir}crwlabels_wwvt_.csv"
  else:
    labelsfile  = f"{rootdir}crwlabels_.csv"
  #wwvt
  print(f"reading labels from {labelsfile}")
  labelsdf_here = pd.read_csv(labelsfile, index_col=0)
  labels = np.array([labelsdf_here.iloc[i]["0"] for i in range(labelsdf_here.shape[0])])
  label2num_dict_here = {l[0]:i for i,l in enumerate(labelsdf_here.values) 
                         if l[0] not in ["Hartigan's"]}
  label2num_dict_here["H"] = 2
  Nfeatures = len(labels)
  print(f"training with {Nfeatures} features")
  #print(len(lcvslist))

  if TEST:
    lcvslist = lcvslist[:50]
    tags = tags[:50]
    print("fraction of positive tags", (tags > 0).sum() / 50)
    tagselect = tagselect[:50]

  for i,lcv in enumerate(lcvslist):

    print(f"\n{i}:{lcv}")
    datetime, run, thisfilter, choosechip, seq, x, y, a = getfromlcvname(lcv)
    #print(x, y, a)
    #if i < 145:
    #  continue

    if verbose:
      print(choosechip, thisfilter, run,datetime, x, y)
    #choosechip, run, thisfilter, datetime = lcv.split("_")
    #lightcurve_ztf_20221020141806_000686_zi_c14_x_0003_4_0708_0764.csv'

    try:
      pos = np.load(config.ROOT + "/" + f"crwpos_{datetime}_{run}_{thisfilter}_{choosechip}.npy") 
    except FileNotFoundError:
      print("missing position!")
      continue

    filedetails, lcvarrayfile, chipfilename, featuresfile, predictionfile, rootdir = config.makenames(datetime, run, thisfilter, choosechip)
    featuresfile = featuresfile.replace("crwfeatures_", "crwfeatures_wwvt_")
    #print(f"reading features from {featuresfile}")
    #print(f"reading chipfilename {chipfilename}")
    features_here = np.load(open(featuresfile, "rb"))
      
    #print(np.unique(pos[:,3]))
    select = np.where((pos[:,0] == x) * (pos[:,1] == y) * (pos[:,3] == a))[0]
    #print(select, seq)
    #print(len(select))
    seqindx = pos[select][:,2] == seq
    
    assert (pos[select][seqindx] == np.array([x,y,seq,a])).all(), "something wrong with the selection"
    
    tagselect[i] = True
    #print(tagselect[i])
    features_here = features_here[select][seqindx]
    assert len(features_here) == 1, "select did not work"
    Nlc += len(features_here)

    message = f"this code expects {Nfeatures} with wavelets set to {wavelets}," 
    message += "got {np.array(features).shape[1]}"

    assert features_here.shape[1] == Nfeatures, message
    features.append(features_here)

    #print(lcvarrayfile)
    theselcvs = np.load(open(lcvarrayfile, "rb"))
    
    if REVIEWLABELS:# and i>135:
      color = 'r' if tags[i] == 1 else 'k'
      #print(pd.read_csv(config.LCVSPATH + "/" + lcvslist[i])[["flux_hap_2"]])
      ax = pd.read_csv(config.LCVSPATH + "/" + lcvslist[i])[["flux_hap_2"]].plot()
      ax.plot(theselcvs[select][seqindx][:,config.ap_column], color=color, alpha=0.5)
      plt.title(f"label {tags[i]}")

      if not i%20:
        plt.show()
        plt.close('all')
        gc.collect()

  if REVIEWLABELS:
    plt.show()
    #plt.show()
    plt.close('all')

  print("labels vs usable labels", len(tags), tagselect.sum())

  features = np.vstack(features)
  #labels = np.vstack(labels)
  tags = tags[tagselect]

  print("arrays shapes:", features.shape, labels.shape, tags.shape)
  #print(lcvslist[tagselect])
  features_scaled = preprocessing.scale(features, axis=0)
  features_scaled.std(axis=0)

  if diagnostic_plots:
    print("plottinghere")
    
    # Create figure with subplots for original features
    fig1, axs = plt.subplots(2, 1, figsize=(14, 10), num=1)
    
    # Prepare data for split violin plot (original features)
    # Create a DataFrame for easier plotting
    data_list = []
    for i, label in enumerate(labels):
        # Add tagged data (tags > 0)
        for val in features[tags > 0, i]:
            data_list.append({'Feature': label, 'Value': val, 'Group': 'bad lcv'})
        # Add non-tagged data (tags == 0)
        for val in features[tags == 0, i]:
            data_list.append({'Feature': label, 'Value': val, 'Group': 'good lcv'})
    
    df = pd.DataFrame(data_list)
    print(df.Group.unique())
    # Split violin plot for original features
    sns.violinplot(x='Feature', y='Value', hue='Group', data=df, 
                   split=True, ax=axs[0], palette=['blue', 'orange'])
    axs[0].set_xlabel('Feature Index')
    axs[0].set_ylabel('Value')
    axs[0].tick_params(axis='x', labelrotation=45)
    axs[0].grid(True, alpha=0.3)
    axs[0].set_yscale('log')
    axs[0].set_title('Original Features')
    
    # Prepare data for scaled features
    data_list_scaled = []
    for i, label in enumerate(labels):
        for val in features_scaled[tags > 0, i]:
            data_list_scaled.append({'Feature': label, 'Value': val, 'Group': 'bad lcv'})
        for val in features_scaled[tags == 0, i]:
            data_list_scaled.append({'Feature': label, 'Value': val, 'Group': 'good lcv'})
    
    df_scaled = pd.DataFrame(data_list_scaled)
    
    # Split violin plot for scaled features
    sns.violinplot(x='Feature', y='Value', hue='Group', data=df_scaled,
                   split=True, ax=axs[1], palette=['blue', 'orange'])
    axs[1].set_xlabel('Feature Index')
    axs[1].set_ylabel('Scaled Value')
    axs[1].tick_params(axis='x', labelrotation=45)
    axs[1].grid(True, alpha=0.3)
    axs[1].set_yscale('log')
    axs[1].set_title('Scaled Features')
    
    plt.tight_layout()

    plt.show()


    #figure = corner.corner(features[tags==0], axes_scale=['log'] * Nfeatures, labels=labels)
    #figure = corner.corner(features[tags==1], axes_scale=
    #['log'] * Nfeatures, labels=labels, fig=figure, color='r')    
    log_data_0 = np.log10(features[tags==0] - np.min(features[tags==0], axis=0) +1e-10)
    log_data_1 = np.log10(features[tags==1] - np.min(features[tags==1], axis=0) +1e-10)

    if 1:
#    try:
      # Plot with linear axes (data is now in log space)
      figure = corner.corner(log_data_0, labels=labels)
      figure = corner.corner(log_data_1, labels=labels, fig=figure, color='r')
#    except:


  features_train, features_test, tags_train, tags_test, lcvs_train, lcvs_test  = train_test_split(features, tags, 
                                                                                                  lcvslist[tagselect],
                                                                                                  test_size=0.2, 
                                                                                                  random_state=302, 
                                                                                                  shuffle=True)

  models, weights, skf = aggregate_rf_cv_weighted(features_train, tags_train)
  #print(len(models))
  if savemodel:
    #save the models
    print("saving model as ../models/crw_rf_models.joblib and ../models/crw_rf_weights.npy")
    import joblib
    # Assuming 'models' is the list of trained Random Forest models
    joblib.dump(models, f'{config.ROOT}/models/crw_rf_models.joblib')
    np.save(f"{config.ROOT}/models/crw_rf_weights.npy", weights)
  

    print("Random Forest models saved successfully to 'crw_rf_models.joblib'")
    print("Random Forest weights  saved successfully to 'crw_rf_weights.joblib'")  

  # Initialize lists to store metrics for each fold
  accuracies = []
  precisions = []
  recalls = []
  f1_scores = []
  fold_importances = []

  # Iterate through the folds generated by StratifiedKFold
  for fold, (train_index, test_index) in enumerate(skf.split(features_train, tags_train)):
    print(f"\n--- Fold {fold + 1}/{skf.n_splits} ---")

    # Split data into training and testing sets for the current fold
    X_train_fold, X_test_fold = features_train[train_index], features_train[test_index]
    y_train_fold, y_test_fold = tags_train[train_index], tags_train[test_index]

    # Instantiate a RandomForestClassifier for the current fold
    #model_fold = RandomForestClassifier(random_state=42, 
    #                                    class_weight='balanced_subsample', max_depth=3)
    model_fold = models[fold]
    ## Train the model
    #model_fold.fit(X_train_fold, y_train_fold)

    fold_importances.append(model_fold.feature_importances_)
    
    
    # Make predictions on the test data
    y_pred_fold = model_fold.predict(X_test_fold)
    y_pred_train_fold= model_fold.predict(X_train_fold)    
    y_pred_unseen_fold= model_fold.predict(features_test)    
    

    # Calculate metrics for the current fold
    accuracy_fold = accuracy_score(y_test_fold, y_pred_fold)
    precision_fold = precision_score(y_test_fold, y_pred_fold, zero_division=0)
    recall_fold = recall_score(y_test_fold, y_pred_fold, zero_division=0)
    f1_fold = f1_score(y_test_fold, y_pred_fold, zero_division=0)
    print(f"overfit train: {accuracy_score(y_train_fold, y_pred_train_fold):.2f} "
          f"test: {accuracy_fold}:.2f")

    print(f"  Accuracy: {accuracy_fold:.4f}")
    print(f"  Precision: {precision_fold:.4f}")
    print(f"  Recall: {recall_fold:.4f}")
    print(f"  F1-Score: {f1_fold:.4f}")


    # Calculate metrics for the current fold on unseen data
    accuracy_fold = accuracy_score(tags_test, y_pred_unseen_fold)
    precision_fold = precision_score(tags_test, y_pred_unseen_fold, zero_division=0)
    recall_fold = recall_score(tags_test, y_pred_unseen_fold, zero_division=0)
    f1_fold = f1_score(tags_test, y_pred_unseen_fold, zero_division=0)
    print(f"overfit train: {accuracy_score(tags_test, y_pred_unseen_fold):.2f} "
          f"test: {accuracy_fold}:.2f")

    # Append metrics to their respective lists
    accuracies.append(accuracy_fold)
    precisions.append(precision_fold)
    recalls.append(recall_fold)
    f1_scores.append(f1_fold)

    print(f"  Accuracy unseen: {accuracy_fold:.4f}")
    print(f"  Precision unseen: {precision_fold:.4f}")
    print(f"  Recall unseen: {recall_fold:.4f}")
    print(f"  F1-Score unseen: {f1_fold:.4f}")


  avg_importances = np.mean(fold_importances, axis=0)
  std_importances = np.std(fold_importances, axis=0)
  print(avg_importances)
  print(std_importances)
  # Sort features by importance
  indices = np.argsort(avg_importances)[::-1]
  
  # Create plot
  fig3 = plt.figure(3, figsize=(6, 4))
  
  # Plot feature importances with error bars
  positions = np.arange(len(indices))
  plt.bar(positions, avg_importances[indices], 
          yerr=std_importances[indices], capsize=5, alpha=0.7,
          color='skyblue', edgecolor='navy')
  
  plt.xticks(positions, [labels[i] for i in indices], rotation=45, ha='right')
  print([labels[i] for i in indices])
  plt.ylabel('Feature Importance')
  #plt.title('Aggregated Feature Importance from 5-Fold Random Forest')
  plt.grid(axis='y', alpha=0.3)
  plt.tight_layout()
  plt.show()#block=False)

  
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
  y_pred_weighted = predict_weighted(models, accuracies / accuracies.sum(), features_test, aggregation='soft')
  y_pred_binary = (y_pred_weighted[:, 1] > rfthreshold).astype(int)
  y_probs = y_pred_weighted[:, 1]
  
  fpr, tpr, thresholds = roc_curve(tags_test, y_probs)
  
  auc_score = auc(tpr, fpr)

  if diagnostic_plots:
    # Plot the Precision-Recall curve
    fig4 = plt.figure(4, figsize=(8, 6))
    plt.plot(fpr, tpr, color='b', label=f'ROC Curve (AUC = {auc_score:.2f})')
    plt.xlabel("False Positives")
    plt.ylabel("True Positives")
    plt.title('ROC Curve')
    for i,t in enumerate(thresholds): 
      plt.text(fpr[i], tpr[i], "%.3f"%t, ha="right")
    plt.grid(True)
    plt.show()#block=False)
  elif savemodel:
    # Plot the Precision-Recall curve
    fig5 = plt.figure(5, figsize=(8, 6))
    plt.plot(fpr, tpr, color='b', label=f'ROC Curve (AUC = {auc_score:.2f})')
    plt.xlabel("False Positives")
    plt.ylabel("True Positives")
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{rootdir}/plots/rf_cmplot{filedetails}.pdf")
  
  if diagnostic_plots:
    cm = confusion_matrix(tags_test, y_pred_binary)
    
    fig6 = plt.figure(6, figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()#block=False) 

  print(f"wrong classifications: ")
  np.save("tmp",  y_pred_binary)
  for i in np.where(~( y_pred_binary == tags_test))[0]:
    print(i, y_pred_binary[i])
    pd.read_csv(config.LCVSPATH + "/" + lcvs_test[i])[["flux_hap_2"]].plot()
    plt.title(f"pred {y_pred_binary[i]}")
  plt.show()

  return  y_pred_binary


def rffeatures_train(csv=None, filename=None, choosechips=None, runsdir=None,
                     thisfilters=None, runs=None,
                     datetimes=None, extension=None,
                     verbose=None, diagnostic_plots=None, savemodel=False):


  if choosechips is None:
    choosechips = [config.choosechip]
    print(choosechips)
  if runsdir is None:
    runsdir = config.runsdir
  if thisfilters is None:
    thisfilters = [config.thisfilter]
  if runs is None:
    runs = [config.run]
  if datetimes is None:
    datetimes = [config.datetime]
  if verbose is None:
    verbose = config.VERBOSE
  if diagnostic_plots is None:
    diagnostic_plots = config.DIAGNOSTICPLOTS
  
  NCl =  config.NCl
  ap_column = config.ap_column
  
  extension = config.extension
  building_trainingset = config.BUILDINGTRAININGSET
  wavelets = config.wavelets  
  wavelet = config.wavelet
  #'haar'  # Start with 'haar', then try 'db4'
  level = config.waveletlevel
  #5        # Choose an appropriate level (e.g., 5-8)    

  rfthreshold = config.rfthreshold
  Nlc = 0
  features = []
  labels = []
  tags = []

  rootdir = config.ROOT
  labelsfile  = f"{rootdir}crwlabels_.csv"
  if wavelets:
    labelsfile  = f"{rootdir}crwlabels_wwvt_.csv"


  print(f"reading labels from {labelsfile}")
  labelsdf_here = pd.read_csv(labelsfile, index_col=0)
  labels = np.array([labelsdf_here.iloc[i]["0"] for i in range(labelsdf_here.shape[0])])
  label2num_dict_here = {l[0]:i for i,l in enumerate(labelsdf_here.values) 
                         if l[0] not in ["Hartigan's"]}
  label2num_dict_here["H"] = 2
  Nfeatures = len(labels)



  for choosechip, run, thisfilter, datetime in zip (choosechips, runs, thisfilters, datetimes):
    print(choosechip, run, thisfilter, datetime)

    filedetails, lcvarrayfile, chipfilename, featuresfile, predictionfile, rootdir = config.makenames(datetime, run, thisfilter, choosechip)
    print(filedetails)
    if wavelets:
      featuresfile = featuresfile.replace('crwfeatures', 'crwfeatures_wwvt')


    print(f"reading features from {featuresfile}")
    features_here = np.load(open(featuresfile, "rb"))

    Nlc += len(features_here)

    message = f"this code expects {Nfeatures} with wavelets set to {wavelets}," 
    message += "got {np.array(features).shape[1]}"

    assert features_here.shape[1] == Nfeatures, message

    features.append(features_here)
    
    ####################################need to change
    #tags = np.load()
    #tags.append(np.random.choice([0, 1], size=len(features_here)))
    
    tags = pd.read()
    #config.tags

    #if filedetails in tags:
    #  tag = tags[filedetails]
    #else:
    #  tag = np.zeros(Nlc)    

  ############################################

  
  tags = np.hstack(tags)

  features = np.vstack(features)
  #labels = np.vstack(labels)
  print(features.shape, labels.shape, tags.shape)

  features_scaled = preprocessing.scale(features, axis=0)
  features_scaled.std(axis=0)

  if diagnostic_plots:
    print("plottinghere")
    fig1, axs = plt.subplots(2,1,figsize=(7, 7), num=1)
    axs = axs.ravel()
    axs[0].boxplot(features[tags > 0])
    axs[0].set_xticks(range(1, features.shape[1]+1))
    axs[0].set_xticklabels(labels)

    axs[0].set_xlabel('Feature Index')
    axs[0].set_ylabel('Value')
    axs[0].tick_params(axis='x', labelrotation=45)
    plt.grid(True)

    axs[1].boxplot(features_scaled[tags > 0])
    axs[1].set_xticks(range(1, features.shape[1]+1))
    axs[1].set_xticklabels(labels)

    axs[1].set_xlabel('Feature Index')
    axs[1].set_ylabel('Scaled Value')
    axs[1].tick_params(axis='x', labelrotation=45)  
    plt.grid(True)
    plt.show()

  features_train, features_test, tags_train, tags_test  = train_test_split(features, tags, test_size=0.2, 
                                    random_state=302, shuffle=True)


  models, weights, skf = aggregate_rf_cv_weighted(features_train, tags_train)
  y_pred_weighted = predict_weighted(models, weights, features_test, aggregation='soft')

  if savemodel:
    #save the models
    print("saving model as ../models/crw_rf_models.joblib and ../models/crw_rf_weights.npy")
    import joblib
    # Assuming 'models' is the list of trained Random Forest models
    joblib.dump(models, f'{config.ROOT}/models/crw_rf_models.joblib')
    np.save(f"{config.ROOT}/models/crw_rf_weights.npy", weights)
  

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
  for fold, (train_index, test_index) in enumerate(skf.split(features_train, tags_train)):
    print(f"\n--- Fold {fold + 1}/{skf.n_splits} ---")

    # Split data into training and testing sets for the current fold
    X_train_fold, X_test_fold = features_train[train_index], features_train[test_index]
    y_train_fold, y_test_fold = tags_train[train_index], tags_train[test_index]

    # Instantiate a RandomForestClassifier for the current fold
    model_fold = RandomForestClassifier(random_state=42, class_weight='balanced_subsample', max_depth=3)
    
    # Train the model
    model_fold.fit(X_train_fold, y_train_fold)
    fold_importances.append(model_fold.feature_importances_)
    
    # Make predictions on the test data
    y_pred_fold = model_fold.predict(X_test_fold)
    y_pred_train_fold= model_fold.predict(X_train_fold)    
    y_pred_unseen_fold= model_fold.predict(features_test)    
    
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
  fig2 = plt.figure(2, figsize=(12, 8))
  
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
  plt.show()#block=False)

  
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
  
  fpr, tpr, thresholds = roc_curve(tags_test, y_probs)
  
  auc_score = auc(tpr, fpr)

  if diagnostic_plots:
    # Plot the Precision-Recall curve
    fig3 = plt.figure(3, figsize=(8, 6))
    plt.plot(fpr, tpr, color='b', label=f'ROC Curve (AUC = {auc_score:.2f})')
    plt.xlabel("False Positives")
    plt.ylabel("True Positives")
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.show()#block=False)

  if savemodel:
    # Plot the Precision-Recall curve
    fig4 = plt.figure(4, figsize=(8, 6))
    plt.plot(fpr, tpr, color='b', label=f'ROC Curve (AUC = {auc_score:.2f})')
    plt.xlabel("False Positives")
    plt.ylabel("True Positives")
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{rootdir}/plots/rf_cmplot{filedetails}.pdf")
  
  y_pred_binary = (y_pred_weighted[:, 1] > rfthreshold).astype(int)
  if diagnostic_plots:
    cm = confusion_matrix(tags_test, y_pred_binary)
    
    fig5 = plt.figure(5, figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()#block=False) 

  print(f"wrong classifications: ")
  np.save("tmp",  y_pred_binary)
  for i in np.where(~( y_pred_binary == tags_test))[0]:
    print(i)
  return  y_pred_binary



  
if __name__ == "__main__":

  
  
    parser = ArgumentParser()
    parser.add_argument("-m", "--mode", dest="mode", default="predict")
    parser.add_argument("-s", "--save", dest="savemodel", 
                        action="store_true", default=False)
    parser.add_argument("--filename", dest="filename", default=None,
                    help="filename if training with a list")
    parser.add_argument("--csv", dest="csv", default=None,
                    help="csv label file for grouplabels")
    parser.add_argument("-c", "--chip", dest="choosechip", default=None,
                    help="chip number(2 digits)")
    parser.add_argument("-d", "--runsdir", dest="runsdir", default=None,
                    help="chip number(2 digits)")
    parser.add_argument("-f", "--filter", dest="filter", default=None,
                    help="chip number(2 digits)")
    parser.add_argument("-r", "--run", dest="run", default=None,
                    help="chip number(2 digits)")
    parser.add_argument("-t", "--datetime", dest="datetime", default=None,
                    help="chip number(2 digits)")
    parser.add_argument("-q", "--quiet",
                    action="store_false", dest="silent", default=True,
                    help="don't print status messages to stdout")

    args = parser.parse_args()
    filename = args.filename
    csv = args.csv
    silent = args.silent
    choosechip =  args.choosechip
    runsdir = args.runsdir
    thisfilter = args.filter
    run = args.run
    mode = args.mode
    savemodel = args.savemodel
    datetime = args.datetime
            
    
    print(f"rffeatures_train({choosechip}, {runsdir}, {thisfilter}, {run}, {datetime}, {silent})")
   
    if csv or filename:
      if savemodel:
        rffeatures_fromfile_train(csv=csv, filename=filename,
                                  diagnostic_plots=False,
                                  savemodel=True)#, diagnostic_plots=True)
      else:
        rffeatures_fromfile_train(filename=filename, csv=csv, diagnostic_plots=True)

    elif mode == "train":
      print(f'rffeatures_train({choosechip.split(",")}, {runsdir}, {thisfilter.split(",")}, {run.split(",")}, {datetime.split(",")}, {silent})')
      #rffeatures_train(choosechip, runsdir, thisfilter, run, datetime, silent)
      if savemodel:
        rffeatures_train(choosechips=choosechip.split(","), runsdir=runsdir,
                         thisfilters=thisfilter.split(","), runs=run.split(","),
                         datetimes=datetime.split(","), savemodel=True, diagnostic_plots=True)
      else:
        rffeatures_train(choosechips=choosechip.split(","), runsdir=runsdir,
                         thisfilters=thisfilter.split(","), runs=run.split(","),
                         datetimes=datetime.split(","), diagnostic_plots=True)  
  
    else:
      rffeatures_predict(choosechip=choosechip, runsdir=runsdir,
                         thisfilter=thisfilter, run=run,
                         datetime=datetime, verbose=~silent)  


  




