from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
import matplotlib.pyplot as plt
import numpy as np

import pylab as plt
import pandas as pd
from sklearn import preprocessing
import pywt
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm
import matplotlib.cm as cm
import crw_config as config
from utils import PR
from showlcv import plotlcv

#from unidip import UniDip
#import unidip.dip as dip

NCl = 2
wavelets = True


def clusterfeatures(choosechip=None, runsdir=None,
                                   thisfilter=None, run=None,
                    datetime=None, extension=None, verbose=None):

  if all(arg is None for arg in [choosechip, runsdir,
                                   thisfilter, run, datetime, extension]):
      NCl=  config.NCl
      ap_column = config.ap_column

      verbose = config.VERBOSE
      diagnostic_plots = config.DIAGNOSTICPLOTS
      building_trainingset = config.BUILDINGTRAININGSET
      
      choosechip = config.choosechip
      runsdir = config.runsdir
      thisfilter = config.thisfilter
      run = config.run
      datetime = config.datetime
      extension = config.extension
      filedetails = config.filedetails
      tags = config.tags
      
      wavelet = config.wavelet
      #'haar'  # Start with 'haar', then try 'db4'
      level = config.waveletlevel
      #5        # Choose an appropriate level (e.g., 5-8)
      
      featuresfile = config.featuresfile
  else:
      featuresfile = f"crwfeatures_{datetime}_{run}_{thisfilter}_{choosechip}.npy"

  
  if  filedetails in tags:
          tag = tags[filedetails]
  else:
          tag = np.zeros(Nlc)    
      
  labelsfile  = "crwlabels_.csv"
  if wavelets:
    featuresfile = featuresfile.replace('crwfeatures', 'crwfeatures_wwvt')
    labelsfile  = "crwlabels_wwvt_.csv"

  print(f"reading features from {featuresfile}")
  features = np.load(open(featuresfile, "rb"))
  labelsdf = pd.read_csv(labelsfile, index_col=0)
  labels = np.array([labelsdf.iloc[i]["0"] for i in range(labelsdf.shape[0])])
  
  Nlc = len(features)
  Nfeatures = len(labels)

  label2num_dict = {l[0]:i for i,l in enumerate(labelsdf.values) if l[0] not in ["Hartigan's"] }
  label2num_dict["H"] = 2

  message = f"this code expects {Nfeatures} with wavelets set to {wavelets}"
  assert features.shape[1]== Nfeatures, message

  features_scaled = preprocessing.scale(features, axis=0)
  features_scaled.std(axis=0)


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


  print("Starting clustering vs tags")
  
  k = NCl  # Initialize KMeans model
  kmeans = KMeans(n_clusters=k, random_state=302)

  cluster_labels_km = kmeans.fit_predict(features_scaled[:,:5])


  print("Cluster labels:")  
  PR(tag, cluster_labels_km, k)


  # Perform hierarchical clustering with 2 clusters
  hierarchical_clustering = AgglomerativeClustering(n_clusters=4)
  cluster_labels_h = hierarchical_clustering.fit_predict(features_scaled)

  print(f"Hierarchical clustering performed. Number of clusters: {len(np.unique(cluster_labels_h))}")
  print(f"Cluster labels (first 10): {cluster_labels_h[:10]}")
  PR(tag, cluster_labels_h, 4)

  db = DBSCAN(eps=0.7, min_samples=5)

  # Fit the model to your features and predict cluster labels
  cluster_labels_db = db.fit_predict(features_scaled[:,:])

  print(f"DBSCAN found {len(np.unique(cluster_labels_db))} clusters (including noise, if any).")
  print("Cluster labels :")
  PR(tag, cluster_labels_db, cluster_labels_db.max())


  fig, axs = plt.subplots(1, 3, figsize=(15,7))
  axs = axs.ravel()
  colors = ["k", "r",  "b", "g"] # max 4 types in tags
  for i,(c,ckind) in enumerate(zip((cluster_labels_km, cluster_labels_h, cluster_labels_db),
                     ("KMeans", "Hierarchical", "DBScan"))):
    for l in np.unique(c):
      for t in np.unique(tag):
        select = ((c == l) & (tag == t)) 
        size = select.sum()
        axs[i].scatter(t, l, s=size, marker='o', c=colors[int(t)])
        axs[i].set_title(ckind)
    
    for t in np.unique(tag):
        select = (tag == t)
        size = select.sum()
        axs[i].scatter(0.5,
                       (axs[i].get_ylim()[1] - axs[i].get_ylim()[0]) * 0.5 + axs[i].get_ylim()[0],
                       c=colors[int(t)], s=size, marker='o', alpha=0.3)
        axs[i].set_title(ckind)      
  plt.show()

  select = [i for i in range(len(tag)) if (cluster_labels_db[i] == -1) & (tag[i] == 0)]
  print("plotting the non tagged in anomaly cluster")
  for s in select:
    plotlcv(s)
    
  plt.show()
  print("plotting the tagged in normal clusters")
  select = [i for i in range(len(tag)) if (cluster_labels_db[i] > -1) & (tag[i] == 1)]

  for s in select:
    plotlcv(s)
  plt.show()    
 
  
if __name__ == "__main__":
  clusterfeatures()




