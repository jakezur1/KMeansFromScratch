import sys
sys.path.insert(1, '/Users/jakezur/Documents/boredom-projects/KMeansClustering/KMeansModel')
from KMeansClustering import KMeans
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pprint
from PIL import Image

file_path = '/Users/jakezur/Documents/VSCode/Boredom Projects/KMeansClustering/Data/IRIS 2.csv'
df = pd.read_csv(file_path)

sepal_length = df['sepal_length'].to_list()
sepal_width = df['sepal_width'].to_list()
petal_length = df['petal_length'].to_list()
petal_width = df['petal_width'].to_list()
species = df['species'].to_list()

cluster_model = KMeans(k=3, df=df, features=['sepal_width', 'petal_length', 'petal_width'])
cluster_model.train_model()

clusters = cluster_model.clusters_as_df()
for cluster in clusters:
    print(cluster)