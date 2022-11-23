import sys
sys.path.insert(1, '/Users/jakezur/Documents/boredom-projects/KMeansFromScratch/KMeansClustering/KMeansModel')
from KMeansClustering import KMeans
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pprint
from PIL import Image

img_path = '/Users/jakezur/Documents/boredom-projects/KMeansFromScratch/KMeansClustering/TestingData/MonarchButterfly.png'
img = mpimg.imread(img_path)

dimmensions = img.shape
x = []
y = []
red = []
blue = []
green = []

for i in range(dimmensions[0]):
    for j in range(dimmensions[1]):
        x.append(j)
        y.append(i)
        if img_path.__contains__('.png'):
            red.append(int(img[i, j][0]*255))
            green.append(int(img[i, j][1]*255))
            blue.append(int(img[i, j][2]*255))
        else:
            red.append(int(img[i, j][0]))
            green.append(int(img[i, j][1]))
            blue.append(int(img[i, j][2]))

df = pd.DataFrame()
df['x'] = x
df['y'] = y
df['red'] = red
df['green'] = green
df['blue'] = blue

imgplot = plt.imshow(img)
plt.show()
print(df)

cluster_model = KMeans(k=5, df=df, features=['red', 'green', 'blue'])
cluster_model.train_model()

for centroid in cluster_model.centroids:
    print(centroid.coordinates)
    print("\033[48;2;"+str(int(centroid.coordinates[0]))+";"+str(int(centroid.coordinates[1]))+";"+str(int(centroid.coordinates[2]))+"m ..................\033[0m")

new_x = []
new_y = []
new_r = []
new_g = []
new_b = []
for index,cluster in enumerate(cluster_model.clusters):
    for point in cluster:
        new_x.append(int(point.attributes[0]))
        new_y.append(int(point.attributes[1]))
        new_r.append(int(cluster_model.centroids[index].coordinates[0]))
        new_g.append(int(cluster_model.centroids[index].coordinates[1]))
        new_b.append(int(cluster_model.centroids[index].coordinates[2]))

new_df = pd.DataFrame()
new_df['x'] = new_x
new_df['y'] = new_y
new_df['red'] = new_r
new_df['green'] = new_g
new_df['blue'] = new_b

new_df = new_df.sort_values(by=['y','x'])
print(new_df)

new_r = new_df['red'].to_list()
new_g = new_df['green'].to_list()
new_b = new_df['blue'].to_list()
plt.clf()
plt.close()
new_img = Image.new('RGB', (dimmensions[1], dimmensions[0]))
new_img.putdata(list(zip(new_r, new_g, new_b)))
imgplt2 = plt.imshow(new_img)
plt.show()