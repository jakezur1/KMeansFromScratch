from attr import attrib
from Point import Point
import random
import math
import pandas as pd
import numpy as np
import pprint
import collections

class KMeans:
    points = []
    centroids = []
    clusters = []

    def __init__(self, k, df, features):
        self.k = k
        self.df = df
        self.features = features
        self.initialize_centroids()
        for row in range(len(df)):
            coordinates = []
            for feature in self.features:
                feature_list = df[feature].to_list()
                coordinates.append(feature_list[row])
            attributes = []
            for column in df.columns:
                is_a_feature = False
                for feature in self.features:
                    if column == feature:
                        is_a_feature = True
                if is_a_feature == False:
                    attribute_list = df[column].to_list()
                    attributes.append(attribute_list[row])
            self.points.append(Point(attributes=attributes, coordinates=coordinates))
        

    def initialize_centroids(self):
        self.centroids = []
        for cluster in range(self.k):
            centroid = Point(attributes=[], coordinates=[])
            for feature in self.features:
                max_val = self.df[feature].max()
                min_val = self.df[feature].min()
                rand_coordinate = None
                try:
                    rand_coordinate = random.randint(min_val, max_val)
                except Exception as e:
                    rand_coordinate = random.uniform(min_val, max_val)
                centroid.coordinates.append(rand_coordinate)
            self.centroids.append(centroid)
    
    def calc_distance(self, point):
        df = pd.DataFrame()
        for num,centroid in enumerate(self.centroids):
            distance = 0
            for index,coordinate in enumerate(centroid.coordinates):
                difference = coordinate - point.coordinates[index]
                difference = math.pow(difference, 2)
                distance+=difference
            distance = math.sqrt(distance)
            distance_series = pd.Series([distance])
            df["centroid" + str(num)] = (distance_series)
        return df

    def update_clusters(self):
        distance_df = pd.DataFrame()
        for point in self.points:
            row = self.calc_distance(point)
            distance_df = pd.concat([distance_df, row])
        clusters = []
        for centroid in self.centroids:
            clusters.append([])
        min_distances = distance_df.min(axis=1).to_list()
        for row in range(len(distance_df)):
            for index,column in enumerate(distance_df.columns):
                column_list = distance_df[column].to_list()
                if column_list[row] == min_distances[row]:
                    clusters[index].append(self.points[row])
                    break
        self.clusters = clusters
        for cluster in self.clusters:
            if len(cluster) == 0:
                self.initialize_centroids()
                self.update_clusters()

    def update_centroids(self):
        centroids_are_stationary = []
        for num,cluster in enumerate(self.clusters):
            cluster_df = pd.DataFrame()
            master_coordinates = []
            threshold = 0.0
            for feature in self.features:
                master_coordinates.append([])
            for point in cluster:
                for index,coordinate in enumerate(point.coordinates):
                    master_coordinates[index].append(coordinate)
            for index,feature in enumerate(self.features):
                cluster_df["x"+str(index)] = master_coordinates[index]
            new_centroid = Point(attributes=[], coordinates=[])
            for count,column in enumerate(cluster_df.columns):
                coordinates_list = cluster_df[column].to_list()
                coordinates_arr = np.asarray(coordinates_list, dtype = np.float64)
                mean_value = coordinates_arr.mean()
                theshold = mean_value*0.001
                new_centroid.coordinates.append(mean_value)
            distance = 0
            for index,coordinate in enumerate(new_centroid.coordinates):
                difference = coordinate - self.centroids[num].coordinates[index]
                difference = math.pow(difference, 2)
                distance+=difference
            distance = math.sqrt(distance)
            if distance > threshold:
                self.centroids[num] = new_centroid
                centroids_are_stationary.append(0)
            else:
                self.centroids[num] = new_centroid
                centroids_are_stationary.append(1)
        if centroids_are_stationary.__contains__(0):
            return 1
        return 0
    
    def train_model(self, n=0):
        self.update_clusters()
        keep_going = self.update_centroids()
        print("Iterations: " + str(n+1))
        if keep_going == 1:
            self.train_model(n+1)
        else:
            self.update_clusters()
            return self.clusters

    def clusters_as_df(self):
        clusters = []
        for cluster in self.clusters:
            cluster_df = pd.DataFrame()
            cluster_arr = []
            for coordinate in cluster[0].coordinates:
                cluster_arr.append([])
            for attribute in cluster[0].attributes:
                cluster_arr.append([])
            for point in cluster:
                for index,coordinate in enumerate(point.coordinates):
                    cluster_arr[index].append(coordinate)
                for num,attribute in enumerate(point.attributes):
                    cluster_arr[num+len(self.features)].append(attribute)
            for iterator,feature in enumerate(self.features):
                cluster_df[feature] = cluster_arr[iterator]
            for count,indexes in enumerate(cluster_arr):
                if count >= len(self.features):
                    cluster_df['attribute'+str(count)] = cluster_arr[count]
            clusters.append(cluster_df)
        return clusters