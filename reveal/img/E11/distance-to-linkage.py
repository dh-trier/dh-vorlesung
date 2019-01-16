#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Filename: topicpca.py
# Author: #cf


"""
Set of functions to perform clustering on data. 
Built for topic probabilities or word frequencies as input. 
Performs Principal Component Analysis or distance-based clustering.
"""

import os, glob, re
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import pygal
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering as AC
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from sklearn import metrics
from scipy.cluster.hierarchy import fcluster


def get_distance_matrix(DistanceMatrixFile): 
    with open(DistanceMatrixFile, "r") as InFile: 
        DistanceMatrix = pd.DataFrame.from_csv(InFile)
        print(DistanceMatrix)
        return DistanceMatrix


def clusteranalysis(DistanceMatrix, Method, Metric):
    """
    docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
    """
    # perform the cluster analysis
    LinkageMatrix = linkage(DistanceMatrix, method=Method, metric=Metric)
    print(LinkageMatrix)
    #with open("linkage-matrix.csv", "w") as OutFile: 
    #    LinkageMatrix.to_csv(OutFile)
    return LinkageMatrix


def make_dendrogram(LinkageMatrix, 
                    Method, Metric, Labels):
    import matplotlib
    plt.figure(figsize=(6,6))
    plt.title("Dendrogramm (Worthäufigkeiten)", fontsize=14)
    #plt.ylabel("Parameters: "+Method+" method, "+Metric+" metric. CorrCoeff: "+str(CorrCoeff)+".")
    plt.xlabel("Distanz\n(Parameter: "+Method+" / "+Metric+")", fontsize=12)
    matplotlib.rcParams['lines.linewidth'] = 1.8
    dendrogram(
        LinkageMatrix,
        truncate_mode="level",
        color_threshold = 1.8,
        show_leaf_counts = True,
        no_labels = False,
        orientation="right",
        labels = Labels, 
        leaf_rotation = 0,  # rotates the x axis labels
        leaf_font_size = 12,  # font size for the x axis labels
        )
    #plt.show()
    plt.savefig("dendrogram_"+Method+"-"+Metric+".png", dpi=300, figsize=(12,18), bbox_inches="tight")
    plt.close()


Labels = ["CorneilleP_1","CorneilleP_2","RacineJ_1","RacineJ_1","CorneilleT_1","CorneilleT_2"]
DistanceMatrix = get_distance_matrix("5_distance-matrix.csv")
LinkageMatrix = clusteranalysis(DistanceMatrix, Method="ward", Metric="euclidean")
make_dendrogram(LinkageMatrix, Method="ward", Metric="euclidean", Labels=Labels)   