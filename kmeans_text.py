# homework 4
# goal: k-means clustering on vectors of TF-IDF values,
#   normalized for every document.
# exports: 
#   student - a populated and instantiated ir470.Student object
#   Clustering - a class which encapsulates the necessary logic for
#       clustering a set of documents by tf-idf 


# ########################################
# first, create a student object
# ########################################

import ir4320
MY_NAME = "Bradley Robinson"
MY_ANUM  = 989743 # put your UID here
MY_EMAIL = "bradley.s.robinson12@gmail.com"

# the COLLABORATORS list contains tuples of 2 items, the name of the helper
# and their contribution to your homework
COLLABORATORS = [ 
    ]

# Set the I_AGREE_HONOR_CODE to True if you agree with the following statement
# "An Aggie does not lie, cheat or steal, or tolerate those who do."
I_AGREE_HONOR_CODE = True

# this defines the student object
student = ir4320.Student(
    MY_NAME,
    MY_ANUM,
    MY_EMAIL,
    COLLABORATORS,
    I_AGREE_HONOR_CODE
    )


# ########################################
# now, write some code
# ########################################
import glob
import numpy as np
import re
import random
from time import time
from collections import Counter
# TODO: Remove this after testing
from sklearn.cluster import KMeans


# Used for testing
def print_time(time0, time1, function_name):
    diff = time1 - time0
    print("{function}: {diff}".format(function=function_name, diff=diff))
    
    
# Kmeans object takes a tf_idf table, and then uses it to compute kmeans

class KMeans_Text(object):
    def __init__(self, tf_idf, x_idx, y_idx, shape):
        self.tf_idf = tf_idf
        self.x_idx = x_idx
        self.y_idx = y_idx
        self.shape = shape
        self.centroids = {}
        self.clusters = {}
        self.cluster_differences = {}
        self.labels = {}
        self.rss_scores = []
        self.shape = shape
        self.N = None
        self.changes = None
    
    def rss(self):
        rss_score = 0
        for k, rows in self.cluster_differences.items():
            rss_score += sum(rows)
        self.rss_scores.append(rss_score)
        
        
    def get_rand_clusters(self):
        x = random.randint(0, self.shape[0]-1)
        return x
                
    
    def initialize_clusters(self, k):
        # Picks random points as centroids
        self.centroids = {}
        centroid_lst= []
        # Set it to something other than zero, otherwise the clustering will
        # end too soon
        self.changes = 1
        for i in range(k):
            x = self.get_rand_clusters()
            while x in centroid_lst:
                x = self.get_rand_clusters()
            self.centroids[i] = self.tf_idf[x]
            centroid_lst.append(x)
            
    
    def euclidean_distance(self, x1, x2):
        distance = (sum([(a-b)**2 for a, b in zip(x1, x2)])**(1/2))
        return distance
    
    
    def check_changes(self):
        # TODO: No need for i or cluster, this just takes a look at stuff
        if len(self.rss_scores)  > 1:
            self.changes = self.rss_scores[-2] - self.rss_scores[-1]
        else:
            self.changes = self.rss_scores[0]
    
    
    def compare_labels(self, labels, clusters, cluster, i, row, squared_error):
        if cluster[1] in labels:
            labels[cluster[1]].append(i)
            clusters[cluster[1]].append(row)
        else:
            labels[cluster[1]] = [i]        
            clusters[cluster[1]] = [row]
        if cluster[1] in squared_error:
            squared_error[cluster[1]].append(cluster[0]**2)
        else:
            squared_error[cluster[1]] = [cluster[0]**2]
    
    
    def label_clusters(self):
        # We need to clear out the old for the new, appending is a bad idea
        labels = {}
        clusters = {}
        squared_error = {}
        # Iterates through each row and finds its corresponding cluster
        i = 0
        for row in self.tf_idf:
            cluster_distances = {}
            for cluster, centroid in self.centroids.items():
                distance = self.euclidean_distance(centroid, row)
                cluster_distances[distance] = cluster
            distance_cluster = min(cluster_distances.items())
            self.compare_labels(labels, clusters, distance_cluster, i, row, squared_error)
            i += 1
        self.labels = labels
        self.clusters = clusters
        self.cluster_differences = squared_error
        self.rss()
            
            
    def calculate_centroids(self, data):
        # TODO: Computes the average point in all the columns
        cols = self.shape[1]
        centroid = [np.mean(data[:,i]) for i in range(cols)]
        return centroid
        
    
    def recompute_centroids(self):
        # Iterates through all of the values of each cluster and recomputes the
        # centroid value
        for cluster, rows in self.labels.items():
            cluster_values = self.tf_idf[rows]
            self.centroids[cluster] = self.calculate_centroids(cluster_values)
        
    def fit(self, k):
        self.initialize_clusters(k)
        iterations = 0
        # tODO: Change this
        no_zero_changes = 0
        while no_zero_changes <= 3 and iterations < 30:
            # TODO: Labeling clusters takes more time, find a way to optimize
            self.label_clusters()
            self.recompute_centroids()
            # TODO: Is this the right place for this?
            self.check_changes()
            iterations += 1
            if self.changes == 0:
                no_zero_changes += 1
        self.rss_final = self.rss_scores[-1]
        
        

# Our Clustering object will contain all logic necessary to crawl a local
# directory of text files, tokenize them, calculate tf-idf vectors on their
# contents then cluster them according to k-means. The Clustering class should
# select r random restarts to ensure getting good clusters and then use RSS, an
# internal metric, to calculate best clusters.  The details are left to the
# student.

class Clustering(object):
    # hint: create something here to hold your dictionary and tf-idf for every
    #   term in every document
    def __init__(self):
        self.term_index = {}
        self.tf_idf = np.array([])
        self.x_idx = {}
        self.y_idx = {}
        # An array of cluster objects
        self.clusters = []
        self.term_pos = 0
        
    
    def initialize(self):
        self.term_index = {}
        self.tf_idf = np.array([])
        self.x_idx = {}
        self.y_idx = {}
        self.clusters = []
        self.N = None
        
        
    # Simple testing code to see how long functions take
    # Hasn't been tested itself (ironically)
    def class_tester(self, function, args, function_name):
        time0 = time()
        function(args)
        time1 = time()
        print_time(time0, time1, function_name)


    # tokenize( text )
    # purpose: convert a string of terms into a list of terms 
    # preconditions: none
    # returns: list of terms contained within the text
    # parameters:
    #   text - a string of terms
    def tokenize(self, text):
        clean_string = re.sub('[^a-z0-9 *]', ' ', text.lower())
        tokens = clean_string.split()
        return tokens
        
    
    
    def index_terms(self, terms, doc):
        for t in terms:
            if t not in self.term_index:
                self.term_index[t] = [doc]
                # Add to the idx
                self.y_idx[t] = self.term_pos
                self.term_pos += 1
            else:
                self.term_index[t].append(doc)
    
    
    def create_term_index(self):
        # Create the array
        self.tf_idf = np.zeros((len(self.x_idx), self.term_pos))
        # Go through each term, find the frequency of each document, then modify
        # the number to have the tf_idf
        for term, doc_list in self.term_index.items():
            y_pos = self.y_idx[term]
            doc_counter = Counter(doc_list)
            doc_frequency = len(doc_counter)
            for d in doc_list:
                x_pos = self.x_idx[d]
                term_count = doc_counter[d]
                tf = 1 + np.log10(term_count)
                idf = np.log10(self.shape[0]/doc_frequency)
                self.tf_idf[x_pos, y_pos] = tf*idf
                
    
    # Will test this implementation against sklearn to see if things are 
    # generally accurate
    def test_kmeans(self, k):
        k_test = KMeans(n_clusters=k)
        k_test.fit(self.tf_idf)
        print(k_test.labels_)
        # TODO:
        pass
    
    # kmeans( k )
    # purpose: Once the directory and tf_idf has been initialized, actually do
    # the stuff
    # preconditions: consume_dir has been called.
    def kmeans(self, k, restarts=5):
        # TODO: create KMeans objects for each restart
        for i in range(restarts):
            kmeans = KMeans_Text(self.tf_idf, self.x_idx, self.y_idx, self.shape)
            kmeans.fit(k)
            self.clusters.append(kmeans)
        
        min_rss = None
        min_index = 0
        for i in range(len(self.clusters)):
            cluster = self.clusters[i]
            rss = cluster.rss_final
            if not min_rss or rss < min_rss:
                min_rss = rss
                min_index = i
                
        self.test_kmeans(k)
        return self.clusters[min_index]
        
    # consume_dir( path, k )
    # purpose: accept a path to a directory of files which need to be clustered
    # preconditions: none
    # returns: list of documents, clustered into k clusters
    #   structured as follows:
    #   [
    #       [ first, cluster, of, docs, ],
    #       [ second, cluster, of, docs, ],
    #       ...
    #   ]
    #   each cluster list above should contain the document name WITHOUT the
    #   preceding path.  JUST The Filename.
    # parameters:
    #   path - string path to directory of documents to cluster
    #   k - number of clusters to generate
    def consume_dir(self, path, k):
        self.initialize()
        dir_files = glob.glob(path + "*")
        self.N = len(dir_files)
        for i in range(self.N):
            file = dir_files[i]
            self.x_idx[file] = i
            contents = open(file, 'r')
            terms = self.tokenize(contents.read())
            self.index_terms(terms, file)
        self.shape = (len(self.x_idx), len(self.y_idx))
        self.create_term_index()
        result = self.kmeans(k)
        #self.print_tf()
        # TODO: Something is wrong here, we don't want to just get two labels:
        return result.labels
    
    
    def print_tf(self):
        # TODO: This should be organized and meaningful
        print(self.x_idx)
        print(self.y_idx)
        print(self.tf_idf)
        print(self.tf_idf.shape)


# now, we'll define our main function which actually starts the clusterer
def main(args):
    print(student)
    clustering = Clustering()
    # TODO: Remove this
    print('test 5 documents')
    print(clustering.consume_dir('test5/', 2))
    
    print("test 10 documents")
    print(clustering.consume_dir('test10/', 5))
    print("test 50 documents")
    print(clustering.consume_dir('test50/', 5))
# this little helper will call main() if this file is executed from the command
# line but not call main() if this file is included as a module
if __name__ == "__main__":
    import sys
    main(sys.argv)

