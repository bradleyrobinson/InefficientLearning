# ########################################
# now, write some code
# ########################################
import glob
import numpy as np
import re
import random
from time import time
from collections import Counter
import os


# Used for testing
def print_time(time0, time1, function_name):
    diff = time1 - time0
    print("{function}: {diff}".format(function=function_name, diff=diff))
    
    
class KMeansText(object):
    # Kmeans object takes a tf_idf table, and then uses it to compute kmeans
    # self.tf_idf - tf_idf table
    # self.shape - length of x and y axis
    # self.centroids - Holds centroids for each cluster
    # self.cluster_differences - Includes differences 
    def __init__(self, tf_idf, shape):
        self.tf_idf = tf_idf
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
        no_zero_changes = 0
        while no_zero_changes < 1 and iterations < 30:
            # TODO: Labeling clusters takes more time, find a way to optimize
            self.label_clusters()
            self.recompute_centroids()
            self.check_changes()
            # print("iteration: {}, rss: {}".format(iterations, self.changes))
            iterations += 1
            if self.changes < 1:
                no_zero_changes += 1
        self.rss_final = self.rss_scores[-1]
        
        


# Clustering algorithm for documents. Reads all documents in a file path
# Creates a tf_idf vector, and computes kmeans.
class Clustering(object):
    # Variables:
    #
    # self.term_index, dictionary that includes each term and the documents
    # that include them.
    # 
    # self.tf_idf, numpy array that stores the weighted tf_idf of terms in docs
    #
    # self.x_idx, self.y_idx, dictionary with x or y name (document, term) and
    # their index in self.tf_idf
    #
    # self.inverse_x, dictionary that is the inverse of self.x_idx
    #
    # self.clusters, List that stores Kmeans_Text objects in order to compare
    # results.
    #
    # self.labels, List that stores the cluster label for each document, in 
    # order
    #
    # self.term_pos, Used to help index terms in self.y_idx
    def __init__(self):
        self.term_index = {}
        self.tf_idf = np.array([])
        self.x_idx = {}
        self.y_idx = {}
        self.inverse_x = {}
        # An array of cluster objects
        self.clusters = []
        self.labels = []
        self.term_pos = 0
        
    # initialize ( )
    # Since the assignment uses the same object, resets values to ensure no
    # values bleed over
    # Requirements: None
    # Parameters: None
    # Returns: None
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
        
    
    # index_terms ( terms, doc )
    # purpose: Add each term that occurs in each document to the term_index
    # Preconditions: Terms are tokenized
    # Returns None
    # Parameters: 
    #   terms - tokenized list of terms
    #   doc - document name that contains the terms
    def index_terms(self, terms, doc):
        for t in terms:
            if t not in self.term_index:
                self.term_index[t] = [doc]
                # Add to the idx
                self.y_idx[t] = self.term_pos
                self.term_pos += 1
            else:
                self.term_index[t].append(doc)
    
    
    # create_term_index ( )
    # Purpose: Create a tf_idf vector
    # Preconditions: self.index_files has been called
    # Returns: None
    # Parameters: None
    def create_term_index(self):
        # Create the array
        self.tf_idf = np.zeros((len(self.x_idx), self.term_pos))
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
                
    
    # test_kmeans ( k )
    # purpose: compares this kmeans algorithm with the sklearn implementation
    # preconditions: the terms have been indexed
    # returns: in-order labels of the cluster names each doc belongs to
    # parameters: k number of clusters to make
    def test_kmeans(self, k):
        from sklearn.cluster import KMeans
        k_test = KMeans(n_clusters=k)
        k_test.fit(self.tf_idf)
        return k_test.labels_
    
    
    # kmeans( k )
    # purpose: Once the directory and tf_idf has been initialized, actually do
    # the stuff
    # preconditions: consume_dir has been called.
    # returns: KMeansText object with the lowest rss score
    def kmeans(self, k, restarts=8):
        for i in range(restarts):
            kmeans = KMeansText(self.tf_idf, self.shape)
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
        
        # To compare the kmeans to popular library sklearn kmeans, uncomment
        # self.test_kmeans(k)
        return self.clusters[min_index]
    
    
    # prepare ( )
    # purpose: Initializes some variables based on the term index
    # preconditions: self.index_files has been called successfully
    # returns: None
    # parameters: None
    def prepare(self):
        self.shape = (len(self.x_idx), len(self.y_idx))
        self.labels = np.zeros(self.shape[0], dtype=np.int)
        self.inverse_x = {y: x for x, y in self.x_idx.items()}
        self.create_term_index()
        
        
    # index_files ( path )
    # purpose: goes through files in a given path and indexes the terms
    # preconditions: self.initialize has been called
    # returns: None
    # parameters: path - a string for the location of a folder to be indexed
    def index_files(self, path):
        dir_files = glob.glob(path + "*")
        self.N = len(dir_files)
        for i in range(self.N):
            file = dir_files[i]
            name = os.path.split(file)[-1]
            self.x_idx[name] = i
            contents = open(file, 'r')
            terms = self.tokenize(contents.read())
            self.index_terms(terms, name)
            
            
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
        self.index_files(path)
        self.prepare()
        result = self.kmeans(k).labels
        # Find the correct labels
        simple_labels = []
        for cluster, doc in result.items():
            c_labels = []
            for d in doc:
                self.labels[d] = cluster
                c_labels.append(self.inverse_x[d])
            simple_labels.append(c_labels)
        # Print out the labels as expected:
        return simple_labels
    
    
    def print_tf(self):
        # TODO: This should be organized and meaningful
        print(self.x_idx)
        print(self.y_idx)
        print(self.tf_idf)
        print(self.tf_idf.shape)


# now, we'll define our main function which actually starts the clusterer
def main(args):
    # Requires two folders to test directly: test10/ and test50/
    clustering = Clustering()    
    """print("test 10 documents")
    print(clustering.consume_dir('test10/', 5))
    print("test 50 documents")
    print(clustering.consume_dir('test50/', 5))"""
    

if __name__ == "__main__":
    import sys
    main(sys.argv)

