'''kmeans.py
Performs K-Means clustering
Olivia Doherty
CS 251/2: Data Analysis and Visualization
Spring 2024
'''
import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, data=None):
        '''KMeans constructor

        (Should not require any changes)
        
        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''
        # k: int. Number of clusters
        self.k = None
        # centroids: ndarray. shape=(k, self.num_features)
        #   k cluster centers
        self.centroids = None
        # data_centroid_labels: ndarray of ints. shape=(self.num_samps,)
        #   Holds index of the assigned cluster of each data sample
        self.data_centroid_labels = None

        # inertia: float.
        #   Mean squared distance between each data sample and its assigned (nearest) centroid
        self.inertia = None

        # num_samps: int. Number of samples in the dataset
        self.num_samps = None
        # num_features: int. Number of features (variables) in the dataset
        self.num_features = None

        if data is not None:
            # data: ndarray. shape=(num_samps, num_features)
            self.data = data.copy()
            self.num_samps, self.num_features = data.shape

    def set_data(self, data):
        '''Replaces data instance variable with `data`.

        Reminder: Make sure to update the number of data samples and features!

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''
        self.data = data.copy()
        self.num_samps, self.num_features = data.shape

    def get_data(self):
        '''Get a COPY of the data

        Returns:
        -----------
        ndarray. shape=(num_samps, num_features). COPY of the data
        '''
        return self.data.copy()

    def get_centroids(self):
        '''Get the K-means centroids

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(k, self.num_features).
        '''
        return self.centroids

    def get_data_centroid_labels(self):
        '''Get the data-to-cluster assignments

        (Should not require any changes)

        Returns:
        -----------
        ndarray of ints. shape=(self.num_samps,)
        '''
        return self.data_centroid_labels

    def dist_pt_to_pt(self, pt_1, pt_2):
        '''Compute the Euclidean distance between data samples `pt_1` and `pt_2`

        Parameters:
        -----------
        pt_1: ndarray. shape=(num_features,)
        pt_2: ndarray. shape=(num_features,)

        Returns:
        -----------
        float. Euclidean distance between `pt_1` and `pt_2`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''
        return np.linalg.norm(pt_1 - pt_2)

    def dist_pt_to_centroids(self, pt, centroids):
        '''Compute the Euclidean distance between data sample `pt` and and all the cluster centroids
        self.centroids

        Parameters:
        -----------
        pt: ndarray. shape=(num_features,)
        centroids: ndarray. shape=(C, num_features)
            C centroids, where C is an int.

        Returns:
        -----------
        ndarray. shape=(C,).
            distance between pt and each of the C centroids in `centroids`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''
        return np.linalg.norm(centroids - pt, axis = 1)

    def initialize(self, k):
        '''Initializes K-means by setting the initial centroids (means) to K unique randomly
        selected data samples

        Parameters:
        -----------
        k: int. Number of clusters

        Returns:
        -----------
        ndarray. shape=(k, self.num_features). Initial centroids for the k clusters.

        NOTE: Can be implemented without any for loops
        '''
        ind = np.random.choice(self.num_samps, k, replace = False)
        return self.data[ind]

    def cluster(self, k=2, tol=1e-2, max_iter=1000, verbose=False, p=2):
        '''Performs K-means clustering on the data

        Parameters:
        -----------
        k: int. Number of clusters
        tol: float. Terminate K-means if the (absolute value of) the difference between all
        the centroid values from the previous and current time step < `tol`.
        max_iter: int. Make sure that K-means does not run more than `max_iter` iterations.
        verbose: boolean. Print out debug information if set to True.

        Returns:
        -----------
        self.inertia. float. Mean squared distance between each data sample and its cluster mean
        int. Number of iterations that K-means was run for

        TODO:
        - Initialize K-means variables
        - Do K-means as long as the max number of iterations is not met AND the absolute value of the
        difference between the previous and current centroid values is > `tol`
        - Set instance variables based on computed values.
        (All instance variables defined in constructor should be populated with meaningful values)
        - Print out total number of iterations K-means ran for
        '''
        if self.num_samps < k:
            raise RuntimeError('Cannot compute kmeans with #data samples < k!')
        if k < 1:
            raise RuntimeError('Cannot compute kmeans with k < 1!')

        centroids = self.initialize(k)
        prev = centroids.copy()
        iteration = 0

        while iteration < max_iter:
            distances = np.array([self.dist_pt_to_centroids(pt, centroids) for pt in self.data])
            labels = np.argmin(distances, axis = 1)

            for i in range(k):
                centroids[i] = np.mean(self.data[labels == i], axis = 0)

            if np.linalg.norm(centroids - prev, ord = p) < tol:
                break
        
            prev = centroids.copy()
            iteration += 1

        self.k = k
        self.centroids = centroids
        self.data_centroid_labels = labels
        self.inertia = np.mean(np.min(distances, axis = 1)) ** 2

        if verbose:
            print(f"Total number of iterations: {iteration}")

        return self.inertia, iteration

    def cluster_batch(self, k=2, n_iter=1, verbose=False, p=2):
        '''Run K-means multiple times, each time with different initial conditions.
        Keeps track of K-means instance that generates lowest inertia. Sets the following instance
        variables based on the best K-mean run:
        - self.centroids
        - self.data_centroid_labels
        - self.inertia

        Parameters:
        -----------
        k: int. Number of clusters
        n_iter: int. Number of times to run K-means with the designated `k` value.
        verbose: boolean. Print out debug information if set to True.
        '''
        best_inertia = float('inf')
        best_centroids = None
        best_labels = None
        best_iteration = None
        
        for _ in range(n_iter):
            inertia, iteration = self.cluster(k = k, tol = 1e-2, max_iter = 1000, verbose = False, p = 2)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = self.centroids.copy()
                best_labels = self.data_centroid_labels.copy()
                best_iteration = iteration

        self.k = k
        self.centroids = best_centroids
        self.data_centroid_labels = best_labels
        self.inertia = best_inertia

        if verbose:
            print(f"Best K-means run: Inertia = {best_inertia}, Iterations = {best_iteration}")

        return self.inertia, best_iteration

    def update_labels(self, centroids):
        '''Assigns each data sample to the nearest centroid

        Parameters:
        -----------
        centroids: ndarray. shape=(k, self.num_features). Current centroids for the k clusters.

        Returns:
        -----------
        ndarray of ints. shape=(self.num_samps,). Holds index of the assigned cluster of each data
            sample. These should be ints (pay attention to/cast your dtypes accordingly).

        Example: If we have 3 clusters and we compute distances to data sample i: [0.1, 0.5, 0.05]
        labels[i] is 2. The entire labels array may look something like this: [0, 2, 1, 1, 0, ...]
        '''
        dis = np.array([self.dist_pt_to_centroids(pt, centroids) for pt in self.data])
        labels = np.argmin(dis, axis = 1)
        return labels

    def update_centroids(self, k, data_centroid_labels, prev_centroids):
        '''Computes each of the K centroids (means) based on the data assigned to each cluster

        Parameters:
        -----------
        k: int. Number of clusters
        data_centroid_labels. ndarray of ints. shape=(self.num_samps,)
            Holds index of the assigned cluster of each data sample
        prev_centroids. ndarray. shape=(k, self.num_features)
            Holds centroids for each cluster computed on the PREVIOUS time step

        Returns:
        -----------
        new_centroids. ndarray. shape=(k, self.num_features).
            Centroids for each cluster computed on the CURRENT time step
        centroid_diff. ndarray. shape=(k, self.num_features).
            Difference between current and previous centroid values

        NOTE: Your implementation should handle the case when there are no samples assigned to a cluster —
        i.e. `data_centroid_labels` does not have a valid cluster index in it at all.
            For example, if `k`=3 and data_centroid_labels = [0, 1, 0, 0, 1], there are no samples assigned to cluster 2.
        In the case of each cluster without samples assigned to it, you should assign make its centroid a data sample
        randomly selected from the dataset.
        '''
        new = np.zeros_like(prev_centroids)
        centroid_diff = np.zeros_like(prev_centroids)

        for i in range(k):
            cluster_data = self.data[data_centroid_labels == i]
            if len(cluster_data) == 0:
                rand = np.random.randint(0, self.num_samps)
                new[i] = self.data[rand]
            else:
                new[i] = np.mean(cluster_data, axis = 0)
            centroid_diff[i] = new[i] - prev_centroids[i]
        
        return new, centroid_diff

    def compute_inertia(self):
        '''Mean squared distance between every data sample and its assigned (nearest) centroid

        Returns:
        -----------
        float. The average squared distance between every data sample and its assigned cluster centroid.
        '''
        ssq = 0.0
        for i in range(self.num_samps):
            centroid_idx = self.data_centroid_labels[i]
            sq = np.sum((self.data[i] - self.centroids[centroid_idx]) ** 2)
            ssq += sq
        inertia = ssq / self.num_samps
        return inertia

    def plot_clusters(self):
        '''Creates a scatter plot of the data color-coded by cluster assignment.

        TODO:
        - Plot samples belonging to a cluster with the same color.
        - Plot the centroids in black with a different plot marker.
        - The default scatter plot color palette produces colors that may be difficult to discern
        (especially for those who are colorblind). To make sure you change your colors to be clearly differentiable,
        use either the Okabe & Ito or one of the Petroff color palettes: https://github.com/proplot-dev/proplot/issues/424
        Each string in the `colors` list that starts with # is the hexadecimal representation of a color (blue, red, etc.)
        that can be passed into the color `c` keyword argument of plt.plot or plt.scatter.
            Pick one of the palettes with a generous number of colors so that you don't run out if k is large (e.g. >6).
        '''
        colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']

        for i in range(self.k):
            samples = self.data[self.data_centroid_labels == i]
            plt.scatter(samples[:, 0], samples[:, 1], color=colors[i], label=f'Cluster {i+1}')

        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], color='black', marker='x', label='Centroids')

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('K-means Clustering')
        plt.legend()
        plt.show()

    def elbow_plot(self, max_k, n_iter = 1):
        '''Makes an elbow plot: cluster number (k) on x axis, inertia on y axis.

        Parameters:
        -----------
        max_k: int. Run k-means with k=1,2,...,max_k.
        n_iter: int. Number of times to run K-means with each k value.

        TODO:
        - Run k-means with k=1,2,...,max_k, record the inertia.
        - Make the plot with appropriate x label, and y label, x tick marks.
        '''
        inertias = []
        for k in range(1, max_k + 1):
            avg = 0
            for _ in range(n_iter):
                self.cluster_batch(k)
                avg += self.compute_inertia()
            inertias.append(avg / n_iter)

        plt.plot(range(1, max_k + 1), inertias, marker='o')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Plot')
        plt.xticks(range(1, max_k + 1))
        plt.show()

    def replace_color_with_centroid(self):
        '''Replace each RGB pixel in self.data (flattened image) with the closest centroid value.
        Used with image compression after K-means is run on the image vector.

        Parameters:
        -----------
        None

        Returns:
        -----------
        None
        '''
        labels = self.update_labels(self.centroids)

        compressed = np.zeros_like(self.data)

        for i, label in enumerate(labels):
            compressed[i] = self.centroids[label]

        return compressed.reshape((self.num_samps, self.num_features))
