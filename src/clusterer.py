import os
import glob
import pickle

import numpy as np
import cv2

from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.metrics import silhouette_score

from face_detector import FaceDetector
from create_embedding import Embeddings
from global_variables import *
from utils import crop, draw_boxes, plot_img

def find_image_pos(cum_sum, pos):
    n = cum_sum.searchsorted(pos, side='right')
    if(n == 0): return (n, pos)
    else: return (n, pos - cum_sum[n-1])

class Clusterer():

    def __init__(self, source_path=None):
        self._source_path = source_path
        if(source_path is None):
            self._source_path = SOURCE_PATH

        # self._source_path = "C:\\Users\\sivaram\\Documents\\FDR"
        self._detector = FaceDetector(self._source_path)
        self._embed = Embeddings(self._source_path)
        
    def flow_from_directory(self, path, n_clusters=None, max_clusters=None, algorithm='kmeans', 
                            eps=.85, min_samples=3, score_threshold=.5, verbose=False):

        assert os.path.isdir(path) == True, "Invalid Directory"

        if(verbose):
            print(f"Reading in {path} directory...")

        self.images = []; self.boxes = []; scores = []; self.embeddings = []; num_faces = []

        files = glob.glob(os.path.join(path, '*.jpg'))
        if(verbose):
            print("Readed files...")

        if(algorithm not in ['kmeans', 'spectral', 'dbscan']):
            raise AssertionError("Invalid algorithm")
        models = {'kmeans': KMeans, 'spectral': SpectralClustering, 'dbscan': DBSCAN}
        
        for file in files:
            if(verbose):
                print('\t', file.split('\\')[-1])

            im = cv2.imread(file)
            im = crop(im)
            box, scr = self._detector(im, score_threshold=score_threshold)

            if(verbose):
                print(f"\t\tFound {box.shape[0]} faces with {score_threshold} threshold")

            self.boxes.append(box)
            scores.append(scr)
            self.images.append(im)
            num_faces.append(box.shape[0])

            self.embeddings += self._embed.generate_embeddings(image_array=im, is_rotate=True, verbose=verbose)

        if(verbose):
            print(f"\nTotal of {len(self.embeddings)} faces found")

        if(max_clusters is None):
            max_clusters = len(self.embeddings)

        if(n_clusters is None and algorithm != 'dbscan'):
            scores = []
            for k in range(2, max_clusters):
                model = models[algorithm](n_clusters=k, random_state=10).fit(self.embeddings)
                scores.append(silhouette_score(self.embeddings, model.labels_, metric='euclidean'))

            n_clusters = np.argmin(np.abs(scores)) + 2

        if(verbose):
            print(f"\nClustered with {n_clusters} clusters")

        if(algorithm == 'dbscan'):
            self.model = models[algorithm](eps=eps, min_samples=min_samples).fit(self.embeddings)
        else:
            self.model = models[algorithm](n_clusters=n_clusters, random_state=10).fit(self.embeddings)

    def tag_similar_faces(self, labels=None):
        if(labels is None):
            labels = self.model.labels_

        cum_sum = np.cumsum([bx.shape[0] for bx in self.boxes])

        result = [None] * len(self.images)
        for i, l in enumerate(labels):
            im_off, face_off = find_image_pos(cum_sum, i)
            bx = np.array([self.boxes[im_off][face_off]])
            if(result[im_off] is None):
                res = draw_boxes(self.images[im_off], bx, labels=[str(l)])
            else:
                res = draw_boxes(res, bx, labels=[str(l)])
            result[im_off] = res

        # for printing purposes
        result_ = list(map(lambda im: cv2.resize(im, (0,0), fx=.5, fy=.5), res))

        plot_img(result_)