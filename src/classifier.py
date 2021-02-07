import os
import glob
import pickle

import numpy as np
import cv2

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from create_embedding import Embeddings
from global_variables import *

class Classifier():
    def __init__(self, source_path=None):
        self._source_path = source_path
        if(source_path is None):
            self._source_path = SOURCE_PATH

        self.clf = make_pipeline(StandardScaler(), SVC())
        
        self._embed = Embeddings(self._source_path)

        self.label_names = []
        
    def flow_from_directory(self, path, model_name=None, verbose=False):

        assert os.path.isdir(path) == True, "Invalid Directory"

        if(verbose):
            print(f"Reading in {path} directory...")

        i = 0
        X = []
        y = []
        
        it = os.scandir(path)
        if(verbose):
            print("Readed folders/files:")

        for entry in it:
            if(entry.is_dir()):
                if(verbose):
                    print('\t', entry.name)

                self.label_names.append(entry.name)
                
                i += 1
                files = glob.glob(os.path.join(path, entry.name, '*.jpg'))
                for file in files:
                    if(verbose):
                        print('\t\t', file.split('\\')[-1])

                    im = cv2.imread(file)
                    X.append(self._embed.generate_embeddings(image_array=im)[0])
                    y.append(i)
            
        X = np.array(X)
        y = np.array(y)

        p = np.random.permutation(len(y))
        self.X = X[p]
        y = y[p]

        self.clf.fit(self.X, y)

        if(model_name is not None):
            with open(model_name, 'wb') as f:
                pickle.dump(self.clf, f)

    def predict(self, image_array=None, image_path=None, is_rotate=True, saved_model_path=None):

        if(saved_model_path is not None):
            with open(saved_model_path, 'rb') as f:
                self.clf = pickle.load(f)

        emb = self._embed.generate_embeddings(image_array=image_array, image_path=image_path, is_rotate=is_rotate)
        
        if(len(self.label_names) > 0):
            return self.label_names[self.clf.predict(emb)[0]]
        else:
            return self.clf.predict(emb)[0]
