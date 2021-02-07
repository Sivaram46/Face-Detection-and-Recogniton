import os
import glob

import numpy as np
import cv2

from face_detector import FaceDetector
from create_embedding import Embeddings
from global_variables import *
from utils import crop, crop_n_align

class Verifier():
    def __init__(self, source_path=None):
        self.source_path = source_path
        if(source_path is None):
            self.source_path = SOURCE_PATH

        self._detector = FaceDetector(self.source_path)
        self._embed = Embeddings(self.source_path)
    
    def flow_from_directory(self, path, random_flip=None, random_rotate=None, verbose=False):
        assert os.path.isdir(path) == True, "Invalid Directory"

        if(verbose):
            print(f"Reading in {path} directory...")

        images = []
        self.label_name = os.path.basename(path)

        files = glob.glob(os.path.join(path, '*.jpg'))
        if(verbose):
            print("Readed files...")

        for file in files:
            if(verbose):
                print('\t', file.split('\\')[-1])

            img = cv2.imread(file)
            img = crop(img)
            img = self._detector.detect_extract_faces(img)[0]
            img = crop_n_align(img, is_rotate=True)
            images.append(img)

        print(f"Found {len(images)} images")

        if(random_flip is not None):
            p = np.random.permutation(len(images)) [:int(random_flip * len(images))]
            fliped = [np.flip(images[i], axis=1) for i in p]
        
        if(random_rotate is not None):
            p = np.random.permutation(len(images))[:int(random_rotate * len(images))]
            rotated = []
            for i in p:
                angle = np.random.randint(-15, 15)
                rows, cols, _ = images[i].shape
                M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
                rotated.append(cv2.warpAffine(images[i], M, (cols, rows), borderMode=cv2.BORDER_REPLICATE))

        if(random_flip is not None):
            images += fliped
        if(random_rotate is not None):
            images += rotated

        print(f"After augmentation: {len(images)}")

        X = [self._embed(img) for img in images]

        self.X = np.array(X)
    
    def predict(self, X, thresh=.85, tol=None, tol_c=3, is_rotate=True, verbose=False):
        # thresold of .85 works fine 
        m = np.mean(self.X, axis=0)
        dist_X = np.linalg.norm(self.X - m, axis=1)

        if(tol is None):
            tol = tol_c * np.std(dist_X)

        if(verbose):
            print("Setted threshold for prediction is: ", thresh)
            print("Setted tolerance for prediction is: ", tol)
            print("Threshold + Tolerance: ", thresh + tol)

        pred = []
        dist_pred = []

        for x in X:
            d = np.linalg.norm(m - x)
            if(d <= thresh + tol): pred.append(1)
            else: pred.append(0)
            dist_pred.append(d)

        return pred, dist_pred