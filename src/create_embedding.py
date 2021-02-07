import os

import cv2
import tensorflow as tf

from face_detector import FaceDetector
from utils import prewhiten, crop_n_align
from global_variables import *

class Embeddings():
    def __init__(self, source_path=None):
        self.source_path = source_path
        if(source_path is None):
            self.source_path = SOURCE_PATH

        path = os.path.join(self.source_path, 'Pretrained-Models', 'facenet_model.pb')
        with tf.io.gfile.GFile(path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        
        graph = tf.Graph()
        with graph.as_default():
            tf.graph_util.import_graph_def(graph_def, name='')
        
        self._images_placeholder = graph.get_tensor_by_name('input:0')
        self._phase_train_placeholder = graph.get_tensor_by_name('phase_train:0')
        self._embedding = graph.get_tensor_by_name('embeddings:0')
        
        self._sess = tf.compat.v1.Session(graph=graph)

        self._detect = FaceDetector(self.source_path)
        
    def __call__(self, img):
        assert (img.shape[0] == 160 and img.shape[1] == 160)
        
        prewhiten_face = prewhiten(img)
        
        feed_dict = {self._images_placeholder: [prewhiten_face], self._phase_train_placeholder: False}
        return self._sess.run(self._embedding, feed_dict=feed_dict)[0]
    
    def generate_embeddings(self, image_path=None, image_array=None, is_rotate=False, verbose=False):         

        if(image_array is None and image_path is None):
            raise AssertionError("Both array and path are None")

        if(image_array is not None):
            image = image_array
            image_path = None

        if(image_array is None and image_path is not None):
            image = cv2.imread(image_path)
            if(image is None):
                raise AssertionError("Invalid image path")

        faces = self._detect.detect_extract_faces(image)
        emb = []

        for face in faces:
            align = crop_n_align(face, is_rotate=is_rotate, verbose=verbose)
            emb.append(self.__call__(align))
            
        return emb