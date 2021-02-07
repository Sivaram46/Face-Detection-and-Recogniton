import os

import numpy as np
import cv2
import tensorflow as tf

from global_variables import *
from utils import draw_boxes

class FaceDetector:
    def __init__(self, source_path=None):
        """
        Parameters
        ----------
        model_path : A String
            Source directory where the project is stored. If None current directory
            will be taken as source directory.

        Returns
        -------
        None.

        """
        if(source_path is None):
            source_path = SOURCE_PATH

        path = os.path.join(source_path, 'Pretrained-Models', 'faceboxes_model.pb')
        with tf.io.gfile.GFile(path, mode='rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        
        graph = tf.Graph()
        with graph.as_default():
            tf.graph_util.import_graph_def(graph_def, name='import')
        
        self._input_image = graph.get_tensor_by_name('import/image_tensor:0')
        self._output_ops = [
            graph.get_tensor_by_name('import/boxes:0'),
            graph.get_tensor_by_name('import/scores:0'),
            graph.get_tensor_by_name('import/num_boxes:0') ]
        self._sess = tf.compat.v1.Session(graph=graph)
    
    def __call__(self, img, score_threshold=0.5):
        """
        Detect faces from an image.

        Parameters
        ----------
        img : A numpy array
            An input array in which faces to be detected.
        score_threshold : A float, optional
            Threshold score. The default is 0.5.

        Returns
        -------
        boxes : A list of arrays of dimenstion #faces, 4
            Coordinates of boxes.
            Note that box coordinates are in the order: ymin, xmin, ymax, xmax
        scores : A list of arrays of dimenstion #faces, 4
            Probability for the detected face.

        """

        h, w, _ = img.shape
        img = np.expand_dims(img, 0)

        boxes, scores, num_boxes = self._sess.run(
            self._output_ops,
            feed_dict={self._input_image: img}
        )

        num_boxes = num_boxes[0]
        boxes = boxes[0][:num_boxes]
        scores = scores[0][:num_boxes]

        to_keep = scores > score_threshold
        boxes = boxes[to_keep]
        scores = scores[to_keep]

        scaler = np.array([h,w,h,w], dtype=np.float32)
        boxes *= scaler

        return boxes, scores
    
    def detect_extract_faces(self, img, score_threshold=.5):
        """
        Detect and extract individual faces from an image of group of people
        Parameters
        ----------
        img : A numpy array
            The input image from which faces to be extracted.
        score_threshold : A float, optional
            Threshold score. The default is 0.5.
        Returns
        -------
        faces : A list of numpy arrays
            Each element in the list is an image.
        """
        boxes, _ = self.__call__(img, score_threshold)
        num_faces = boxes.shape[0]
        faces = []
        
        for i in range(num_faces):
            ymin, xmin, ymax, xmax = boxes[i].astype('int')
            face = img[ymin:ymax, xmin:xmax]
            faces.append(face)
        
        return faces

def main():
    
    MODEL_PATH = "C:\\Users\\sivaram\\Documents\\Face detection and recognition\\FaceBoxes-tensorflow-master\\faceboxes_model.pb"
    face_detector = FaceDetector(MODEL_PATH)

    img_path = "C:\\Users\\sivaram\\Documents\\Face detection and recognition\\img_2.jpg"
    image_array = cv2.imread(img_path)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        
    boxes, scores = face_detector(image_array, score_threshold=0.3)
    image_copy = draw_boxes(image_array, boxes, scores)
    # image_copy.show()
    
    cv2.imshow('img', image_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if(__name__ == '__main__'):
    main()

