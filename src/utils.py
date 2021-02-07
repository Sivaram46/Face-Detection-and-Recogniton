import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def draw_boxes(image, boxes, scores=None, labels=None):
    """
    Draw bounding boxes for each face in the image.
    Parameters
    ----------
    image : A numpy array or PIL Image object.
        The input image for which the bounding box to be drawn.
    boxes : A List.
        Coordinates for the bounding boxes list returns by an instance of FaceDetector.
    scores : A list, optional
        Scores for each bouding boxes returns by an instance of FaceDetector. The default is None.
    Returns
    -------
    A numpy array
        Array of bouding box drawned image.
    """

    if(isinstance(image, np.ndarray)):
        image = Image.fromarray(image)
        
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy, 'RGBA')
    font = ImageFont.truetype('arial.ttf', 100)
    
    for i, b in enumerate(boxes):
        ymin, xmin, ymax, xmax = b
        fill = (255, 0, 0, 45)
        outline = 'red'
        draw.rectangle(
            [(xmin, ymin), (xmax, ymax)],
            fill=fill, outline=outline
        )

        if(labels is not None):
            draw.text((xmin, ymin), text=labels[i], font=font)
        
        if(scores is not None):
            draw.text((xmin, ymin), text='{:.3f}'.format(scores[i]))
    
    return np.array(image_copy)

def extract_faces(img, boxes):
    """
    Extract individual faces from an image of group of people
    Parameters
    ----------
    img : A numpy array
        The input image from which faces to be extracted.
    boxes : A list
        Coordinates for the bounding boxes list returns by an instance of FaceDetector.
    Returns
    -------
    faces : A list of numpy arrays
        Each element in the list is an image.
    """
    num_faces = boxes.shape[0]
    faces = []
    
    for i in range(num_faces):
        ymin, xmin, ymax, xmax = boxes[i].astype('int')
        face = img[ymin:ymax, xmin:xmax]
        faces.append(face)
    
    return faces

def crop_n_align(img, output_shape=(160, 160), is_rotate=True, verbose=False, plot_eyes=False):
    """
    For aligning the picture which is best to be fed to Facenet.
    Aligning means, make the eyes horizontal, resizing the image proportionally.
    
    Parameters
    ----------
    img : TYPE
        DESCRIPTION.

    Returns
    -------
    A numpy array of rotated and resized image.

    """
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    resized = cv2.resize(img, output_shape, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray) # returns x, y, w, h for two eyes

    if(is_rotate is False):
        return resized
    
    if(len(eyes) < 2):
        if(verbose):
            print("Unable to detect eyes")
        return resized

    eye1 = eyes[0]
    eye2 = eyes[1]

    dist = np.linalg.norm(np.array(eye1) - np.array(eye2))
    dist /= output_shape[0]

    prop1 = eye1[1] / output_shape[1]
    prop2 = eye2[1] / output_shape[1]

    # print(dist, prop1, prop2)

    if(dist < .4 or dist > .6 or prop1 < .1 or prop1 > .5 or prop2 < .1 or prop2 > .5):
        if(verbose):
            print("Eye detection doesn't seem to work well")
        return resized

    # finding left and right eyes 
    if(eye1[0] < eye2[0]):
        left_eye = eye1
        right_eye = eye2
    else:
        left_eye = eye2
        right_eye = eye1
    
    # finding center of eyes
    left_eye_center_x = left_eye[0] + left_eye[2] // 2
    left_eye_center_y = left_eye[1] + left_eye[3] // 2
    right_eye_center_x = right_eye[0] + right_eye[2] // 2
    right_eye_center_y = right_eye[1] + right_eye[3] // 2
    
    dx = (right_eye_center_x - left_eye_center_x)
    dy = (right_eye_center_y - left_eye_center_y)
    
    angle = np.degrees(np.arctan(dy/dx))

    if(verbose):
        print(f"Rotated {round(angle, 2)} degrees")
    
    if(plot_eyes):
        cv2.rectangle(resized, (left_eye[0], left_eye[1]), (left_eye[0]+left_eye[2], left_eye[1]+left_eye[3]), (0,0,255), 3)
        cv2.rectangle(resized, (right_eye[0], right_eye[1]), (right_eye[0]+right_eye[2], right_eye[1]+right_eye[3]), (0,0,255), 3)
        cv2.circle(resized, (left_eye_center_x, left_eye_center_y), 5, (255, 0, 0) , -1)
        cv2.circle(resized, (right_eye_center_x, right_eye_center_y), 5, (255, 0, 0) , -1)
    
    h, w = resized.shape[:2]
    center = (w // 2, h // 2)
    
    # getRotationMatrix2D returns a 2*3 matrix used for affine transfrom
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    rotated = cv2.warpAffine(resized, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    return rotated

def plot_img(img):
    """
    Plot images
    Parameters
    ----------
    img : A numpy array or List of numpy arrays
        The input image(s) to be plotted.
    Returns
    -------
    None.
    """
    if isinstance(img, np.ndarray):
        cv2.imshow('img', img)
        
    if isinstance(img, list):
        for i, j in enumerate(img):
            cv2.imshow(str(i+1), j)
    cv2.waitKey(0)
    cv2.destroyAllWindows

def prewhiten(x):
    """
    Pre-whiten an image for faster learning
    Parameters
    ----------
    x : A numpy array
        The input image.
    Returns
    -------
    y : A numpy array
        A pre-whitened image.
    """
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

def crop(img, min_dim=1000):
    s = img.shape[:2]
    ind = np.argmin(s)
    x = int(max(s)*min_dim/min(s))

    if(ind == 0):
        return cv2.resize(img, (x, min_dim))
    else:
        return cv2.resize(img, (min_dim, x))

def hist_equalize(img, format='RGB'):
    if(format == 'RGB'): y = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
    else: y = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    
    chan = cv2.split(y)
    cv2.equalizeHist(chan[0], chan[0])
    cv2.merge(chan, y)

    if(format == 'RGB'): img = cv2.cvtColor(y, cv2.COLOR_YCR_CB2RGB)
    else: img = cv2.cvtColor(y, cv2.COLOR_YCR_CB2BGR)

    return img
