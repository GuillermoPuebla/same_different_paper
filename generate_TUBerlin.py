import io
import os
import random
import itertools
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import math
import requests, zipfile

# Helper functions.

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array

def thin_image(img):
    inv = cv2.bitwise_not(img)
    thinned = cv2.ximgproc.thinning(inv)
    return cv2.bitwise_not(thinned)

def sq_point_in_circle(r, center=(0, 0)):
    """
    Generate a random point in an r radius circle 
    centered around the start of the axis
    """

    t = 2*math.pi*np.random.uniform()
    R = (np.random.uniform(0,1) ** 0.5) * r

    return [R*math.cos(t)+center[0], R*math.sin(t)+center[1]]

def sample_points_circle(r=384, center=(576, 576), n=4):
    points = [sq_point_in_circle(r=r, center=center) for i in range(n)]
    return np.array(points)
    
def get_augmented_imgs(file_path, n=60):
    """Returns a list of n resized and augmented images."""
    
    # Read image.
    img = cv2.imread(file_path)

    # Add border such that img size is a multiple of 128.
    img = cv2.copyMakeBorder(img,20,21,20,21,cv2.BORDER_CONSTANT,value=[255,255,255])

    # Convert to grayscale.
    original_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get combinations of transformations.
    flips = [True, False]
    rotations = list(range(-25, 25))
    translations_x = list(range(-40, 40))
    translations_y = list(range(-40, 40))

    # Get all combinations of transformations.
    transorms = list(itertools.product(flips, rotations, translations_x, translations_y))

    # Sample without reposition n transformations.
    transorms = random.sample(transorms, n)

    # Apply transformations.
    transformed_imgs = []
    for transorm in transorms:
        # Flip images.
        flipped = cv2.flip(original_gray, 1) if transorm[0] else original_gray

        # Rotate original.
        rot_m = cv2.getRotationMatrix2D((1111/2, 1111/2), transorm[1], 1.0)
        rot = cv2.warpAffine(flipped, rot_m, (1111, 1111), borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
        
        # Translate images.
        t_matrix = np.float32([[1, 0, transorm[2]], [0, 1, transorm[3]]])
        trans = cv2.warpAffine(rot, t_matrix, (1111, 1111), borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
        
        # Resize, threshold and thin rotated images.
        res = cv2.resize(trans, dsize=(128, 128), interpolation=cv2.INTER_AREA)
        thres = cv2.adaptiveThreshold(res, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        thin = thin_image(thres)

        # Append final image.
        transformed_imgs.append(thin)

    # Make shure that the original image is in the sample.
    transformed_imgs.pop()
    res = cv2.resize(original_gray, dsize=(128, 128), interpolation=cv2.INTER_AREA)
    thres = cv2.adaptiveThreshold(res, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    thin = thin_image(thres)
    transformed_imgs.append(thin)

    return transformed_imgs

def get_gray_rescaled_img(file_path):
    # Read image.
    img = cv2.imread(file_path)

    # Add border such that img size is a multiple of 128.
    img = cv2.copyMakeBorder(img,20,21,20,21,cv2.BORDER_CONSTANT,value=[255,255,255])

    # Convert to grayscale.
    original_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize, threshold and thin rotated images.
    res = cv2.resize(original_gray, dsize=(128, 128), interpolation=cv2.INTER_AREA)
    thres = cv2.adaptiveThreshold(res, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    thin = thin_image(thres)

    return thin

# Check original and resized images.
def check_images(file_path):
    transformed_imgs = get_augmented_imgs(file_path, n=10)
    fig, axs = plt.subplots(2, 5, sharex=False, sharey=False, figsize=(30, 12))
    i = 0
    for i, ax in enumerate(fig.axes):
        ax.imshow(transformed_imgs[i], cmap='gray', vmin=0, vmax=255, interpolation='none')
        i += 1

    plt.savefig("output.jpg") #save as png
    plt.show()

# Make dictionary from classes names to indexes.
def make_class_to_indx_dict(filelist_path):
    with open(filelist_path, 'r') as tf:
        lines = tf.read().split('\n')
    class_names = list(set([x.split('/')[0] for x in lines if len(x) > 0]))
    class_names.sort()
    return {k: v for v, k in enumerate(class_names)}

def indx_to_onehot(indx, size=250):
    return np.eye(size)[indx]


if __name__ == '__main__':
    # Download and extract dataset.
    zip_url = "http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_png.zip"
    request = requests.get(zip_url)
    zipDocument = zipfile.ZipFile(io.BytesIO(request.content))
    zipDocument.extractall()

    # Make dictionary from classes names to indexes.
    filelist_path =  'png/filelist.txt'
    class_to_indx = make_class_to_indx_dict(filelist_path)

    # Get sample 64 from 80 (20%) for training.
    train_indxs = random.sample(range(1, 81), 64)
    test_indxs = [x for x in range(1, 81) if x not in train_indxs]

    # Augment train and test indexes for the 250 classes.
    train_indxs = [x + y for x in train_indxs for y in range(0, 80*250, 80)]
    test_indxs = [x + y for x in test_indxs for y in range(0, 80*250, 80)]

    # Iterate over subfolders.
    with open(filelist_path) as f:
        files = [line.rstrip('\n') for line in f]

    root = 'png/'

    # Train dataset.
    tfrecord_file_name = "data/TUBerlin_train.tfrecords"
    with tf.io.TFRecordWriter(tfrecord_file_name) as writer:
        random.shuffle(files)
        for filename in files:
            if filename.endswith(".png"):
                label_str = filename.split('/')[0]
                label_indx = class_to_indx[label_str]

                img_name_end = filename.split('/')[1]
                img_number = int(img_name_end.split('.')[0])
                
                if img_number in train_indxs:
                    filepath = os.path.join(root, filename)
                    agumented_imgs = get_augmented_imgs(filepath) # 80*60*0.8 imgs per class
                    for img in agumented_imgs:
                        # Save image to bytes.
                        pil_image=Image.fromarray(img)
                        buf = io.BytesIO()
                        pil_image.save(buf, format='PNG')
                        byte_im = buf.getvalue()
                        # Parse example.
                        feature = {
                        'label': _int64_feature(label_indx),
                        'image_raw': _bytes_feature(byte_im)}
                        # Parse example.
                        example = tf.train.Example(features=tf.train.Features(feature=feature))
                        # Write example.
                        writer.write(example.SerializeToString())
    print('This is going to take a while...')
    print('Train dataset done! Starting training dataset...')

    # Test dataset.
    tfrecord_file_name = "data/TUBerlin_test.tfrecords"
    with tf.io.TFRecordWriter(tfrecord_file_name) as writer:
        random.shuffle(files)
        for filename in files:
            if filename.endswith(".png"):
                label_str = filename.split('/')[0]
                label_indx = class_to_indx[label_str]

                img_name_end = filename.split('/')[1]
                img_number = int(img_name_end.split('.')[0])
                
                if img_number in test_indxs:
                    filepath = os.path.join(root, filename)
                    # agumented_imgs = get_augmented_imgs(filepath, n=50) # 0.2*4000 per class
                    resized_img = [get_gray_rescaled_img(filepath)]
                    for img in resized_img:
                        # Save image to bytes.
                        pil_image=Image.fromarray(img)
                        buf = io.BytesIO()
                        pil_image.save(buf, format='PNG')
                        byte_im = buf.getvalue()
                        # Parse example.
                        feature = {
                        'label': _int64_feature(label_indx),
                        'image_raw': _bytes_feature(byte_im)}
                        # Parse example.
                        example = tf.train.Example(features=tf.train.Features(feature=feature))
                        # Write example.
                        writer.write(example.SerializeToString())
    print('This is going to take a while...')
    print('Test dataset done!')
