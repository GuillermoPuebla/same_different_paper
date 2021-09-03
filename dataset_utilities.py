# Read tfrecord and test dataset.
import tensorflow as tf
import numpy as np

def get_dataset_simulations_1_to_4(batch_size, tfrecord_dir, is_training=True, process_img=True):
    # Load dataset.
    raw_image_dataset = tf.data.TFRecordDataset(tfrecord_dir)
    
    # Define example reading function.
    def read_tfrecord(serialized_example):
        # Create a dictionary describing the features.
        feature_description = {
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'coordinates': tf.io.FixedLenFeature([], tf.string),
            'relative_position': tf.io.FixedLenFeature([], tf.int64)}
        
        # Parse example.
        example = tf.io.parse_single_example(serialized_example, feature_description)

        # Get image.
        image = tf.io.decode_png(example['image_raw'], channels=3)

        # Ensure shape dimensions are constant.
        image = tf.reshape(image, [128, 128, 3])

        # Preprocess image.
        if process_img:
            image = tf.cast(image, tf.float64)
            image /= 255.0
            # Sample-wise center image.
            mean = tf.reduce_mean(image)
            image -= mean
            # Sample-wise std normalization.
            std = tf.math.reduce_std(image)
            image /= std
        
        # Cast label to int64
        label = example['label']
        label = tf.cast(label, tf.int64)

        # Get coordinates.
        b_coors = example['coordinates']
        coors = tf.io.parse_tensor(b_coors, out_type=tf.float64) # restore 2D array from byte string
        coors = tf.reshape(coors, [4])

        # Cast relative position to int64
        rel_pos = example['relative_position']
        rel_pos = tf.cast(rel_pos, tf.int64)

        return image, (label, coors, rel_pos)
        
    # Parse dataset.
    dataset = raw_image_dataset.map(read_tfrecord)
    
    if is_training:
        # I'm shuffling the datasets at creation time, so this is no necesarry for now.
        dataset = dataset.shuffle(11200)
        # Infinite dataset to avoid the potential last partial batch in each epoch.
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)
    
    return dataset

def get_dataset_sort_of_clever(batch_size, tfrecord_dir, is_training=True, process_img=True):
    # Load dataset.
    raw_image_dataset = tf.data.TFRecordDataset(tfrecord_dir)
    
    # Define example reading function.
    def read_tfrecord(serialized_example):
        # Create a dictionary describing the features.
        feature_description = {
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'question': tf.io.FixedLenFeature([], tf.string),
            'answer': tf.io.FixedLenFeature([], tf.string)}
        
        # Parse example.
        example = tf.io.parse_single_example(serialized_example, feature_description)

        # Process image.
        image = tf.io.decode_png(example['image_raw'], channels=3)

        # Ensure shape dimensions are constant.
        image = tf.reshape(image, [128, 128, 3])

        # Preprocess image.
        if process_img:
            image = tf.cast(image, tf.float64)
            image /= 255.0
            # Sample-wise center image.
            mean = tf.reduce_mean(image)
            image -= mean
            # Sample-wise std normalization.
            std = tf.math.reduce_std(image)
            image /= std
        
        # Process question and aswers.
        b_question = example['question'] # get byte string
        question = tf.io.parse_tensor(b_question, out_type=tf.int64)
        question = tf.reshape(question, [11])

        b_answer = example['answer'] # get byte string
        answer = tf.io.parse_tensor(b_answer, out_type=tf.int64)
        answer = tf.reshape(answer, [10])

        return (image, question), answer
        
    # Parse dataset.
    dataset = raw_image_dataset.map(read_tfrecord)
    
    if is_training:
        # I'm shuffling the datasets at creation time, so this is no necesarry for now.
        # dataset = dataset.shuffle(56000)
        # Infinite dataset to avoid the potential last partial batch in each epoch.
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)
    
    return dataset

# Functions to interpret sort-of-clevr questions and answers.
def translate_question(q):
    if len(q) != 11:
        return 'Not a proper question'
    colors = ['red', 'blue', 'green', 'orange', 'yellow', 'gray']
    idx= np.argwhere(q[:6])[0][0]
    color = colors[idx]
    if q[6]:
        if q[8]:
            return "What's the shape of the nearest object to the object in " + color + "?" 
        elif q[9]:
            return "What's the shape of the farthest object away from the object in " + color + "?"
        elif q[10]:
            return 'How many objects have the same shape as the object in ' + color + '?'
    else:
        if q[8]:
            return 'Is the object in ' + color + ' a circle or a square?'
        elif q[9]:
            return 'Is the object in ' + color + ' on the bottom of the image?'
        elif q[10]:
            return 'Is the object in ' + color + ' on the left of the image?'
        
def translate_answer(a):
    if len(a) != 10:
        return 'Not a proper answer'
    if a[0]:
        return 'yes'
    if a[1]:
        return 'no'
    if a[2]:
        return 'square'
    if a[3]:
        return 'circle'
    return np.argwhere(a[4:])[0][0] + 1
