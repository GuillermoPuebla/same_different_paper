# Imports.
import os
import numpy as np
import pandas as pd

import math
import copy
import random
import cv2
import statistics

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Lambda, Concatenate, Average, Conv2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import ResNet50

# Relation Network functions.
def bn_layer(x, kernel_size, stride):
    '''Convolutional layers with batch normalization and RELU activation'''
    def f(inputs):
        md = Conv2D(x, (kernel_size, kernel_size), strides=(stride, stride), padding='same', activation='relu')(inputs)
        md = BatchNormalization(momentum=0.9)(md)
        return md
    return f

def conv_net(inputs):
    '''Batch normalization layer, RELU activation'''
    model = bn_layer(24, 3, 2)(inputs)
    model = bn_layer(24, 3, 2)(model)
    model = bn_layer(24, 3, 2)(model)
    model = bn_layer(24, 3, 2)(model)
    return model

def get_dense(n, units):
    r = []
    for k in range(n):
        r.append(Dense(units, activation='relu'))
    return r

def get_MLP(n, denses):
    def g(x):
        d = x
        for k in range(n):
            d = denses[k](d)
        return d
    return g

def dropout_dense(x, units):
    y = Dense(units, activation='relu')(x)
    y = Dropout(0.5)(y)
    return y

def build_tag(conv):
    d = K.int_shape(conv)[2]
    tag = np.zeros((d,d,2))
    for i in range(d):
        for j in range(d):
            tag[i,j,0] = float(int(i%d))/(d-1)*2-1
            tag[i,j,1] = float(int(j%d))/(d-1)*2-1
    tag = K.variable(tag)
    tag = K.expand_dims(tag, axis=0)
    batch_size = K.shape(conv)[0]
    tag = K.tile(tag, [batch_size,1,1,1])
    return tag

def slice_1(t):
    return t[:, 0, :, :]

def slice_2(t):
    return t[:, 1:, :, :]

def slice_3(t):
    return t[:, 0, :]

def slice_4(t):
    return t[:, 1:, :]

def make_soc_baseline_model():
    # Inputs.
    image = Input((128, 128, 3))
    question = Input((11,))
    
    # Get CNN features.
    cnn_features = conv_net(image)
    
    # Concatenate CNN features and question.
    cnn_features = Flatten()(cnn_features)
    cnn_with_question = Concatenate()([cnn_features, question])
    
    # g function.
    g_MLP = get_MLP(4, get_dense(4, units=256))
    hidden = g_MLP(cnn_with_question)
    
    # f function.
    hidden = Dense(256, activation='relu')(hidden)
    hidden = dropout_dense(hidden, units=256)
    
    # Output.
    answer = Dense(10, activation='softmax')(hidden)
    model = Model(inputs=[image, question], outputs=answer)
    
    return model

def make_original_relnet(task='sort-of-clevr'):
    """
    Args:
        task: sort-of-clevr or same-different.
    """
    # Inputs.
    image = Input((128, 128, 3))
    
    if task=='sort-of-clevr':
        question = Input((11,))

    # Get CNN features.
    cnn_features = conv_net(image)

    # Make tag and append to cnn features.
    tag = build_tag(cnn_features)
    cnn_features = Concatenate()([cnn_features, tag])

    # Make list with objects.
    shapes = cnn_features.shape
    w, h = shapes[1], shapes[2]
            
    slice_layer1 = Lambda(slice_1)
    slice_layer2 = Lambda(slice_2)
    slice_layer3 = Lambda(slice_3)
    slice_layer4 = Lambda(slice_4)

    features = []
    for k1 in range(w):
        features1 = slice_layer1(cnn_features)
        cnn_features = slice_layer2(cnn_features)
        for k2 in range(h):
            features2 = slice_layer3(features1)
            features1 = slice_layer4(features1)
            features.append(features2)
        
    # Make list with all combinations of objects.
    relations = []
    concat = Concatenate()
    if task=='sort-of-clevr':
        for feature1 in features:
            for feature2 in features:
                relations.append(concat([feature1, feature2, question]))
    elif task == 'same-different':
        for feature1 in features:
            for feature2 in features:
                relations.append(concat([feature1, feature2]))

    # g function.
    g_MLP = get_MLP(4, get_dense(4, units=256))
    mid_relations = []
    for r in relations:
        mid_relations.append(g_MLP(r))
    combined_relation = Average()(mid_relations)

    # f function.
    rn = Dense(256, activation='relu')(combined_relation)
    rn = dropout_dense(rn, units=256)
    
    # Output.
    if task == 'sort-of-clevr':
        output_units = 10
        answer = Dense(output_units, activation='softmax')(rn)
        model = Model(inputs=[image, question], outputs=answer)
    elif task == 'same-different':
        output_units = 1
        answer = Dense(output_units, activation='sigmoid')(rn)
        model = Model(inputs=[image], outputs=answer)
    
    return model

def make_ResNet50_relnet(task='sort-of-clevr', trainable=False):
    # Inputs.
    image = Input((128, 128, 3))
    
    if task=='sort-of-clevr':
        question = Input((11,))

    # Get CNN features.
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=image)
    # To get a specific layer:
    layer_name = 'conv5_block3_out'
    cnn_features = base_model.get_layer(layer_name).output
    
    # Freeze the base_model.
    base_model.trainable = trainable
    
    # Make tag and append to cnn features.
    tag = build_tag(cnn_features)
    cnn_features = Concatenate()([cnn_features, tag])
    
    # Make list with objects.
    shapes = cnn_features.shape
    w, h = shapes[1], shapes[2]
    
    slice_layer1 = Lambda(slice_1)
    slice_layer2 = Lambda(slice_2)
    slice_layer3 = Lambda(slice_3)
    slice_layer4 = Lambda(slice_4)

    features = []
    for k1 in range(w):
        features1 = slice_layer1(cnn_features)
        cnn_features = slice_layer2(cnn_features)
        for k2 in range(h):
            features2 = slice_layer3(features1)
            features1 = slice_layer4(features1)
            features.append(features2)

    # Make list with all combinations of objects.
    relations = []
    concat = Concatenate()
    if task=='sort-of-clevr':
        for feature1 in features:
            for feature2 in features:
                relations.append(concat([feature1, feature2, question]))
    elif task == 'same-different':
        for feature1 in features:
            for feature2 in features:
                relations.append(concat([feature1, feature2]))

    # g function.
    g_MLP = get_MLP(4, get_dense(4, units=512))
    mid_relations = []
    for r in relations:
        mid_relations.append(g_MLP(r))
    combined_relation = Average()(mid_relations)

    # f function.
    rn = Dense(512, activation='relu')(combined_relation)
    rn = dropout_dense(rn, units=512)
    
    # Output.
    if task == 'sort-of-clevr':
        output_units = 10
        answer = Dense(output_units, activation='softmax')(rn)
        model = Model(inputs=[image, question], outputs=answer)
    elif task == 'same-different':
        output_units = 1
        answer = Dense(output_units, activation='sigmoid')(rn)
        model = Model(inputs=[image], outputs=answer)
    
    return base_model, model

def get_train_dataset(batch_size,
                      tfrecord_dir,
                      is_training=True,
                      normalize_img=True,
                      samplewise_center=True,
                      samplewise_std_normalization=True):
    # Load dataset.
    raw_image_dataset = tf.data.TFRecordDataset(tfrecord_dir)
  
    # Define example reading function.
    def read_tfrecord(serialized_example):
        # Create a dictionary describing the features.
        feature_description = {'image_raw': tf.io.FixedLenFeature([], tf.string),
                               'question': tf.io.FixedLenFeature([], tf.string),
                               'answer': tf.io.FixedLenFeature([], tf.string)}

        # Parse example.
        example = tf.io.parse_single_example(serialized_example, feature_description)

        # Get image.
        image = tf.io.decode_png(example['image_raw'], channels=3)

        # Ensure shape dimensions are constant.
        image = tf.reshape(image, [128, 128, 3])

        # Preprocess image.
        if normalize_img:
            image = tf.cast(image, tf.float64)
            image /= 255.0
        if samplewise_center:
            # Sample-wise center image.
            mean = tf.reduce_mean(image)
            image -= mean
        if samplewise_std_normalization:
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
    train_dataset = raw_image_dataset.map(read_tfrecord)
  
    if is_training:
        train_dataset = train_dataset.shuffle(4000, reshuffle_each_iteration=True) # test dataset size is 200*20
        # Infinite dataset to avoid the potential last partial batch in each epoch.
        train_dataset = train_dataset.repeat()

    train_dataset = train_dataset.batch(batch_size)
  
    return train_dataset

# Function to train models.
def train_and_save_original_relnet(
    train_data,
    steps,
    epochs,
    lr):

    # Get model.
    relnet = make_original_relnet(task='sort-of-clevr')

    # Compile.
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    relnet.compile(loss='categorical_crossentropy',
                 optimizer=opt,
                 metrics=['accuracy'])

    # Train.
    log_name = 'benchmark/sort_of_clevr_baseline_log.csv'
    history_logger=tf.keras.callbacks.CSVLogger(log_name, separator=",", append=True)
    relnet.fit(
        train_data,
        steps_per_epoch=steps,
        epochs=epochs,
        callbacks=[history_logger])
    
    # Save weights.
    relnet.save_weights('benchmark/sort_of_clevr_baseline_weights.h5')
    return

def train_and_save_soc_baseline(
    train_data,
    steps,
    epochs,
    lr):

    # Get model.
    model = make_soc_baseline_model()

    # Compile.
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'])

    # Train.
    log_name = 'benchmark/sort_of_clevr_original_relation_network_log.csv'
    history_logger=tf.keras.callbacks.CSVLogger(log_name, separator=",", append=True)
    model.fit(
        train_data,
        steps_per_epoch=steps,
        epochs=epochs,
        callbacks=[history_logger])
    
    # Save weights.
    model.save_weights('benchmark/sort_of_clevr_original_relation_network_weights.h5')
    return
    
def train_and_save_ResNet50_relnet(
    train_data,
    epochs_top,
    steps,
    epochs,
    full_model_lr):

    # Define model.
    base_model, train_model = make_ResNet50_relnet(task='sort-of-clevr', trainable=False)

    # Fine-tuning: train the top layer
    lr = 0.0003
    train_model.compile(loss='categorical_crossentropy',
                        optimizer=tf.keras.optimizers.Adam(lr),
                        metrics=['accuracy'])

    train_model.fit(train_data,
                    steps_per_epoch=steps,
                    epochs=epochs_top)

    # Fine-tuning: train the entire model with a very low learning rate.
    base_model.trainable = True

    train_model.compile(loss='categorical_crossentropy',
                        optimizer=tf.keras.optimizers.Adam(full_model_lr),  # Low learning rate.
                        metrics=['accuracy'])
    
    # Train.
    log_name = 'benchmark/sort_of_clevr_ResNet50_relation_network_log.csv'
    history_logger=tf.keras.callbacks.CSVLogger(log_name, separator=",", append=True)
    train_model.fit(
        train_data,
        steps_per_epoch=steps,
        epochs=epochs,
        callbacks=[history_logger])

    train_model.save_weights('benchmark/sort_of_clevr_ResNet50_relation_network_weights.h5')
    return

# Test models function.
def test_all_models(filename, batch_size):
    # Get test datasets.
    path_nonrel = 'data/sort_of_clevr_128_non_relational_test.tfrecords'
    nonrel_ds = get_train_dataset(
        batch_size=batch_size,
        tfrecord_dir=path_nonrel,
        is_training=False,
        normalize_img=True,
        samplewise_center=False,
        samplewise_std_normalization=False)

    path_rel = 'data/sort_of_clevr_128_relational_test.tfrecords'
    rel_ds = get_train_dataset(
        batch_size=batch_size,
        tfrecord_dir=path_rel,
        is_training=False,
        normalize_img=True,
        samplewise_center=False,
        samplewise_std_normalization=False)

    # Load models.
    b_model = make_soc_baseline_model()
    b_model.load_weights('benchmark/sort_of_clevr_baseline_weights.h5')
    b_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    o_model = make_original_relnet(task='sort-of-clevr')
    o_model.load_weights('benchmark/sort_of_clevr_original_relation_network_weights.h5')
    o_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    _ , r_model = make_ResNet50_relnet(task='sort-of-clevr', trainable=True)
    r_model.load_weights('benchmark/sort_of_clevr_ResNet50_relation_network_weights.h5')
    r_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Get accuracy and save to csv.
    _, b_acc_nonrel = b_model.evaluate(x=nonrel_ds)
    _, b_acc_rel = b_model.evaluate(x=rel_ds)

    _, o_acc_nonrel = o_model.evaluate(x=nonrel_ds)
    _, o_acc_rel = o_model.evaluate(x=rel_ds)

    _, r_acc_nonrel = r_model.evaluate(x=nonrel_ds)
    _, r_acc_rel = r_model.evaluate(x=rel_ds)

    data = {'Question': ['Non-Rel', 'Rel', 'Non-Rel', 'Rel', 'Non-Rel', 'Rel'],
            'Model': ['CNN+MLP', 'CNN+MLP', 'CNN+RN', 'CNN+RN', 'ResNet50+RN', 'ResNet50+RN'],
            'Accuracy': [b_acc_nonrel, b_acc_rel, o_acc_nonrel, o_acc_rel, r_acc_nonrel, r_acc_rel]
            }

    df = pd.DataFrame(data, columns = ['Question', 'Model', 'Accuracy'])

    # Save the dataframe.
    df.to_csv(filename)

if __name__ == '__main__':
    # Parameters.
    BATCH_SIZE = 64
    EPOCHS = 20
    EPOCHS_TOP = 1
    STEPS_PER_EPOCH = 9800*20 // BATCH_SIZE
    LR_OLD = 2.5e-4
    LR_NEW = 0.0001
    TEST_BATCH_SIZE = 36  # test dataset is 200*6*3=3600 (img, question, answers) pairs

    # Define datasets.
    train_ds = get_train_dataset(
        batch_size=BATCH_SIZE,
        tfrecord_dir='data/sort_of_clevr_128_train.tfrecords',
        is_training=True,
        normalize_img=True,
        samplewise_center=True,
        samplewise_std_normalization=True)
    
    test_non_relational_ds = get_train_dataset(
        batch_size=BATCH_SIZE,
        tfrecord_dir='data/sort_of_clevr_128_non_relational_test.tfrecords',
        is_training=False,
        normalize_img=True,
        samplewise_center=False,
        samplewise_std_normalization=False)

    test_relational_ds = get_train_dataset(
        batch_size=BATCH_SIZE,
        tfrecord_dir='data/sort_of_clevr_128_relational_test.tfrecords',
        is_training=False,
        normalize_img=True,
        samplewise_center=False,
        samplewise_std_normalization=False)

    # Train models.
    train_and_save_original_relnet(
        train_data=train_ds,
        steps=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        lr=LR_OLD)

    train_and_save_soc_baseline(
        train_data=train_ds,
        steps=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        lr=LR_OLD)
    
    train_and_save_ResNet50_relnet(
        train_data=train_ds,
        epochs_top=EPOCHS_TOP,
        steps=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        full_model_lr=LR_NEW)
    
    # Test models.
    test_all_models(
        filename='benchmark/sort_of_clevr_benchmark.csv',
        batch_size=TEST_BATCH_SIZE)
