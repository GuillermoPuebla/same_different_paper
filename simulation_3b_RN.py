# Imports
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate, Lambda, Average, Dropout


# Relation Network functions.
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

def make_ResNet50_relnet_with_ds_label(
    dataset='SVRT',
    resnet_layer='last_size_8',
    trainable=True,
    secondary_outputs=True):

    # Inputs.
    image = Input((128, 128, 3))
    
    if dataset=='sort-of-clevr':
        question = Input((11,))
    elif dataset=='SVRT':
        question = Input((2,)) # same-different ([1, 0]) or task classification ([0, 1]).
    else:
        raise ValueError('dataset not supported!')
    
    # Get CNN features.
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=image)
        
    if resnet_layer=='last_size_4':
        layer_name = 'conv5_block3_out' # shape: (None, 4, 4, 2048)
    elif resnet_layer=='last_size_8':
        layer_name = 'conv4_block6_out' # shape: (None, 8, 8, 1024)
    else:
        raise ValueError('layer not supported!')
    cnn_features = base_model.get_layer(layer_name).output
    
    # Freeze the base_model if needed
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
    for feature1 in features:
        for feature2 in features:
            relations.append(concat([feature1, feature2, question]))
    
    # g function.
    g_MLP = get_MLP(4, get_dense(4, units=512))
    mid_relations = []
    for r in relations:
        mid_relations.append(g_MLP(r))
    combined_relation = Average()(mid_relations)

    # f function.
    rn = Dense(512, activation='relu')(combined_relation)
    rn = dropout_dense(rn, units=512)
    
    # SD answer.
    if dataset == 'sort-of-clevr':
        output_units = 10
        answer = Dense(output_units, activation='softmax')(rn)
    elif dataset == 'SVRT':
        output_units = 1
        answer = Dense(output_units, activation='sigmoid', name='sd')(rn)
        
    if secondary_outputs:
        ds_class = Dense(10, activation='sigmoid', name='ds_class')(rn)
        model = Model(inputs=[image, question], outputs=[answer, ds_class])
    else:
        model = Model(inputs=[image, question], outputs=answer)
    
    return base_model, model

# Dataset
def get_ds_with_ds_label(
    batch_size,
    tfrecord_dir,
    autotune_settings,
    ds_name,
    is_training=True,
    process_img=True,
    sd_sample_weight=True,
    tc_sample_weight=True,
    relnet=False
    ):
    # Load dataset
    raw_image_dataset = tf.data.TFRecordDataset(
        tfrecord_dir,
        num_parallel_reads=autotune_settings
        )
    # Define example reading function
    def read_tfrecord(serialized_example):
        # Create a dictionary describing the features
        feature_description = {
            'label': tf.io.FixedLenFeature([], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'coordinates': tf.io.FixedLenFeature([], tf.string),
            'relative_position': tf.io.FixedLenFeature([], tf.int64)
            }
        # Parse example
        example = tf.io.parse_single_example(
            serialized_example,
            feature_description
            )
        # Cast label to int64
        label = example['label']
        label = tf.cast(label, tf.int64)
        # Get image
        image = tf.image.decode_png(example['image_raw'])
        # Ensure shape dimensions are constant
        image = tf.reshape(image, [128, 128, 3])
        # Process image
        if process_img:
            image = tf.cast(image, tf.float64)
            image /= 255.0
            # Sample-wise center image
            mean = tf.reduce_mean(image)
            image -= mean
            # Sample-wise std normalization
            std = tf.math.reduce_std(image)
            image /= std
        # Cast relative position to int64
        rel_pos = example['relative_position']
        rel_pos = tf.cast(rel_pos, tf.int64)
        # Sample weights.
        sd_w = tf.constant(1, dtype=tf.int64) if sd_sample_weight else tf.constant(0, dtype=tf.int64)
        rp_w = tf.constant(1, dtype=tf.int64) if tc_sample_weight else tf.constant(0, dtype=tf.int64)
        # Get dataset label
        ds_name_to_label = {
            'original': 0,
            'irregular': 1,
            'regular': 2,
            'open': 3, 
            'wider': 4,
            'scrambled': 5,
            'random': 6,
            'filled': 7,
            'lines': 8,
            'arrows': 9
            }
        ds_label = tf.constant(
            np.eye(10, dtype=int)[ds_name_to_label[ds_name]].tolist(), 
            dtype=tf.int64
            )
        if relnet:
            question = tf.constant([1, 0], dtype=tf.int64) if sd_sample_weight else tf.constant([0, 1], dtype=tf.int64)
            rp_w = tf.constant(0, dtype=tf.int64) if sd_sample_weight else tf.constant(1, dtype=tf.int64)
            return (image, question), (label, ds_label), (sd_w, rp_w)
        else:
            return image, (label, ds_label), (sd_w, rp_w)
    # Parse dataset
    dataset = raw_image_dataset.map(
        read_tfrecord, 
        num_parallel_calls=autotune_settings
        )
    # Always shuffle for simplicity
    dataset = dataset.shuffle(7000, reshuffle_each_iteration=True)
    # Infinite dataset to avoid the potential last partial batch in each epoch
    if is_training:
        dataset = dataset.repeat()
    if batch_size is not None:
        dataset = dataset.batch(batch_size).prefetch(autotune_settings)
    
    return dataset

def get_task_classification_dataset(
    autotune_settings, 
    batch_size, 
    ds_dir, 
    ds_type='train', 
    process_img=True,
    relnet=False
    ):
    """ds_type: 'train' or 'val' """
    is_training = True if ds_type == 'train' else False
    tc_datasets = []
       
    # All others (task-classification)
    # Original
    ds_original_tc = get_ds_with_ds_label(
        batch_size=None,
        tfrecord_dir=f'{ds_dir}/original_{ds_type}.tfrecords',
        autotune_settings=autotune_settings,
        ds_name='original',
        is_training=is_training,
        process_img=process_img,
        sd_sample_weight=False,
        tc_sample_weight=True,
        relnet=relnet
        )
    tc_datasets.append(ds_original_tc)
    # Irregular
    ds_irregular_tc = get_ds_with_ds_label(
        batch_size=None,
        tfrecord_dir=f'{ds_dir}/irregular_{ds_type}.tfrecords',
        autotune_settings=autotune_settings,
        ds_name='irregular',
        is_training=is_training,
        process_img=process_img,
        sd_sample_weight=False,
        tc_sample_weight=True,
        relnet=relnet
        )
    tc_datasets.append(ds_irregular_tc)
    # Regular
    ds_regular_tc = get_ds_with_ds_label(
        batch_size=None,
        tfrecord_dir=f'{ds_dir}/regular_{ds_type}.tfrecords',
        autotune_settings=autotune_settings,
        ds_name='regular',
        is_training=is_training,
        process_img=process_img,
        sd_sample_weight=False,
        tc_sample_weight=True,
        relnet=relnet
        )
    tc_datasets.append(ds_regular_tc)
    # Open
    ds_open_tc = get_ds_with_ds_label(
        batch_size=None,
        tfrecord_dir=f'{ds_dir}/open_{ds_type}.tfrecords',
        autotune_settings=autotune_settings,
        ds_name='open',
        is_training=is_training,
        process_img=process_img,
        sd_sample_weight=False,
        tc_sample_weight=True,
        relnet=relnet
        )
    tc_datasets.append(ds_open_tc)
    # Wider
    ds_wider_tc = get_ds_with_ds_label(
        batch_size=None,
        tfrecord_dir=f'{ds_dir}/wider_line_{ds_type}.tfrecords',
        autotune_settings=autotune_settings,
        ds_name='wider',
        is_training=is_training,
        process_img=process_img,
        sd_sample_weight=False,
        tc_sample_weight=True,
        relnet=relnet
        )
    tc_datasets.append(ds_wider_tc)
    # Scrambled
    ds_scrambled_tc = get_ds_with_ds_label(
        batch_size=None,
        tfrecord_dir=f'{ds_dir}/scrambled_{ds_type}.tfrecords',
        autotune_settings=autotune_settings,
        ds_name='scrambled',
        is_training=is_training,
        process_img=process_img,
        sd_sample_weight=False,
        tc_sample_weight=True,
        relnet=relnet
        )
    tc_datasets.append(ds_scrambled_tc)
    # Random color
    ds_random_tc = get_ds_with_ds_label(
        batch_size=None,
        tfrecord_dir=f'{ds_dir}/random_color_{ds_type}.tfrecords',
        autotune_settings=autotune_settings,
        ds_name='random',
        is_training=is_training,
        process_img=process_img,
        sd_sample_weight=False,
        tc_sample_weight=True,
        relnet=relnet
        )
    tc_datasets.append(ds_random_tc)
    # Filled
    ds_filled_tc = get_ds_with_ds_label(
        batch_size=None,
        tfrecord_dir=f'{ds_dir}/filled_{ds_type}.tfrecords',
        autotune_settings=autotune_settings,
        ds_name='filled',
        is_training=is_training,
        process_img=process_img,
        sd_sample_weight=False,
        tc_sample_weight=True,
        relnet=relnet
        )
    tc_datasets.append(ds_filled_tc)
    # Lines
    ds_lines_tc = get_ds_with_ds_label(
        batch_size=None,
        tfrecord_dir=f'{ds_dir}/lines_{ds_type}.tfrecords',
        autotune_settings=autotune_settings,
        ds_name='lines',
        is_training=is_training,
        process_img=process_img,
        sd_sample_weight=False,
        tc_sample_weight=True,
        relnet=relnet
        )
    tc_datasets.append(ds_lines_tc)
    # Arrows
    ds_arrows_tc = get_ds_with_ds_label(
        batch_size=None,
        tfrecord_dir=f'{ds_dir}/arrows_{ds_type}.tfrecords',
        autotune_settings=autotune_settings,
        ds_name='arrows',
        is_training=is_training,
        process_img=process_img,
        sd_sample_weight=False,
        tc_sample_weight=True,
        relnet=relnet
        )
    tc_datasets.append(ds_arrows_tc)
    # no 'weights' argument gives you uniform distribution
    task_classification_dataset = tf.data.Dataset.sample_from_datasets(
        tc_datasets,
        weights=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        )
    task_classification_dataset = task_classification_dataset.batch(batch_size).prefetch(autotune_settings)
    return task_classification_dataset

def get_master_ds_with_ds_labels(
    autotune_settings, 
    batch_size, 
    ds_dir, 
    process_img=True,
    relnet=False
    ):
    """Builds dataset that samples each batch from one of the training datasets
    assiging a same-different sample weight of 1 only for the original
    condition and a task classification sample weight 1 for all condtions."""
    
    # Make datasets and append if it is not the dont_include dataset.
    all_datasets = []
    
    # Original (same-different)
    ds_original_sd = get_ds_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/original_train.tfrecords',
        autotune_settings=autotune_settings,
        ds_name='original',
        is_training=True,
        process_img=process_img,
        sd_sample_weight=True,
        tc_sample_weight=False,
        relnet=relnet
        )    
    # All others (task-classification)
    ds_task_classification = get_task_classification_dataset(
        autotune_settings=AUTO,
        batch_size=batch_size, 
        ds_dir=ds_dir, 
        ds_type='train', 
        process_img=process_img,
        relnet=relnet
        )
    all_datasets = [ds_original_sd, ds_task_classification]
    # Sample task (same-diff, task classification)
    choice_tensor = tf.constant(value=[0, 1], dtype=tf.int64)
    choice_dataset = tf.data.Dataset.from_tensor_slices(choice_tensor).repeat()
    
    return tf.data.Dataset.choose_from_datasets(all_datasets, choice_dataset)

# Training
def pre_train_relnet_with_ds_label(
    strategy,
    train_ds,
    val_ds,
    save_name,
    model_name,
    epochs,
    steps_per_epoch,
    validation_steps,
    n=10,
    lr=0.0003,
    ):
    with strategy.scope():
        # Define model
        base_model, model = make_ResNet50_relnet_with_ds_label(
            dataset='SVRT',
            resnet_layer='last_size_8',
            trainable=True,
            secondary_outputs=True
            )
        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss={'sd': 'binary_crossentropy', 'ds_class': 'categorical_crossentropy'},
            metrics={'sd': 'binary_accuracy', 'ds_class': 'categorical_accuracy'})
        
        for i in range(n):
            # Initialize weights
            for layer in model.layers:
                if hasattr(layer, 'kernel_initializer') and hasattr(layer, 'bias_initializer'):
                    layer.set_weights(
                        [
                         layer.kernel_initializer(shape=np.asarray(layer.kernel.shape)),
                         layer.bias_initializer(shape=np.asarray(layer.bias.shape))
                         ])
            # Load imagenet weights
            base_model.load_weights('Resnet50_no_top.h5')
            # Pre train on task classification only
            filename = save_name + '/' + model_name +'_run_' + str(i) + '_log.csv'
            history_logger = tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)
            # Train on same-different and task classification
            model.fit(
                train_ds,
                epochs=epochs,
                verbose=0,
                steps_per_epoch=steps_per_epoch,
                validation_data=val_ds,
                validation_steps=validation_steps,
                callbacks=[history_logger]
                )
            # Save weights
            weights_name = save_name + '/' + model_name + '_instance_' + str(i) + '.hdf5'
            model.save_weights(weights_name)
    return

def fine_tune_relnet_with_ds_label(
    strategy,
    train_ds,
    val_ds,
    save_name,
    model_name,
    epochs,
    steps_per_epoch,
    validation_steps,
    n=10,
    lr=0.0001,
    ):
    with strategy.scope():
        # Define model
        base_model, model = make_ResNet50_relnet_with_ds_label(
            dataset='SVRT',
            resnet_layer='last_size_8',
            trainable=True,
            secondary_outputs=True
            )
        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss={'sd': 'binary_crossentropy', 'ds_class': 'categorical_crossentropy'},
            metrics={'sd': 'binary_accuracy', 'ds_class': 'categorical_accuracy'})
        
        for i in range(n):
            # Load weights
            weights_name = save_name + '/' + model_name + '_instance_' + str(i) + '.hdf5'
            model.load_weights(weights_name)
            # Pre train on task classification only
            filename = save_name + '/' + model_name +'_run_' + str(i) + '_log.csv'
            history_logger = tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)
            # Train on same-different and task classification
            model.fit(
                train_ds,
                epochs=epochs,
                verbose=0,
                steps_per_epoch=steps_per_epoch,
                validation_data=val_ds,
                validation_steps=validation_steps,
                callbacks=[history_logger]
                )
            # Save weights
            weights_name = save_name + '/' + model_name + '_instance_' + str(i) + '.hdf5'
            model.save_weights(weights_name)
    return

# Test
def get_dataset_relnet_test_with_ds_label(
    batch_size,
    tfrecord_dir,
    ds_name,
    autotune_settings,
    task='SD',
    is_training=False,
    process_img=True
    ):
    # Load dataset
    raw_image_dataset = tf.data.TFRecordDataset(
        tfrecord_dir,
        num_parallel_reads=autotune_settings
        )
    # Define example reading function
    def read_tfrecord(serialized_example):
        # Create a dictionary describing the features.
        feature_description = {
            'label': tf.io.FixedLenFeature([], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'coordinates': tf.io.FixedLenFeature([], tf.string),
            'relative_position': tf.io.FixedLenFeature([], tf.int64)
            }
        # Parse example
        example = tf.io.parse_single_example(serialized_example, feature_description)
        # Cast label to int64
        label = example['label']
        label = tf.cast(label, tf.int64)
        # Get image
        image = tf.image.decode_png(example['image_raw'])
        # Ensure shape dimensions are constant
        image = tf.reshape(image, [128, 128, 3])
        # Process image
        if process_img:
            image = tf.cast(image, tf.float64)
            image /= 255.0
            # Sample-wise center image.
            mean = tf.reduce_mean(image)
            image -= mean
            # Sample-wise std normalization.
            std = tf.math.reduce_std(image)
            image /= std
        # Cast relative position to int64
        rel_pos = example['relative_position']
        rel_pos = tf.cast(rel_pos, tf.int64)
        # Get dataset label
        ds_name_to_label = {
            'original': 0,
            'irregular': 1,
            'regular': 2,
            'open': 3, 
            'wider': 4,
            'scrambled': 5,
            'random': 6,
            'filled': 7,
            'lines': 8,
            'arrows': 9
            }
        ds_label = tf.constant(
            np.eye(10, dtype=int)[ds_name_to_label[ds_name]].tolist(), 
            dtype=tf.int64
            )
        # Task-dependent data
        question = tf.constant([1, 0], dtype=tf.int64) if task=='SD' else tf.constant([0, 1], dtype=tf.int64)
        sd_w = tf.constant(1, dtype=tf.int64) if task=='SD' else tf.constant(0, dtype=tf.int64)
        rp_w = tf.constant(0, dtype=tf.int64) if task=='SD' else tf.constant(1, dtype=tf.int64)
        
        return (image, question), (label, ds_label), (sd_w, rp_w)
        
    # Parse dataset
    dataset = raw_image_dataset.map(
        read_tfrecord, 
        num_parallel_calls=autotune_settings
        )
    
    # Always shuffle for simplicity.
    dataset = dataset.shuffle(7000)
    # Infinite dataset to avoid the potential last partial batch in each epoch
    if is_training:
        dataset = dataset.repeat()
    if batch_size is not None:
        dataset = dataset.batch(batch_size).prefetch(autotune_settings)
    return dataset

def test_relnet_auc_with_ds_label(ds, weights_dir, model, model_name, condition, task='SD'):
    """Tests 10 versions of a single model in a single condition using the area under the ROC curve. 
    Args:
        ds: dataset with cases from the 'same' and 'different' classes.
        weights_dir: directory of models weights to test.
        model_name: model name. String.
        condition: condition name. String.
    Returns:
        A list with test data: model_name, condition, area under the ROC curve.
    """
    # Get list of weights from path
    weights_list = [f for f in listdir(weights_dir) if f.endswith('.hdf5')]
    weights_paths = [join(weights_dir, f) for f in weights_list if isfile(join(weights_dir, f))]
    
    # Test each model
    models_data = []
    for w_path in weights_paths:
        model.load_weights(w_path)
        metrics = model.evaluate(ds)
        # model.metrics_names = ['loss', 'sd_loss', 'rel_pos_loss', 'sd_auc_1', 'rel_pos_auc_1']
        if task=='SD':
            models_data.append([model_name, condition, 'Same-Different', metrics[3]])
        elif task=='RP':
            models_data.append([model_name, condition, 'Condition classification', metrics[4]])
        else:
            raise ValueError('Unrecognized task!')
    
    return models_data

def test_relnets_all_ds_auc(
    strategy,
    autotune_settings,
    batch_size,
    weights_dir,
    ds_dir
    ):
    # Load same/different datasets.
    datasets = []
    
    # Original SD
    ds = get_dataset_relnet_test_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/original_test.tfrecords',
        ds_name='original',
        autotune_settings=autotune_settings,
        is_training=False,
        task='SD',
        process_img=True)
    datasets.append(ds)
    
    # Original RP
    ds = get_dataset_relnet_test_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/original_test.tfrecords',
        ds_name='original',
        autotune_settings=autotune_settings,
        is_training=False,
        task='RP',
        process_img=True)
    datasets.append(ds)
    
    # Regular SD
    ds = get_dataset_relnet_test_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/regular_test.tfrecords',
        ds_name='regular',
        autotune_settings=autotune_settings,
        is_training=False,
        task='SD',
        process_img=True)
    datasets.append(ds)
    
    # Regular RP
    ds = get_dataset_relnet_test_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/regular_test.tfrecords',
        ds_name='regular',
        autotune_settings=autotune_settings,
        task='RP',
        is_training=False,
        process_img=True)
    datasets.append(ds)
    
    # Irregular SD
    ds = get_dataset_relnet_test_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/irregular_test.tfrecords',
        ds_name='irregular',
        autotune_settings=autotune_settings,
        task='SD',
        is_training=False,
        process_img=True)
    datasets.append(ds)
    
    # Irregular RP
    ds = get_dataset_relnet_test_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/irregular_test.tfrecords',
        ds_name='irregular',
        autotune_settings=autotune_settings,
        task='RP',
        is_training=False,
        process_img=True)
    datasets.append(ds)
    
    # Open SD
    ds = get_dataset_relnet_test_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/open_test.tfrecords',
        ds_name='open',
        autotune_settings=autotune_settings,
        task='SD',
        is_training=False,
        process_img=True)
    datasets.append(ds)
    
    # Open RP
    ds = get_dataset_relnet_test_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/open_test.tfrecords',
        ds_name='open',
        autotune_settings=autotune_settings,
        task='RP',
        is_training=False,
        process_img=True)
    datasets.append(ds)

    # Wider line SD
    ds = get_dataset_relnet_test_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/wider_line_test.tfrecords',
        ds_name='wider',
        autotune_settings=autotune_settings,
        task='SD',
        is_training=False,
        process_img=True)
    datasets.append(ds)
    
    # Wider line RP
    ds = get_dataset_relnet_test_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/wider_line_test.tfrecords',
        ds_name='wider',
        autotune_settings=autotune_settings,
        task='RP',
        is_training=False,
        process_img=True)
    datasets.append(ds)
    
    # Scrambled SD
    ds = get_dataset_relnet_test_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/scrambled_test.tfrecords',
        ds_name='scrambled',
        autotune_settings=autotune_settings,
        task='SD',
        is_training=False,
        process_img=True)
    datasets.append(ds)
    
    # Scrambled RP
    ds = get_dataset_relnet_test_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/scrambled_test.tfrecords',
        ds_name='scrambled',
        autotune_settings=autotune_settings,
        task='RP',
        is_training=False,
        process_img=True)
    datasets.append(ds)
    
    # Random color SD
    ds = get_dataset_relnet_test_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/random_color_test.tfrecords',
        ds_name='random',
        autotune_settings=autotune_settings,
        task='SD',
        is_training=False,
        process_img=True)
    datasets.append(ds)
    
    # Random color RP
    ds = get_dataset_relnet_test_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/random_color_test.tfrecords',
        ds_name='random',
        autotune_settings=autotune_settings,
        task='RP',
        is_training=False,
        process_img=True)
    datasets.append(ds)
    
    # Filled SD
    ds = get_dataset_relnet_test_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/filled_test.tfrecords',
        ds_name='filled',
        autotune_settings=autotune_settings,
        task='SD',
        is_training=False,
        process_img=True)
    datasets.append(ds)
    
    # Filled RP
    ds = get_dataset_relnet_test_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/filled_test.tfrecords',
        ds_name='filled',
        autotune_settings=autotune_settings,
        task='RP',
        is_training=False,
        process_img=True)
    datasets.append(ds)
    
    # Lines SD
    ds = get_dataset_relnet_test_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/lines_test.tfrecords',
        ds_name='lines',
        autotune_settings=autotune_settings,
        task='SD',
        is_training=False,
        process_img=True)
    datasets.append(ds)
    
    # Lines RP
    ds = get_dataset_relnet_test_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/lines_test.tfrecords',
        ds_name='lines',
        autotune_settings=autotune_settings,
        task='RP',
        is_training=False,
        process_img=True)
    datasets.append(ds)
    
    # Arrows SD
    ds = get_dataset_relnet_test_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/arrows_test.tfrecords',
        ds_name='arrows',
        autotune_settings=autotune_settings,
        task='SD',
        is_training=False,
        process_img=True)
    datasets.append(ds)
    
    # Arrows RP
    ds = get_dataset_relnet_test_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/arrows_test.tfrecords',
        ds_name='arrows',
        autotune_settings=autotune_settings,
        task='RP',
        is_training=False,
        process_img=True)
    datasets.append(ds)
    
    conditions = ['Original', 'Original', 'Regular', 'Regular', 'Irregular', 'Irregular', 'Open', 'Open', 'Wider Line', 'Wider Line',
                  'Scrambled', 'Scrambled', 'Random Color', 'Random Color', 'Filled', 'Filled', 'Lines', 'Lines', 'Arrows', 'Arrows']
    
    tasks = ['SD', 'RP'] * 10

    # Define model architecture
    with strategy.scope():
        # Define model
        base_model, model = make_ResNet50_relnet_with_ds_label(
            dataset='SVRT',
            resnet_layer='last_size_8',
            trainable=True,
            secondary_outputs=True
            )
        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.003),
            loss={'sd': 'binary_crossentropy', 'ds_class': 'categorical_crossentropy'},
            metrics=[tf.keras.metrics.AUC()]
            )
        # Test
        results = []
        for ds, condition, task in zip(datasets, conditions, tasks):
            # Fine-tune, imagenet, GAP.
            ft_im_gap = test_relnet_auc_with_ds_label(
                ds=ds,
                weights_dir=weights_dir,
                model=model,
                model_name='RN',
                condition=condition,
                task=task
                )
            results.extend(ft_im_gap)
        
    return pd.DataFrame(results, columns=['Model', 'Condition', 'Task', 'AUC'])

def validate_relnets_all_ds_auc(
    strategy,
    autotune_settings,
    batch_size,
    weights_dir,
    ds_dir
    ):
    # Load same/different datasets.
    datasets = []
    
    # Original SD
    ds = get_dataset_relnet_test_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/original_test.tfrecords',
        ds_name='original',
        autotune_settings=autotune_settings,
        is_training=False,
        task='SD',
        process_img=True)
    datasets.append(ds)
    
    # Original RP
    ds = get_dataset_relnet_test_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/original_test.tfrecords',
        ds_name='original',
        autotune_settings=autotune_settings,
        is_training=False,
        task='RP',
        process_img=True)
    datasets.append(ds)
    
    # Regular RP
    ds = get_dataset_relnet_test_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/regular_test.tfrecords',
        ds_name='regular',
        autotune_settings=autotune_settings,
        task='RP',
        is_training=False,
        process_img=True)
    datasets.append(ds)
    
    # Irregular RP
    ds = get_dataset_relnet_test_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/irregular_test.tfrecords',
        ds_name='irregular',
        autotune_settings=autotune_settings,
        task='RP',
        is_training=False,
        process_img=True)
    datasets.append(ds)
    
    # Open RP
    ds = get_dataset_relnet_test_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/open_test.tfrecords',
        ds_name='open',
        autotune_settings=autotune_settings,
        task='RP',
        is_training=False,
        process_img=True)
    datasets.append(ds)
    
    # Wider line RP
    ds = get_dataset_relnet_test_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/wider_line_test.tfrecords',
        ds_name='wider',
        autotune_settings=autotune_settings,
        task='RP',
        is_training=False,
        process_img=True)
    datasets.append(ds)
    
    # Scrambled RP
    ds = get_dataset_relnet_test_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/scrambled_test.tfrecords',
        ds_name='scrambled',
        autotune_settings=autotune_settings,
        task='RP',
        is_training=False,
        process_img=True)
    datasets.append(ds)
    
    # Random color RP
    ds = get_dataset_relnet_test_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/random_color_test.tfrecords',
        ds_name='random',
        autotune_settings=autotune_settings,
        task='RP',
        is_training=False,
        process_img=True)
    datasets.append(ds)
    
    # Filled RP
    ds = get_dataset_relnet_test_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/filled_test.tfrecords',
        ds_name='filled',
        autotune_settings=autotune_settings,
        task='RP',
        is_training=False,
        process_img=True)
    datasets.append(ds)
    
    # Lines RP
    ds = get_dataset_relnet_test_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/lines_test.tfrecords',
        ds_name='lines',
        autotune_settings=autotune_settings,
        task='RP',
        is_training=False,
        process_img=True)
    datasets.append(ds)
    
    # Arrows RP
    ds = get_dataset_relnet_test_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/arrows_test.tfrecords',
        ds_name='arrows',
        autotune_settings=autotune_settings,
        task='RP',
        is_training=False,
        process_img=True)
    datasets.append(ds)
    
    conditions = [
                  'Original', 
                  'Original', 
                  'Regular', 
                  'Irregular', 
                  'Open',
                  'Wider Line', 
                  'Scrambled',
                  'Random Color', 
                  'Filled', 
                  'Lines', 
                  'Arrows'
                  ]
    
    tasks = ['SD', 
             'RP',
             'RP',
             'RP',
             'RP',
             'RP',
             'RP',
             'RP',
             'RP',
             'RP',
             'RP'
             ]

    # Define model architecture
    with strategy.scope():
        # Define model
        base_model, model = make_ResNet50_relnet_with_ds_label(
            dataset='SVRT',
            resnet_layer='last_size_8',
            trainable=True,
            secondary_outputs=True
            )
        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.003),
            loss={'sd': 'binary_crossentropy', 'ds_class': 'categorical_crossentropy'},
            metrics=[tf.keras.metrics.AUC()]
            )
        # Test
        results = []
        for ds, condition, task in zip(datasets, conditions, tasks):
            # Fine-tune, imagenet, GAP.
            ft_im_gap = test_relnet_auc_with_ds_label(
                ds=ds,
                weights_dir=weights_dir,
                model=model,
                model_name='RN',
                condition=condition,
                task=task
                )
            results.extend(ft_im_gap)
        
    return pd.DataFrame(results, columns=['Model', 'Condition', 'Task', 'AUC'])


if __name__ == '__main__':
    # Training hyperparameters.
    EPOCHS_TOP = 5
    EPOCHS = 15
    BATCH_SIZE = 64
    STEPS_PER_EPOCH = 28000 // BATCH_SIZE
    VALIDATION_STEPS = 2800 // BATCH_SIZE

    # Pretrain
    AUTO = tf.data.experimental.AUTOTUNE
    strategy = tf.distribute.get_strategy()

    train_ds = get_master_ds_with_ds_labels(
        autotune_settings=AUTO,
        batch_size=BATCH_SIZE,
        ds_dir='data',
        process_img=True,
        relnet=True
        )
    pre_train_relnet_with_ds_label(
        strategy=strategy,
        train_ds=train_ds,
        val_ds=None,
        save_name='simulation_3/task_classification/RN_instances',
        model_name='RN',
        epochs=EPOCHS_TOP,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=None,
        n=10,
        lr=0.0003
        )
    # Train
    train_ds = get_master_ds_with_ds_labels(
    autotune_settings=AUTO,
    batch_size=BATCH_SIZE,
    ds_dir='data',
    process_img=True,
    relnet=True
    )
    fine_tune_relnet_with_ds_label(
        strategy=strategy,
        train_ds=train_ds,
        val_ds=None,
        save_name='simulation_3/task_classification/RN_instances',
        model_name='RN',
        epochs=10, # 10 extra epochs
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=None,
        n=10,
        lr=0.0001
        )
    # Test
    df = test_relnets_all_ds_auc(
        strategy=strategy,
        autotune_settings=AUTO,
        batch_size=560, 
        weights_dir='simulation_3/task_classification/RN_instances',
        ds_dir='data'
        )
    df.to_csv('simulation_3/task_classification/sim_3_RN_taskclass_test_auc.csv')

    # Validate RN
    df = validate_relnets_all_ds_auc(
        strategy=strategy,
        autotune_settings=AUTO,
        batch_size=560, 
        weights_dir='simulation_3/task_classification/RN_instances',
        ds_dir='data'
        )
    df.to_csv('simulation_3/task_classification/sim_3_RN_taskclass_val_auc.csv')
