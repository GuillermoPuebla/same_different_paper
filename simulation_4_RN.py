from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.applications.resnet50 import ResNet50
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

def make_ResNet50_relnet(
    dataset='SVRT',
    resnet_layer='last_size_8',
    trainable=False,
    secondary_outputs=True):

    # Inputs.
    image = Input((128, 128, 3))
    
    if dataset=='sort-of-clevr':
        question = Input((11,))
    elif dataset=='SVRT':
        question = Input((2,)) # same-different ([1, 0]) or relative position ([0, 1]).
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
        rel_pos = Dense(1, activation='sigmoid', name='rel_pos')(rn)
        model = Model(inputs=[image, question], outputs=[answer, rel_pos])
    else:
        model = Model(inputs=[image, question], outputs=answer)
    
    return base_model, model

def get_dataset(
    batch_size,
    tfrecord_dir,
    is_training=True,
    process_img=True,
    sd_sample_weight=True,
    relnet=False):

    # Load dataset.
    if type(tfrecord_dir) == list:
        raw_image_dataset = tf.data.TFRecordDataset(tfrecord_dir, num_parallel_reads=len(tfrecord_dir))
    else:
        raw_image_dataset = tf.data.TFRecordDataset(tfrecord_dir)
    
    # Define example reading function.
    def read_tfrecord(serialized_example):
        # Create a dictionary describing the features.
        feature_description = {
            'label': tf.io.FixedLenFeature([], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'coordinates': tf.io.FixedLenFeature([], tf.string),
            'relative_position': tf.io.FixedLenFeature([], tf.int64)}
        
        # Parse example.
        example = tf.io.parse_single_example(serialized_example, feature_description)

        # Cast label to int64
        label = example['label']
        label = tf.cast(label, tf.int64)

        # Get image.
        image = tf.image.decode_png(example['image_raw'])
        
        # Ensure shape dimensions are constant.
        image = tf.reshape(image, [128, 128, 3])
        
        # Process image.
        if process_img:
            image = tf.cast(image, tf.float64)
            image /= 255.0
            # Sample-wise center image.
            mean = tf.reduce_mean(image)
            image -= mean
            # Sample-wise std normalization.
            std = tf.math.reduce_std(image)
            image /= std

        # Get coordinates.
        b_coors = example['coordinates']
        coors = tf.io.parse_tensor(b_coors, out_type=tf.float64) # restore 2D array from byte string
        coors = tf.reshape(coors, [4])

        # Cast relative position to int64
        rel_pos = example['relative_position']
        rel_pos = tf.cast(rel_pos, tf.int64)

        # Sample weights.
        sd_w = tf.constant(1, dtype=tf.int64) if sd_sample_weight else tf.constant(0, dtype=tf.int64)
        rp_w = tf.constant(1, dtype=tf.int64)
        
        if relnet:
            question = tf.constant([1, 0], dtype=tf.int64) if sd_sample_weight else tf.constant([0, 1], dtype=tf.int64)
            rp_w = tf.constant(0, dtype=tf.int64) if sd_sample_weight else tf.constant(1, dtype=tf.int64)
            
            return (image, question), (label, rel_pos), (sd_w, rp_w)
        else:
            return image, (label, rel_pos), (sd_w, rp_w)
            
    # Parse dataset.
    dataset = raw_image_dataset.map(read_tfrecord)
    
    # Always shuffle for simplicity.
    dataset = dataset.shuffle(5600)
    
    if is_training:
        # Infinite dataset to avoid the potential last partial batch in each epoch.
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)
    
    return dataset

def get_master_dataset_relnet(batch_size, dont_include, process_img=True):
    """Builds dataset that samples each batch from one of the training datasets
    assiging a same-different sample weight of 1 for all but the dont_include
    dataset and a relative-position sample weight 1 for all other datasets."""
    
    # Make datasets and append if it is not the dont_include dataset.
    datasets = []
    
    # Original: SD.
    ds_original_SD = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/original_train.tfrecords',
        is_training=True,
        process_img=process_img,
        sd_sample_weight=True,
        relnet=True)
    if dont_include != 'Original':
        datasets.append(ds_original_SD)
    
    # Original: RP.
    ds_original_RP = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/original_train.tfrecords',
        is_training=True,
        process_img=process_img,
        sd_sample_weight=False,
        relnet=True)
    datasets.append(ds_original_RP)

    # Regular: SD.
    ds_regular_SD = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/regular_train.tfrecords',
        is_training=True,
        process_img=process_img,
        sd_sample_weight=True,
        relnet=True)
    if dont_include != 'Regular':
        datasets.append(ds_regular_SD)

    # Regular: RP.
    ds_regular_RP = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/regular_train.tfrecords',
        is_training=True,
        process_img=process_img,
        sd_sample_weight=False,
        relnet=True)
    datasets.append(ds_regular_RP)
    
    # Irregular SD.
    ds_irregular_SD = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/irregular_train.tfrecords',
        is_training=True,
        process_img=process_img,
        sd_sample_weight=True,
        relnet=True)
    if dont_include != 'Irregular':
        datasets.append(ds_irregular_SD)
    
    # Irregular RP.
    ds_irregular_RP = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/irregular_train.tfrecords',
        is_training=True,
        process_img=process_img,
        sd_sample_weight=False,
        relnet=True)
    datasets.append(ds_irregular_RP)
    
    # Open SD.
    ds_open_SD = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/open_train.tfrecords',
        is_training=True,
        process_img=process_img,
        sd_sample_weight=True,
        relnet=True)
    if dont_include != 'Open':
        datasets.append(ds_open_SD)
    
    # Open RP.
    ds_open_RP = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/open_train.tfrecords',
        is_training=True,
        process_img=process_img,
        sd_sample_weight=False,
        relnet=True)
    datasets.append(ds_open_RP)
    
    # Wider line SD.
    ds_wider_SD = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/wider_line_train.tfrecords',
        is_training=True,
        process_img=process_img,
        sd_sample_weight=True,
        relnet=True)
    if dont_include != 'Wider line':
        datasets.append(ds_wider_SD)
    
    # Wider line RP.
    ds_wider_RP = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/wider_line_train.tfrecords',
        is_training=True,
        process_img=process_img,
        sd_sample_weight=False,
        relnet=True)
    datasets.append(ds_wider_RP)

    # Scrambled SD.
    ds_scrambled_SD = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/scrambled_train.tfrecords',
        is_training=True,
        process_img=process_img,
        sd_sample_weight=True,
        relnet=True)
    if dont_include != 'Scrambled':
        datasets.append(ds_scrambled_SD)
    
    # Scrambled RP.
    ds_scrambled_RP = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/scrambled_train.tfrecords',
        is_training=True,
        process_img=process_img,
        sd_sample_weight=False,
        relnet=True)
    datasets.append(ds_scrambled_RP)
    
    # Random color SD.
    ds_random_SD = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/random_color_train.tfrecords',
        is_training=True,
        process_img=process_img,
        sd_sample_weight=True,
        relnet=True)
    if dont_include != 'Random color':
        datasets.append(ds_random_SD)
    
    # Random color RP.
    ds_random_RP = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/random_color_train.tfrecords',
        is_training=True,
        process_img=process_img,
        sd_sample_weight=False,
        relnet=True)
    datasets.append(ds_random_RP)
    
    # Filled SD.
    ds_filled_SD = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/filled_train.tfrecords',
        is_training=True,
        process_img=process_img,
        sd_sample_weight=True,
        relnet=True)
    if dont_include != 'Filled':
        datasets.append(ds_filled_SD)
    
    # Filled RP.
    ds_filled_RP = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/filled_train.tfrecords',
        is_training=True,
        process_img=process_img,
        sd_sample_weight=False,
        relnet=True)
    datasets.append(ds_filled_RP)
    
    # Lines SD.
    ds_lines_SD = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/lines_train.tfrecords',
        is_training=True,
        process_img=process_img,
        sd_sample_weight=True,
        relnet=True)
    if dont_include != 'Lines':
        datasets.append(ds_lines_SD)
    
    # Lines RP.
    ds_lines_RP = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/lines_train.tfrecords',
        is_training=True,
        process_img=process_img,
        sd_sample_weight=False,
        relnet=True)
    datasets.append(ds_lines_RP)
    
    # Arrows SD.
    ds_arrows_SD = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/arrows_train.tfrecords',
        is_training=True,
        process_img=process_img,
        sd_sample_weight=True,
        relnet=True)
    if dont_include != 'Arrows':
        datasets.append(ds_arrows_SD)
    
    # Arrows RP.
    ds_arrows_RP = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/arrows_train.tfrecords',
        is_training=True,
        process_img=process_img,
        sd_sample_weight=False,
        relnet=True)
    datasets.append(ds_arrows_RP)
    
    # Define a dataset containing range(0, 9).
    # Note. I don't need to oversample the original dataset here because I'm training
    # same/different in all datasets except the one that is going to be tested.
    choice_dataset = tf.data.Dataset.range(len(datasets)).repeat()
    
    return tf.data.experimental.choose_from_datasets(datasets, choice_dataset)

def fine_tune_relnet(
    train_ds,
    val_ds,
    save_name,
    epochs_top,
    epochs,
    steps_per_epoch,
    validation_steps,
    n=10,
    lr=0.0001):
    
    # Repate trainign 10 times.
    for i in range(n):
        # Define best relation network sim1: 'last_size_8'.
        base_model, model = make_ResNet50_relnet(
            dataset='SVRT',
            resnet_layer='last_size_8',
            trainable=False,
            secondary_outputs=True)
        
        # Compile.
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.0003),
            loss={'sd': 'binary_crossentropy', 'rel_pos': 'binary_crossentropy'},
            metrics={'sd': 'binary_accuracy', 'rel_pos': 'binary_accuracy'})
        
        # Train.
        model.fit(
            train_ds,
            epochs=epochs_top,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,
            validation_steps=validation_steps)
        
        # Unfreeze Resnet50.
        base_model.trainable = True
        
        # Re-compile.
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss={'sd': 'binary_crossentropy', 'rel_pos': 'binary_crossentropy'},
            metrics={'sd': 'binary_accuracy', 'rel_pos': 'binary_accuracy'})
        
        # Train.
        model.fit(
            train_ds,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,
            validation_steps=validation_steps)
        
        # Save weights.
        weights_name = save_name + 'instance_' + str(i) + '.hdf5'
        model.save_weights(weights_name)
    
    return

def train_in_all_sd_relnet(
    epochs_top,
    epochs,
    steps_per_epoch,
    validation_steps,
    n=10,
    lr=0.0001,
    batch_size=64):
    
    # Define all master datasets.
    no_irregular_ds = get_master_dataset_relnet(batch_size=batch_size, dont_include='Irregular', process_img=True)
    no_regular_ds = get_master_dataset_relnet(batch_size=batch_size, dont_include='Regular', process_img=True)
    no_open_ds = get_master_dataset_relnet(batch_size=batch_size, dont_include='Open', process_img=True)
    no_wider_ds = get_master_dataset_relnet(batch_size=batch_size, dont_include='Wider line', process_img=True)
    no_scrambled_ds = get_master_dataset_relnet(batch_size=batch_size, dont_include='Scrambled', process_img=True)
    no_random_ds = get_master_dataset_relnet(batch_size=batch_size, dont_include='Random color', process_img=True)
    no_filled_ds = get_master_dataset_relnet(batch_size=batch_size, dont_include='Filled', process_img=True)
    no_lines_ds = get_master_dataset_relnet(batch_size=batch_size, dont_include='Lines', process_img=True)
    no_arrows_ds = get_master_dataset_relnet(batch_size=batch_size, dont_include='Arrows', process_img=True)
    
    # Validation dataset.
    val_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/original_val.tfrecords',
        is_training=False,
        process_img=True,
        relnet=True)
    
    # Train model in each dataset 10 times.
    ds_and_names = [
        (no_irregular_ds, 'simulation_4/relnet_no_irregular/'),
        (no_regular_ds, 'simulation_4/relnet_no_regular/'),
        (no_open_ds, 'simulation_4/relnet_no_open/'),
        (no_wider_ds, 'simulation_4/relnet_no_wider/'),
        (no_scrambled_ds, 'simulation_4/relnet_no_scrambled/'),
        (no_random_ds, 'simulation_4/relnet_no_random/'),
        (no_filled_ds, 'simulation_4/relnet_no_filled/'),
        (no_lines_ds, 'simulation_4/relnet_no_lines/'),
        (no_arrows_ds, 'simulation_4/relnet_no_arrows/')]
    
    for ds, name in ds_and_names:
        fine_tune_relnet(
            train_ds=ds,
            val_ds=val_ds,
            save_name=name,
            epochs_top=epochs_top,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            n=n,
            lr=lr,
            base_weights='imagenet')
    return

def test_relnet_auc(ds, weights_dir, model_name, condition, task='SD'):
    """Tests 10 versions of a single model in a single condition using the area under the ROC curve. 
    Args:
        ds: dataset with cases from the 'same' and 'different' classes.
        weights_dir: directory of models weights to test.
        model_name: model name. String.
        condition: condition name. String.
    Returns:
        A list with test data: model_name, condition, area under the ROC curve.
    """
    
    # Define best relation network sim1: 'last_size_8'.
    base_model, model = make_ResNet50_relnet(
        dataset='SVRT',
        resnet_layer='last_size_8',
        trainable=False,
        secondary_outputs=True)
    
    # Compile: set metric to auc (default is area under the ROC curve).
    model.compile(
            optimizer=tf.keras.optimizers.Adam(0.0003),
            loss='binary_crossentropy',
            metrics=[tf.keras.metrics.AUC()])
    
    # Get list of weights from path.
    weights_paths = [join(weights_dir, f) for f in listdir(weights_dir) if isfile(join(weights_dir, f))]
    
    # Test each model.
    models_data = []
    for w_path in weights_paths:
        model.load_weights(w_path)
        metrics = model.evaluate(ds)
        # model.metrics_names = ['loss', 'sd_loss', 'rel_pos_loss', 'sd_auc_1', 'rel_pos_auc_1']
        if task=='SD':
            models_data.append([model_name, condition, 'Same-Different', metrics[3]])
        elif task=='RP':
            models_data.append([model_name, condition, 'Relative position', metrics[4]])
        else:
            raise ValueError('Unrecognized task!')
    
    return models_data

# Redefine get_dataset to test relnet.
def get_dataset_relnet_test(
    batch_size,
    tfrecord_dir,
    task='SD',
    is_training=False,
    process_img=True):

    # Load dataset.
    if type(tfrecord_dir) == list:
        raw_image_dataset = tf.data.TFRecordDataset(tfrecord_dir, num_parallel_reads=len(tfrecord_dir))
    else:
        raw_image_dataset = tf.data.TFRecordDataset(tfrecord_dir)
    
    # Define example reading function.
    def read_tfrecord(serialized_example):
        # Create a dictionary describing the features.
        feature_description = {
            'label': tf.io.FixedLenFeature([], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'coordinates': tf.io.FixedLenFeature([], tf.string),
            'relative_position': tf.io.FixedLenFeature([], tf.int64)}
        
        # Parse example.
        example = tf.io.parse_single_example(serialized_example, feature_description)

        # Cast label to int64
        label = example['label']
        label = tf.cast(label, tf.int64)

        # Get image.
        image = tf.image.decode_png(example['image_raw'])
        
        # Ensure shape dimensions are constant.
        image = tf.reshape(image, [128, 128, 3])
        
        # Process image.
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
        
        # Task-dependent data.
        question = tf.constant([1, 0], dtype=tf.int64) if task=='SD' else tf.constant([0, 1], dtype=tf.int64)
        sd_w = tf.constant(1, dtype=tf.int64) if task=='SD' else tf.constant(0, dtype=tf.int64)
        rp_w = tf.constant(0, dtype=tf.int64) if task=='SD' else tf.constant(1, dtype=tf.int64)
        
        return (image, question), (label, rel_pos), (sd_w, rp_w)
        
    # Parse dataset.
    dataset = raw_image_dataset.map(read_tfrecord)
    
    # Always shuffle for simplicity.
    dataset = dataset.shuffle(5600)
    
    if is_training:
        # Infinite dataset to avoid the potential last partial batch in each epoch.
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)
    
    return dataset

def test_relnets_all_ds_auc(batch_size):
    # Load same/different datasets.
    datasets = []
    
    # Regular SD.
    ds = get_dataset_relnet_test(
        batch_size=batch_size,
        tfrecord_dir='data/regular_test.tfrecords',
        is_training=False,
        task='SD',
        process_img=True)
    datasets.append(ds)
    
    # Regular RP.
    ds = get_dataset_relnet_test(
        batch_size=batch_size,
        tfrecord_dir='data/regular_test.tfrecords',
        task='RP',
        is_training=False,
        process_img=True)
    datasets.append(ds)
    
    # Irregular SD.
    ds = get_dataset_relnet_test(
        batch_size=batch_size,
        tfrecord_dir='data/irregular_test.tfrecords',
        task='SD',
        is_training=False,
        process_img=True)
    datasets.append(ds)
    
    # Irregular RP.
    ds = get_dataset_relnet_test(
        batch_size=batch_size,
        tfrecord_dir='data/irregular_test.tfrecords',
        task='RP',
        is_training=False,
        process_img=True)
    datasets.append(ds)
    
    # Open SD.
    ds = get_dataset_relnet_test(
        batch_size=batch_size,
        tfrecord_dir='data/open_test.tfrecords',
        task='SD',
        is_training=False,
        process_img=True)
    datasets.append(ds)
    
    # Open RP.
    ds = get_dataset_relnet_test(
        batch_size=batch_size,
        tfrecord_dir='data/open_test.tfrecords',
        task='RP',
        is_training=False,
        process_img=True)
    datasets.append(ds)

    # Wider line SD.
    ds = get_dataset_relnet_test(
        batch_size=batch_size,
        tfrecord_dir='data/wider_line_test.tfrecords',
        task='SD',
        is_training=False,
        process_img=True)
    datasets.append(ds)
    
    # Wider line RP.
    ds = get_dataset_relnet_test(
        batch_size=batch_size,
        tfrecord_dir='data/wider_line_test.tfrecords',
        task='RP',
        is_training=False,
        process_img=True)
    datasets.append(ds)
    
    # Scrambled SD.
    ds = get_dataset_relnet_test(
        batch_size=batch_size,
        tfrecord_dir='data/scrambled_test.tfrecords',
        task='SD',
        is_training=False,
        process_img=True)
    datasets.append(ds)
    
    # Scrambled RP.
    ds = get_dataset_relnet_test(
        batch_size=batch_size,
        tfrecord_dir='data/scrambled_test.tfrecords',
        task='RP',
        is_training=False,
        process_img=True)
    datasets.append(ds)
    
    # Random color SD.
    ds = get_dataset_relnet_test(
        batch_size=batch_size,
        tfrecord_dir='data/random_color_test.tfrecords',
        task='SD',
        is_training=False,
        process_img=True)
    datasets.append(ds)
    
    # Random color RP.
    ds = get_dataset_relnet_test(
        batch_size=batch_size,
        tfrecord_dir='data/random_color_test.tfrecords',
        task='RP',
        is_training=False,
        process_img=True)
    datasets.append(ds)
    
    # Filled SD.
    ds = get_dataset_relnet_test(
        batch_size=batch_size,
        tfrecord_dir='data/filled_test.tfrecords',
        task='SD',
        is_training=False,
        process_img=True)
    datasets.append(ds)
    
    # Filled RP.
    ds = get_dataset_relnet_test(
        batch_size=batch_size,
        tfrecord_dir='data/filled_test.tfrecords',
        task='RP',
        is_training=False,
        process_img=True)
    datasets.append(ds)
    
    # Lines SD.
    ds = get_dataset_relnet_test(
        batch_size=batch_size,
        tfrecord_dir='data/lines_test.tfrecords',
        task='SD',
        is_training=False,
        process_img=True)
    datasets.append(ds)
    
    # Lines RP.
    ds = get_dataset_relnet_test(
        batch_size=batch_size,
        tfrecord_dir='data/lines_test.tfrecords',
        task='RP',
        is_training=False,
        process_img=True)
    datasets.append(ds)
    
    # Arrows SD.
    ds = get_dataset_relnet_test(
        batch_size=batch_size,
        tfrecord_dir='data/arrows_test.tfrecords',
        task='SD',
        is_training=False,
        process_img=True)
    datasets.append(ds)
    
    # Arrows RP.
    ds = get_dataset_relnet_test(
        batch_size=batch_size,
        tfrecord_dir='data/arrows_test.tfrecords',
        task='RP',
        is_training=False,
        process_img=True)
    datasets.append(ds)
    
    conditions = ['Regular', 'Regular', 'Irregular', 'Irregular', 'Open', 'Open', 'Wider Line', 'Wider Line',
                  'Scrambled', 'Scrambled', 'Random Color', 'Random Color', 'Filled', 'Filled', 'Lines', 'Lines', 'Arrows', 'Arrows']
    
    tasks = ['SD', 'RP'] * 9
    
    weights_dirs = ['simulation_4/relnet_no_regular', 'simulation_4/relnet_no_regular',
                    'simulation_4/relnet_no_irregular', 'simulation_4/relnet_no_irregular',
                    'simulation_4/relnet_no_open', 'simulation_4/relnet_no_open',
                    'simulation_4/relnet_no_wider', 'simulation_4/relnet_no_wider',
                    'simulation_4/relnet_no_scrambled', 'simulation_4/relnet_no_scrambled',
                    'simulation_4/relnet_no_random', 'simulation_4/relnet_no_random',
                    'simulation_4/relnet_no_filled', 'simulation_4/relnet_no_filled',
                    'simulation_4/relnet_no_lines', 'simulation_4/relnet_no_lines',
                    'simulation_4/relnet_no_arrows', 'simulation_4/relnet_no_arrows']
    
    results = []
    for ds, condition, task, w_dir in zip(datasets, conditions, tasks, weights_dirs):
        # Fine-tune, imagenet, GAP.
        ft_im_gap = test_relnet_auc(
            ds=ds,
            weights_dir=w_dir,
            model_name='RN',
            condition=condition,
            task=task)
        results.extend(ft_im_gap)
        
    return pd.DataFrame(results, columns=['Model', 'Condition', 'Task', 'AUC'])

if __name__ == '__main__':
    # Training hyperparameters.
    EPOCHS_TOP = 5
    EPOCHS = 13
    BATCH_SIZE = 64
    STEPS_PER_EPOCH = 28000 // BATCH_SIZE
    VALIDATION_STEPS = 2800 // BATCH_SIZE

    # Train Relation Network.
    train_in_all_sd_relnet(
        epochs_top=EPOCHS_TOP,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        n=10,
        lr=0.0001,
        batch_size=BATCH_SIZE)
    
    # Test Relation Network.
    df = test_relnets_all_ds_auc(batch_size=112)
    df.to_csv('simulation_4/sim_4_relnet_auc.csv')
