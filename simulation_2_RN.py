# Imports
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152
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
    trainable=False
    ):
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
    
    # Output.
    if dataset == 'sort-of-clevr':
        output_units = 10
        answer = Dense(output_units, activation='softmax')(rn)
    elif dataset == 'SVRT':
        output_units = 1
        answer = Dense(output_units, activation='sigmoid')(rn)
    
    model = Model(inputs=[image, question], outputs=answer)
    
    return base_model, model

# Dataset reading functions
def get_dataset(
    batch_size, 
    tfrecord_dir, 
    autotune_settings,
    is_training=True, 
    process_img=True, 
    relnet=False
    ):
    # Load dataset
    raw_image_dataset = tf.data.TFRecordDataset(tfrecord_dir, num_parallel_reads=autotune_settings)
    
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
        
        if relnet:
            question = tf.constant([1, 0], dtype=tf.int64)
            return (image, question), label
        else:
            return image, label
            
    # Parse dataset.
    dataset = raw_image_dataset.map(read_tfrecord, num_parallel_calls=autotune_settings)
    
    # Always shuffle for simplicity.
    dataset = dataset.shuffle(5600)
    
    if is_training:
        # Infinite dataset to avoid the potential last partial batch in each epoch.
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size).prefetch(autotune_settings)
    
    return dataset

def get_master_dataset(
    autotune_settings,
    batch_size,
    ds_dir,
    dont_include,
    process_img=True,
    relnet=False
    ):
    """Builds dataset that samples each batch from one of the training
    datasets except 'dont_include'."""
    
    # Make datasets and append if it is not the dont_include dataset.
    datasets = []
    
    # Original.
    ds_original = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/original_train.tfrecords',
        autotune_settings=autotune_settings,
        is_training=True,
        process_img=process_img,
        relnet=relnet
        )
    datasets.append(ds_original)
    
    # Regular.
    ds_regular = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/regular_train.tfrecords',
        autotune_settings=autotune_settings,
        is_training=True,
        process_img=process_img,
        relnet=relnet
        )
    datasets.append(ds_regular)
    
    # Irregular.
    ds_irregular = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/irregular_train.tfrecords',
        autotune_settings=autotune_settings,
        is_training=True,
        process_img=process_img,
        relnet=relnet
        )
    datasets.append(ds_irregular)
    
    # Open.
    ds_open = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/open_train.tfrecords',
        autotune_settings=autotune_settings,
        is_training=True,
        process_img=process_img,
        relnet=relnet
        )
    datasets.append(ds_open)

    # Wider line.
    ds_wider = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/wider_line_train.tfrecords',
        autotune_settings=autotune_settings,
        is_training=True,
        process_img=process_img,
        relnet=relnet
        )
    datasets.append(ds_wider)
    
    # Scrambled.
    ds_scrambled = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/scrambled_train.tfrecords',
        autotune_settings=autotune_settings,
        is_training=True,
        process_img=process_img,
        relnet=relnet
        )
    datasets.append(ds_scrambled)
    
    # Random color.
    ds_random = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/random_color_train.tfrecords',
        autotune_settings=autotune_settings,
        is_training=True,
        process_img=process_img,
        relnet=relnet
        )
    datasets.append(ds_random)
    
    # Filled.
    ds_filled = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/filled_train.tfrecords',
        autotune_settings=autotune_settings,
        is_training=True,
        process_img=process_img,
        relnet=relnet
        )
    datasets.append(ds_filled)
    
    # Lines.
    ds_lines = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/lines_train.tfrecords',
        autotune_settings=autotune_settings,
        is_training=True,
        process_img=process_img,
        relnet=relnet
        )
    datasets.append(ds_lines)
    
    # Arrows.
    ds_arrows = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/arrows_train.tfrecords',
        autotune_settings=autotune_settings,
        is_training=True,
        process_img=process_img,
        relnet=relnet
        )
    datasets.append(ds_arrows)

    # Define dataset sampling. Because 'Lines' is harder to learn when learning 
    # the other at the same time, oversample it!
    if dont_include == 'Original':
        ds_to_sample_from = [8, 1, 2, 3, 4, 8, 5, 6, 7, 8, 9]
    elif dont_include == 'Regular':
        ds_to_sample_from = [0, 8, 2, 3, 8, 4, 5, 6, 7, 8, 9]
    elif dont_include == 'Irregular':
        ds_to_sample_from = [0, 1, 8, 3, 4, 8, 5, 6, 7, 8, 9]
    elif dont_include == 'Open':
        ds_to_sample_from = [8, 0, 1, 2, 8, 4, 5, 6, 7, 8, 9]
    elif dont_include == 'Wider line':
        ds_to_sample_from = [8, 0, 1, 2, 3, 8, 5, 6, 7, 8, 9]
    elif dont_include == 'Scrambled':
        ds_to_sample_from = [8, 0, 1, 2, 3, 4, 8, 6, 7, 8, 9]
    elif dont_include == 'Random color':
        ds_to_sample_from = [8, 0, 1, 2, 3, 4, 5, 8, 7, 8, 9]
    elif dont_include == 'Filled':
        ds_to_sample_from = [8, 0, 1, 2, 8, 3, 4, 5, 6, 8, 9]
    elif dont_include == 'Lines':
        ds_to_sample_from = [0, 1, 2, 3, 4, 5, 6, 7, 9]
    elif dont_include == 'Arrows':
        ds_to_sample_from = [0, 1, 8, 2, 3, 4, 8, 5, 6, 7, 8]
    else:
        raise ValueError('unrecognized dont_include condition!')

    choice_tensor = tf.constant(value=ds_to_sample_from, dtype=tf.int64)
    choice_dataset = tf.data.Dataset.from_tensor_slices(choice_tensor).repeat()
    
    return tf.data.experimental.choose_from_datasets(datasets, choice_dataset)

# Training function
def pretrain_best_relnet_sim1(
    strategy,
    train_ds,
    val_ds,
    save_name,
    model,
    base_model,
    model_name,
    epochs,
    steps_per_epoch,
    validation_steps,
    start=0,
    stop=10,
    lr=0.0003
    ):
    # Repate trainign 10 times
    for i in range(start, stop):
        # Initialize weights
        for layer in model.layers:
            if hasattr(layer, 'kernel_initializer') and hasattr(layer, 'bias_initializer'):
                layer.set_weights(
                    [layer.kernel_initializer(shape=np.asarray(layer.kernel.shape)),
                     layer.bias_initializer(shape=np.asarray(layer.bias.shape))
                     ])
        # Load imagenet weights
        base_model.load_weights('Resnet50_no_top.h5')
        # Train
        filename = save_name + model_name +'_run_' + str(i) + '_log.csv'
        history_logger = tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)
        model.fit(
            train_ds,
            epochs=epochs,
            verbose=2,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,
            validation_steps=validation_steps,
            callbacks=[history_logger]
            )
        # Save weights
        weights_name = save_name + model_name + '_instance_' + str(i) + '.hdf5'
        model.save_weights(weights_name)
    return

def pretrain_relnet_in_all_datasets(
    strategy,
    autotune_settings,
    ds_dir,
    sim_dir,
    epochs,
    steps_per_epoch,
    validation_steps,
    n=10,
    lr=0.0003,
    batch_size=64):
    
    # Define all master datasets
    no_irregular_ds = get_master_dataset(batch_size=batch_size, dont_include='Irregular', ds_dir=ds_dir, process_img=True, relnet=True, autotune_settings=autotune_settings)
    no_regular_ds = get_master_dataset(batch_size=batch_size, dont_include='Regular', ds_dir=ds_dir, process_img=True, relnet=True, autotune_settings=autotune_settings)
    no_open_ds = get_master_dataset(batch_size=batch_size, dont_include='Open', ds_dir=ds_dir, process_img=True, relnet=True, autotune_settings=autotune_settings)
    no_wider_ds = get_master_dataset(batch_size=batch_size, dont_include='Wider line', ds_dir=ds_dir, process_img=True, relnet=True, autotune_settings=autotune_settings)
    no_scrambled_ds = get_master_dataset(batch_size=batch_size, dont_include='Scrambled', ds_dir=ds_dir, process_img=True, relnet=True, autotune_settings=autotune_settings)
    no_random_ds = get_master_dataset(batch_size=batch_size, dont_include='Random color', ds_dir=ds_dir, process_img=True, relnet=True, autotune_settings=autotune_settings)
    no_filled_ds = get_master_dataset(batch_size=batch_size, dont_include='Filled', ds_dir=ds_dir, process_img=True, relnet=True, autotune_settings=autotune_settings)
    no_lines_ds = get_master_dataset(batch_size=batch_size, dont_include='Lines', ds_dir=ds_dir, process_img=True, relnet=True, autotune_settings=autotune_settings)
    no_arrows_ds = get_master_dataset(batch_size=batch_size, dont_include='Arrows', ds_dir=ds_dir, process_img=True, relnet=True, autotune_settings=autotune_settings)
    
    ds_and_names = [
        (no_irregular_ds, sim_dir+'/relnet_no_irregular/'),
        (no_regular_ds, sim_dir+'/relnet_no_regular/'),
        (no_open_ds, sim_dir+'/relnet_no_open/'),
        (no_wider_ds, sim_dir+'/relnet_no_wider/'),
        (no_scrambled_ds, sim_dir+'/relnet_no_scrambled/'),
        (no_random_ds, sim_dir+'/relnet_no_random/'),
        (no_filled_ds, sim_dir+'/relnet_no_filled/'),
        (no_lines_ds, sim_dir+'/relnet_no_lines/'),
        (no_arrows_ds, sim_dir+'/relnet_no_arrows/')
        ]
    with strategy.scope():
        # Define model
        base_model, model = make_ResNet50_relnet(
            dataset='SVRT',
            resnet_layer='last_size_8',
            trainable=False
            )
        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss='binary_crossentropy',
            metrics=['accuracy'],
            steps_per_execution=50
            )
        for ds, name in ds_and_names:
            pretrain_best_relnet_sim1(
                strategy=strategy,
                train_ds=ds,
                val_ds=None,
                save_name=name,
                model=model,
                base_model=base_model,
                model_name='RN',
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_steps=None,
                start=0,
                stop=10,
                lr=lr
                )
    return

def train_best_relnet_sim1(
    strategy,
    train_ds,
    val_ds,
    save_name,
    model,
    model_name,
    epochs,
    steps_per_epoch,
    validation_steps,
    start=0,
    stop=10,
    lr=0.0001
    ):
    # Repeate trainign 10 times
    for i in range(start, stop):
        # Load weights 
        weights_name = save_name + model_name + '_instance_' + str(i) + '.hdf5'
        model.load_weights(weights_name)
        # Train
        filename = save_name + model_name +'_run_' + str(i) + '_log.csv'
        history_logger = tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)
        model.fit(
            train_ds,
            epochs=epochs,
            verbose=2,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,
            validation_steps=validation_steps,
            callbacks=[history_logger]
            )
        # Save weights
        weights_name = save_name + model_name + '_instance_' + str(i) + '.hdf5'
        model.save_weights(weights_name)
    return

def train_relnet_in_all_datasets(
    strategy,
    autotune_settings,
    ds_dir,
    sim_dir,
    epochs,
    steps_per_epoch,
    validation_steps,
    n=10,
    lr=0.0001,
    batch_size=64):
    
    # Define all master datasets
    no_irregular_ds = get_master_dataset(batch_size=batch_size, dont_include='Irregular', ds_dir=ds_dir, process_img=True, relnet=True, autotune_settings=autotune_settings)
    no_regular_ds = get_master_dataset(batch_size=batch_size, dont_include='Regular', ds_dir=ds_dir, process_img=True, relnet=True, autotune_settings=autotune_settings)
    no_open_ds = get_master_dataset(batch_size=batch_size, dont_include='Open', ds_dir=ds_dir, process_img=True, relnet=True, autotune_settings=autotune_settings)
    no_wider_ds = get_master_dataset(batch_size=batch_size, dont_include='Wider line', ds_dir=ds_dir, process_img=True, relnet=True, autotune_settings=autotune_settings)
    no_scrambled_ds = get_master_dataset(batch_size=batch_size, dont_include='Scrambled', ds_dir=ds_dir, process_img=True, relnet=True, autotune_settings=autotune_settings)
    no_random_ds = get_master_dataset(batch_size=batch_size, dont_include='Random color', ds_dir=ds_dir, process_img=True, relnet=True, autotune_settings=autotune_settings)
    no_filled_ds = get_master_dataset(batch_size=batch_size, dont_include='Filled', ds_dir=ds_dir, process_img=True, relnet=True, autotune_settings=autotune_settings)
    no_lines_ds = get_master_dataset(batch_size=batch_size, dont_include='Lines', ds_dir=ds_dir, process_img=True, relnet=True, autotune_settings=autotune_settings)
    no_arrows_ds = get_master_dataset(batch_size=batch_size, dont_include='Arrows', ds_dir=ds_dir, process_img=True, relnet=True, autotune_settings=autotune_settings)
    
    ds_and_names = [
        (no_irregular_ds, sim_dir+'/relnet_no_irregular/'),
        (no_regular_ds, sim_dir+'/relnet_no_regular/'),
        (no_open_ds, sim_dir+'/relnet_no_open/'),
        (no_wider_ds, sim_dir+'/relnet_no_wider/'),
        (no_scrambled_ds, sim_dir+'/relnet_no_scrambled/'),
        (no_random_ds, sim_dir+'/relnet_no_random/'),
        (no_filled_ds, sim_dir+'/relnet_no_filled/'),
        (no_lines_ds, sim_dir+'/relnet_no_lines/'),
        (no_arrows_ds, sim_dir+'/relnet_no_arrows/')
        ]
    with strategy.scope():
        # Define model
        base_model, model = make_ResNet50_relnet(
            dataset='SVRT',
            resnet_layer='last_size_8',
            trainable=True
            )
        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss='binary_crossentropy',
            metrics=['accuracy'],
            steps_per_execution=50
            )
        for ds, name in ds_and_names:
            train_best_relnet_sim1(
                strategy=strategy,
                train_ds=ds,
                val_ds=None,
                save_name=name,
                model=model,
                model_name='RN',
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_steps=None,
                start=2,
                stop=10,
                lr=lr
                )
    return

# Validation functions
def test_model_val_auc(
    ds_val, 
    ds_name,
    weights_dir,
    model,
    model_name, 
    condition
    ):
    """
    Tests 10 versions of a single model in a single condition using the area under the ROC curve. 
    Args:
        ds_val: single dataset with cases from the 'same' and 'different' classes.
        ds_name: name of the dataset being tested. String.
        weights_dir: directory of models weights to test.
        gap: wheather to use GAP. True or False.
        model_name: model name. String.
        condition: name of the dataset in which the model was not trained on. String.
    Returns:
        A list with test data: model_name, condition, ds_name, area under the ROC curve.
    """
    
    # Get list of weights from path
    weights_list = [f for f in listdir(weights_dir) if f.endswith('.hdf5')]
    weights_paths = [join(weights_dir, f) for f in weights_list if isfile(join(weights_dir, f))]

    # Test each model
    models_data = []
    for w_path in weights_paths:
        model.load_weights(w_path)
        metrics_val = model.evaluate(ds_val)
        models_data.append([model_name, condition, ds_name, metrics_val[1]])
    return models_data

def test_model_val_auc_on_datastets(
    names_and_datasets,
    weights_dir,
    condition,
    model,
    model_name='RN'
    ):
    """
    Test 10 models in weights_dir on each dataset in names_and_datasets.
    Args:
        names_and_datasets: list of tuples (name, dataset).
        weights_dir: directory with the weights of 10 runs of a model.
        condition: name of the dataset the model was not trained on.
        gap: wheather to use GAP. True or False.
        model_name: model name. String.
    Returns:
        A list with test data: model_name, condition, ds_name, area under the ROC curve.
    """
    datasets_data = []
    for ds_name, ds in names_and_datasets:
        models_data = test_model_val_auc(
            ds_val=ds, 
            ds_name=ds_name, 
            weights_dir=weights_dir, 
            model=model, 
            model_name=model_name, 
            condition=condition
            )
        datasets_data.extend(models_data)
    return datasets_data 

def test_all_conditions_val_auc(
    strategy,
    autotune_settings,
    batch_size, 
    ds_dir,
    sim_dir,
    relnet=True
    ):
    original_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/original_val.tfrecords',
        is_training=False,
        process_img=True,
        relnet=relnet,
        autotune_settings=autotune_settings
        )
    regular_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/regular_val.tfrecords',
        is_training=False,
        process_img=True,
        relnet=relnet,
        autotune_settings=autotune_settings
        )
    irregular_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/irregular_val.tfrecords',
        is_training=False,
        process_img=True,
        relnet=relnet,
        autotune_settings=autotune_settings
        )
    open_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/open_val.tfrecords',
        is_training=False,
        process_img=True,
        relnet=relnet,
        autotune_settings=autotune_settings
        )
    wider_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/wider_line_val.tfrecords',
        is_training=False,
        process_img=True,
        relnet=relnet,
        autotune_settings=autotune_settings
        )
    scrambled_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/scrambled_val.tfrecords',
        is_training=False,
        process_img=True,
        relnet=relnet,
        autotune_settings=autotune_settings
        )
    random_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/random_color_val.tfrecords',
        is_training=False,
        process_img=True,
        relnet=relnet,
        autotune_settings=autotune_settings
        )
    filled_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/filled_val.tfrecords',
        is_training=False,
        process_img=True,
        relnet=relnet,
        autotune_settings=autotune_settings
        )
    lines_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/lines_val.tfrecords',
        is_training=False,
        process_img=True,
        relnet=relnet,
        autotune_settings=autotune_settings
        )
    arrows_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/arrows_val.tfrecords',
        is_training=False,
        process_img=True,
        relnet=relnet,
        autotune_settings=autotune_settings
        )
    names_and_ds = [
                    ('Original', original_ds),
                    ('Regular', regular_ds),
                    ('Irregular', irregular_ds),
                    ('Open', open_ds),
                    ('Wider Line', wider_ds),
                    ('Scrambled', scrambled_ds),
                    ('Random Color', random_ds),
                    ('Filled', filled_ds),
                    ('Lines', lines_ds),
                    ('Arrows', arrows_ds)
                    ]
    weights_dirs = [
                    sim_dir+'/relnet_no_regular', 
                    sim_dir+'/relnet_no_irregular', 
                    sim_dir+'/relnet_no_open',
                    sim_dir+'/relnet_no_wider', 
                    sim_dir+'/relnet_no_scrambled',
                    sim_dir+'/relnet_no_random',
                    sim_dir+'/relnet_no_filled', 
                    sim_dir+'/relnet_no_lines', 
                    sim_dir+'/relnet_no_arrows'
                    ]
    # Define model architecture
    with strategy.scope():
        base_model, model = make_ResNet50_relnet(
            dataset='SVRT',
            resnet_layer='last_size_8',
            trainable=True
            )
        # Compile: set metric to auc
        model.compile(
                optimizer=tf.keras.optimizers.Adam(0.0003),
                loss='binary_crossentropy',
                metrics=[tf.keras.metrics.AUC()]
                )
        results = []
        for w_dir in weights_dirs:
            # Identify ds to ignore for condition
            if w_dir.endswith('no_regular'):
                ds_to_ignore = 'Regular'
            elif w_dir.endswith('no_irregular'):
                ds_to_ignore = 'Irregular'
            elif w_dir.endswith('no_open'):
                ds_to_ignore = 'Open'
            elif w_dir.endswith('no_wider'):
                ds_to_ignore = 'Wider Line'
            elif w_dir.endswith('no_scrambled'):
                ds_to_ignore = 'Scrambled'
            elif w_dir.endswith('no_random'):
                ds_to_ignore = 'Random Color'
            elif w_dir.endswith('no_filled'):
                ds_to_ignore = 'Filled'
            elif w_dir.endswith('no_lines'):
                ds_to_ignore = 'Lines'
            elif w_dir.endswith('no_arrows'):
                ds_to_ignore = 'Arrows'
            else:
                raise ValueError('Unrecognised dataset name!')
            condition_names_and_ds = [x for x in names_and_ds if x[0] != ds_to_ignore]
            condition_data = test_model_val_auc_on_datastets(
                names_and_datasets=condition_names_and_ds,
                weights_dir=w_dir,
                condition=ds_to_ignore,
                model=model,
                model_name='RN'
                )
            results.extend(condition_data)
    return pd.DataFrame(results, columns=['Model', 'Condition', 'Testing data', 'AUC'])

# Test functions
# Note get_master_test_dataset is not used because I have the validation datasets
def get_master_test_dataset(
    batch_size, 
    ds_dir, 
    dont_include, 
    autotune_settings, 
    relnet=False
    ):
    """Returns test dataset with all the s/d files except the ones
    corresponding to the dont_include condition."""
    
    # Define all condition records.
    all_records = []

    if dont_include != 'Original':
        all_records.append(ds_dir+'/original_test.tfrecords')
        
    if dont_include != 'Regular':
        all_records.append(ds_dir+'/regular_test.tfrecords')
        
    if dont_include != 'Irregular':
        all_records.append(ds_dir+'/irregular_test.tfrecords')
        
    if dont_include != 'Open':
        all_records.append(ds_dir+'/open_test.tfrecords')
    
    if dont_include != 'Wider line':
        all_records.append(ds_dir+'/wider_line_test.tfrecords')
        
    if dont_include != 'Scrambled':
        all_records.append(ds_dir+'/scrambled_test.tfrecords')
        
    if dont_include != 'Random color':
        all_records.append(ds_dir+'/random_color_test.tfrecords')
        
    if dont_include != 'Filled':
        all_records.append(ds_dir+'/filled_test.tfrecords')
        
    if dont_include != 'Lines':
        all_records.append(ds_dir+'/lines_test.tfrecords')
    
    if dont_include != 'Arrows':
        all_records.append(ds_dir+'/arrows_test.tfrecords')
    
    ds = get_dataset(
        batch_size=batch_size,
        autotune_settings=autotune_settings,
        tfrecord_dir=all_records,
        is_training=False,
        process_img=True,
        relnet=relnet)
    
    return ds

def test_relnet_auc(
    ds_untrained, 
    weights_dir,
    model,
    model_name, 
    condition
    ):
    """Tests 10 versions of a single relation network in a single condition using the area under the ROC curve. 
    Args:
        ds_untrained: data from the condition not trained on.
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
        metrics_untrained = model.evaluate(ds_untrained)
        models_data.append([model_name, condition, 'Untrained', metrics_untrained[1]])
    
    return models_data

def test_relnet_all_ds_auc(
    strategy,
    autotune_settings,
    batch_size, 
    ds_dir, 
    sim_dir
    ):
    # Load same/different datasets.
    untrained_dss = []
        
    # Regular
    regular_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/regular_test.tfrecords',
        is_training=False,
        process_img=True,
        relnet=True,
        autotune_settings=autotune_settings
        )    
    untrained_dss.append(regular_ds)
    
    # Irregular
    irregular_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/irregular_test.tfrecords',
        is_training=False,
        process_img=True,
        relnet=True,
        autotune_settings=autotune_settings
        )
    untrained_dss.append(irregular_ds)
    
    # Open
    open_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/open_test.tfrecords',
        is_training=False,
        process_img=True,
        relnet=True,
        autotune_settings=autotune_settings
        )    
    untrained_dss.append(open_ds)

    # Wider line
    wider_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/wider_line_test.tfrecords',
        is_training=False,
        process_img=True,
        relnet=True,
        autotune_settings=autotune_settings
        )    
    untrained_dss.append(wider_ds)
    
    # Scrambled
    scrambled_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/scrambled_test.tfrecords',
        is_training=False,
        process_img=True,
        relnet=True,
        autotune_settings=autotune_settings
        )    
    untrained_dss.append(scrambled_ds)
    
    # Random color
    random_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/random_color_test.tfrecords',
        is_training=False,
        process_img=True,
        relnet=True,
        autotune_settings=autotune_settings
        )    
    untrained_dss.append(random_ds)
    
    # Filled
    filled_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/filled_test.tfrecords',
        is_training=False,
        process_img=True,
        relnet=True,
        autotune_settings=autotune_settings
        )    
    untrained_dss.append(filled_ds)
    
    # Lines
    lines_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/lines_test.tfrecords',
        is_training=False,
        process_img=True,
        relnet=True,
        autotune_settings=autotune_settings
        )    
    untrained_dss.append(lines_ds)

    # Arrows
    arrows_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/arrows_test.tfrecords',
        is_training=False,
        process_img=True,
        relnet=True,
        autotune_settings=autotune_settings
        )    
    untrained_dss.append(arrows_ds)
    
    conditions = ['Regular', 'Irregular', 'Open', 'Wider Line', 'Scrambled', 'Random Color', 'Filled', 'Lines', 'Arrows']
    weights_dirs = [
                    sim_dir+'/relnet_no_regular', 
                    sim_dir+'/relnet_no_irregular', 
                    sim_dir+'/relnet_no_open',
                    sim_dir+'/relnet_no_wider', 
                    sim_dir+'/relnet_no_scrambled',
                    sim_dir+'/relnet_no_random',
                    sim_dir+'/relnet_no_filled', 
                    sim_dir+'/relnet_no_lines', 
                    sim_dir+'/relnet_no_arrows'
                    ]
    # Define model architecture
    with strategy.scope():
        base_model, model = make_ResNet50_relnet(
            dataset='SVRT',
            resnet_layer='last_size_8',
            trainable=True
            )
        # Compile: set metric to auc (default is area under the ROC curve)
        model.compile(
                optimizer=tf.keras.optimizers.Adam(0.0003),
                loss='binary_crossentropy',
                metrics=[tf.keras.metrics.AUC()]
                )
        results = []
        for ds2, condition, w_dir in zip(untrained_dss, conditions, weights_dirs):
            data = test_relnet_auc(
                ds_untrained=ds2,
                weights_dir=w_dir,
                model=model,
                model_name='RN',
                condition=condition)
            results.extend(data)
    
    return pd.DataFrame(results, columns=['Model', 'Condition', 'Testing data', 'AUC'])


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
    pretrain_relnet_in_all_datasets(
        strategy=strategy,
        autotune_settings=AUTO,
        ds_dir='data',
        sim_dir='simulation_2',
        epochs=EPOCHS_TOP,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        n=10,
        lr=0.0003,
        batch_size=BATCH_SIZE
        )
    # Train
    train_relnet_in_all_datasets(
        strategy=strategy,
        autotune_settings=AUTO,
        ds_dir='data',
        sim_dir='simulation_2',
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        n=10,
        lr=0.0001,
        batch_size=BATCH_SIZE
        )
    # Validate and save
    df = test_all_conditions_val_auc(
        strategy=strategy,
        autotune_settings=AUTO,
        batch_size=560, 
        ds_dir='data',
        sim_dir='simulation_2',
        relnet=True
        )
    df.to_csv('simulation_2/sim_2_RN_val_auc.csv')

    # Test and save.
    df = test_relnet_all_ds_auc(
        strategy=strategy,
        autotune_settings=AUTO,
        batch_size=560,
        ds_dir='data',
        sim_dir='simulation_2'
        )
    df.to_csv('simulation_2/sim_2_RN_test_auc.csv')
    
