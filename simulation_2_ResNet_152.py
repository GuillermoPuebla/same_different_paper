# Imports
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Input


# Model definition
def make_ResNet50_sketch_classifier(weights=None, trainable=True):
    # Inputs.
    image = Input((128, 128, 3))

    # Get CNN features.
    base_model = ResNet152(weights=weights, include_top=False, input_tensor=image)

    # Freeze the base_model.
    base_model.trainable = trainable
    
    # Add a global spatial average pooling layer.
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # Add a fully-connected layer.
    x = Dense(1024, activation='relu')(x)
    
    # Add a logistic layer.
    predictions = Dense(250, activation='softmax')(x)
    
    # Model to train.
    model = Model(inputs=base_model.input, outputs=predictions)

    return base_model, model

def make_ResNet50_sd_classifier(base_weights='imagenet', tuberlin_path=None, base_trainable=False, gap=True):
    # Define input tensor
    image = Input((128, 128, 3))
    # Base pre-trained model
    if base_weights == 'imagenet':
        base_model = ResNet152(weights='imagenet', include_top=False, input_tensor=image)
    elif base_weights is None:
        base_model = ResNet152(weights=None, include_top=False, input_tensor=image)
    elif base_weights == 'TU-Berlin':
        # Get weights from sketch classifier base
        sketch_b, sketch_m = make_ResNet50_sketch_classifier(weights=None, trainable=True)
        sketch_m.load_weights(tuberlin_path)
        base_weights = []
        for layer in sketch_b.layers:
            base_weights.append(layer.get_weights())
        # Set weights of base model
        base_model = ResNet152(weights=None, include_top=False, input_tensor=image)
        i = 0
        for layer in base_model.layers:
            layer.set_weights(base_weights[i])
            i += 1
    # Freeze the if necesarry base_model
    base_model.trainable = base_trainable
    # Add a global spatial average pooling layer
    x = base_model.output
    if gap:
        x = GlobalAveragePooling2D()(x)
    else:
        x = Flatten()(x)
    # Add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # Add logistic layer
    predictions = Dense(1, activation='sigmoid')(x)
    # This is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

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
        is_training=True,
        process_img=process_img,
        relnet=relnet,
        autotune_settings=autotune_settings)
    
    # if dont_include != 'Original':
    datasets.append(ds_original)
    
    # Regular.
    ds_regular = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/regular_train.tfrecords',
        is_training=True,
        process_img=process_img,
        relnet=relnet,
        autotune_settings=autotune_settings)
    
    # if dont_include != 'Regular':
    datasets.append(ds_regular)
    
    # Irregular.
    ds_irregular = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/irregular_train.tfrecords',
        is_training=True,
        process_img=process_img,
        relnet=relnet,
        autotune_settings=autotune_settings)
    
    # if dont_include != 'Irregular':
    datasets.append(ds_irregular)
    
    # Open.
    ds_open = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/open_train.tfrecords',
        is_training=True,
        process_img=process_img,
        relnet=relnet,
        autotune_settings=autotune_settings)
    
    # if dont_include != 'Open':
    datasets.append(ds_open)

    # Wider line.
    ds_wider = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/wider_line_train.tfrecords',
        is_training=True,
        process_img=process_img,
        relnet=relnet,
        autotune_settings=autotune_settings)
    
    # if dont_include != 'Wider line':
    datasets.append(ds_wider)

    # Scrambled.
    ds_scrambled = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/scrambled_train.tfrecords',
        is_training=True,
        process_img=process_img,
        relnet=relnet,
        autotune_settings=autotune_settings)

    # if dont_include != 'Scrambled':
    datasets.append(ds_scrambled)
    
    # Random color.
    ds_random = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/random_color_train.tfrecords',
        is_training=True,
        process_img=process_img,
        relnet=relnet,
        autotune_settings=autotune_settings)
    
    # if dont_include != 'Random color':
    datasets.append(ds_random)
    
    # Filled.
    ds_filled = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/filled_train.tfrecords',
        is_training=True,
        process_img=process_img,
        relnet=relnet,
        autotune_settings=autotune_settings)
    
    # if dont_include != 'Filled':
    datasets.append(ds_filled)
    
    # Lines.
    ds_lines = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/lines_train.tfrecords',
        is_training=True,
        process_img=process_img,
        relnet=relnet,
        autotune_settings=autotune_settings)
    
    # if dont_include != 'Lines':
    datasets.append(ds_lines)
    
    # Arrows.
    ds_arrows = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/arrows_train.tfrecords',
        is_training=True,
        process_img=process_img,
        relnet=relnet,
        autotune_settings=autotune_settings)
    
    # if dont_include != 'Arrows':
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
    
    # return tf.data.experimental.choose_from_datasets(datasets, choice_dataset)
    return tf.data.Dataset.choose_from_datasets(datasets, choice_dataset)

# Training functions
def train_best_sim1_model(
    strategy,
    train_ds,
    val_ds,
    save_name,
    model_name,
    epochs_top,
    epochs,
    steps_per_epoch,
    validation_steps,
    n=10,
    lr=0.0001
    ):
    with strategy.scope():
        for i in range(n):
            # Define best model sim1: ImageNet & fine-tune & GAP
            base_model, model = make_ResNet50_sd_classifier(
                base_weights='imagenet',
                base_trainable=False,
                gap=True
                )
            # Compile
            model.compile(
                optimizer=tf.keras.optimizers.Adam(0.0003),
                loss='binary_crossentropy',
                metrics=['accuracy']
                )
            # Train
            filename = save_name + model_name +'_run_' + str(i) + '_log.csv'
            history_logger = tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)
            model.fit(
                train_ds,
                epochs=epochs_top,
                verbose=2,
                steps_per_epoch=steps_per_epoch,
                validation_data=val_ds,
                validation_steps=validation_steps,
                callbacks=[history_logger]
                )
            # Unfreeze Resnet50.
            base_model.trainable = True
            # Re-compile
            model.compile(
                optimizer=tf.keras.optimizers.Adam(lr),
                loss='binary_crossentropy',
                metrics=['accuracy']
                )
            # Train
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
            weights_name = save_name + model_name + 'instance_' + str(i) + '.hdf5'
            model.save_weights(weights_name)
    return

def train_in_all_datasets(
    strategy,
    autotune_settings,
    ds_dir,
    sim_dir,
    epochs_top,
    epochs,
    steps_per_epoch,
    validation_steps,
    n=10,
    lr=0.0001,
    batch_size=64
    ):
    # Define all master datasets.
    no_irregular_ds = get_master_dataset(batch_size=batch_size, ds_dir=ds_dir, dont_include='Irregular', process_img=True, autotune_settings=autotune_settings)
    no_regular_ds = get_master_dataset(batch_size=batch_size, ds_dir=ds_dir, dont_include='Regular', process_img=True, autotune_settings=autotune_settings)
    no_open_ds = get_master_dataset(batch_size=batch_size, ds_dir=ds_dir, dont_include='Open', process_img=True, autotune_settings=autotune_settings)
    no_wider_ds = get_master_dataset(batch_size=batch_size, ds_dir=ds_dir, dont_include='Wider line', process_img=True, autotune_settings=autotune_settings)
    no_scrambled_ds = get_master_dataset(batch_size=batch_size, ds_dir=ds_dir, dont_include='Scrambled', process_img=True, autotune_settings=autotune_settings)
    no_random_ds = get_master_dataset(batch_size=batch_size, ds_dir=ds_dir, dont_include='Random color', process_img=True, autotune_settings=autotune_settings)
    no_filled_ds = get_master_dataset(batch_size=batch_size, ds_dir=ds_dir, dont_include='Filled', process_img=True, autotune_settings=autotune_settings)
    no_lines_ds = get_master_dataset(batch_size=batch_size, ds_dir=ds_dir, dont_include='Lines', process_img=True, autotune_settings=autotune_settings)
    no_arrows_ds = get_master_dataset(batch_size=batch_size, ds_dir=ds_dir, dont_include='Arrows', process_img=True, autotune_settings=autotune_settings)
    
    # Train model in each dataset 10 times.
    ds_and_names = [
                    (no_irregular_ds, sim_dir+'/ResNet152_no_irregular/'),
                    (no_regular_ds, sim_dir+'/ResNet152_no_regular/'),
                    (no_open_ds, sim_dir+'/ResNet152_no_open/'),
                    (no_wider_ds, sim_dir+'/ResNet152_no_wider/'),
                    (no_scrambled_ds, sim_dir+'/ResNet152_no_scrambled/'),
                    (no_random_ds, sim_dir+'/ResNet152_no_random/'),
                    (no_filled_ds, sim_dir+'/ResNet152_no_filled/'),
                    (no_lines_ds, sim_dir+'/ResNet152_no_lines/'),
                    (no_arrows_ds, sim_dir+'/ResNet152_no_arrows/')
                    ]
    
    for ds, name in ds_and_names:
        print(name)
        train_best_sim1_model(
            strategy=strategy,
            train_ds=ds,
            val_ds=None,
            save_name=name,
            model_name='ResNet152',
            epochs_top=epochs_top,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=None,
            n=n,
            lr=lr
            )
    return

# Validation functions
def test_model_val_auc(ds_val, ds_name, weights_dir, gap, model_name, condition):
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
    # Define model
    base_model, model = make_ResNet50_sd_classifier(base_weights=None, base_trainable=True, gap=gap)
    # Compile: set metric to auc (default is area under the ROC curve)
    model.compile(
            optimizer=tf.keras.optimizers.Adam(0.0003),
            loss='binary_crossentropy',
            metrics=[tf.keras.metrics.AUC()]
            )
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
    gap=True,
    model_name='ResNet152',
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
            gap=True, 
            model_name='ResNet152', 
            condition=condition
            )
        datasets_data.extend(models_data)
    return datasets_data 

def test_all_conditions_val_auc(
    autotune_settings,
    batch_size, 
    ds_dir,
    sim_dir,
    relnet=False
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
                    sim_dir+'/ResNet152_no_regular', 
                    sim_dir+'/ResNet152_no_irregular', 
                    sim_dir+'/ResNet152_no_open',
                    sim_dir+'/ResNet152_no_wider', 
                    sim_dir+'/ResNet152_no_scrambled',
                    sim_dir+'/ResNet152_no_random',
                    sim_dir+'/ResNet152_no_filled', 
                    sim_dir+'/ResNet152_no_lines', 
                    sim_dir+'/ResNet152_no_arrows'
                    ]
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
        print(ds_to_ignore)
        condition_names_and_ds = [x for x in names_and_ds if x[0] != ds_to_ignore]
        condition_data = test_model_val_auc_on_datastets(
            names_and_datasets=condition_names_and_ds,
            weights_dir=w_dir,
            condition=ds_to_ignore
            )
        results.extend(condition_data)
    return pd.DataFrame(results, columns=['Model', 'Condition', 'Testing data', 'AUC'])

# Test functions
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

def test_model_auc(
    ds_untrained, 
    weights_dir, 
    gap, 
    model_name, 
    condition
    ):
    """Tests 10 versions of a single model in a single condition using the area under the ROC curve. 
    Args:
        ds_untrained: data from the condition not trained on.
        weights_dir: directory of models weights to test.
        gap: wheather to use GAP. True or False.
        model_name: model name. String.
        condition: condition name. String.
    Returns:
        A list with test data: model_name, condition, area under the ROC curve.
    """
    
    # Define model architecture.
    base_model, model = make_ResNet50_sd_classifier(base_weights=None, base_trainable=True, gap=gap)
    
    # Compile: set metric to auc (default is area under the ROC curve).
    model.compile(
            optimizer=tf.keras.optimizers.Adam(0.0003),
            loss='binary_crossentropy',
            metrics=[tf.keras.metrics.AUC()])
    
    # Get list of weights from path.
    weights_list = [f for f in listdir(weights_dir) if f.endswith('.hdf5')]
    weights_paths = [join(weights_dir, f) for f in weights_list if isfile(join(weights_dir, f))]
    
    # Test each model.
    models_data = []
    for w_path in weights_paths:
        model.load_weights(w_path)
        metrics_untrained = model.evaluate(ds_untrained)
        models_data.append([model_name, condition, 'Untrained', metrics_untrained[1]])
    
    return models_data

def test_all_ds_auc(batch_size, ds_dir, sim_dir, autotune_settings):
    # Load same/different datasets.
    untrained_dss = []

    # Regular.
    regular_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/regular_test.tfrecords',
        is_training=False,
        process_img=True,
        autotune_settings=autotune_settings
        )
    untrained_dss.append(regular_ds)
    
    # Irregular.
    irregular_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/irregular_test.tfrecords',
        is_training=False,
        process_img=True,
        autotune_settings=autotune_settings
        )
    untrained_dss.append(irregular_ds)
    
    # Open.
    open_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/open_test.tfrecords',
        is_training=False,
        process_img=True,
        autotune_settings=autotune_settings
        )
    untrained_dss.append(open_ds)

    # Wider line.
    wider_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/wider_line_test.tfrecords',
        is_training=False,
        process_img=True,
        autotune_settings=autotune_settings
        )
    untrained_dss.append(wider_ds)
    
    # Scrambled.
    scrambled_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/scrambled_test.tfrecords',
        is_training=False,
        process_img=True,
        autotune_settings=autotune_settings
        )
    untrained_dss.append(scrambled_ds)
    
    # Random color.
    random_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/random_color_test.tfrecords',
        is_training=False,
        process_img=True,
        autotune_settings=autotune_settings
        )
    untrained_dss.append(random_ds)
    
    # Filled.
    filled_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/filled_test.tfrecords',
        is_training=False,
        process_img=True,
        autotune_settings=autotune_settings
        )
    untrained_dss.append(filled_ds)
    
    # Lines.
    lines_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/lines_test.tfrecords',
        is_training=False,
        process_img=True,
        autotune_settings=autotune_settings
        )
    untrained_dss.append(lines_ds)
    
    # Arrows.
    arrows_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/arrows_test.tfrecords',
        is_training=False,
        process_img=True,
        autotune_settings=autotune_settings
        )
    untrained_dss.append(arrows_ds)
    
    conditions = ['Regular', 'Irregular', 'Open', 'Wider Line', 'Scrambled', 'Random Color', 'Filled', 'Lines', 'Arrows']
    weights_dirs = [
                    sim_dir+'/ResNet152_no_regular', 
                    sim_dir+'/ResNet152_no_irregular', 
                    sim_dir+'/ResNet152_no_open',
                    sim_dir+'/ResNet152_no_wider', 
                    sim_dir+'/ResNet152_no_scrambled',
                    sim_dir+'/ResNet152_no_random',
                    sim_dir+'/ResNet152_no_filled', 
                    sim_dir+'/ResNet152_no_lines', 
                    sim_dir+'/ResNet152_no_arrows'
                    ]
    results = []
    for ds, condition, w_dir in zip(untrained_dss, conditions, weights_dirs):
        data = test_model_auc(
            ds_untrained=ds,
            weights_dir=w_dir,
            gap=True,
            model_name='ResNet152',
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

    # Train instances in each training condition.
    AUTO = tf.data.experimental.AUTOTUNE
    strategy = tf.distribute.get_strategy()
    train_in_all_datasets(
        strategy=strategy,
        autotune_settings=AUTO,
        ds_dir='data',
        sim_dir='/simulation_2',
        epochs_top=EPOCHS_TOP,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        n=10,
        lr=0.0001,
        batch_size=BATCH_SIZE
        )
    # Validate and save
    df = test_all_conditions_val_auc(
        autotune_settings=AUTO,
        batch_size=560, 
        ds_dir='data',
        sim_dir='simulation_2',
        )
    df.to_csv('simulation_2/sim_2_resnet152_val_auc.csv')

    # Test and save.
    df = test_all_ds_auc(
        autotune_settings=AUTO,
        batch_size=560,
        ds_dir='data',
        sim_dir='simulation_2'
        )
    df.to_csv('simulation_2/sim_2_resnet152_test_auc.csv')