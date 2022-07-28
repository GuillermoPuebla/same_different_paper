# Imports
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Input

# Model
def make_ResNet50_sd_classifier(
    base_weights='imagenet',
    base_trainable=False,
    gap=True,
    secondary_outputs=True
    ):
    # Define input tensor
    image = Input((128, 128, 3))
    # Base pre-trained model
    if base_weights == 'imagenet':
        base_model = ResNet152(weights='imagenet', include_top=False, input_tensor=image)
    elif base_weights is None:
        base_model = ResNet152(weights=None, include_top=False, input_tensor=image)
    elif base_weights == 'TU-Berlin':
        # Get weights from sketch classifier base.
        sketch_b, sketch_m = make_ResNet50_sketch_classifier(weights=None, trainable=True)
        sketch_m.load_weights('cogsci_sims/sketches/ResNet50_classifier_sketches_weights.10-2.92.hdf5')
        base_weights = []
        for layer in sketch_b.layers:
            base_weights.append(layer.get_weights())
        # Set weights of base model.
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
    predictions = Dense(1, activation='sigmoid', name='sd')(x)
    if secondary_outputs:
        rel_pos = Dense(1, activation='sigmoid', name='rel_pos')(x)
        # This is the model we will train
        model = Model(inputs=base_model.input, outputs=[predictions, rel_pos])
    else:
        # This is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)
    return base_model, model

# Dataset
def get_dataset(
    batch_size,
    tfrecord_dir,
    autotune_settings,
    is_training=True,
    process_img=True,
    sd_sample_weight=True,
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
            'relative_position': tf.io.FixedLenFeature([], tf.int64)}
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
            # Sample-wise center image
            mean = tf.reduce_mean(image)
            image -= mean
            # Sample-wise std normalization
            std = tf.math.reduce_std(image)
            image /= std
        # Cast relative position to int64
        rel_pos = example['relative_position']
        rel_pos = tf.cast(rel_pos, tf.int64)
        # Sample weights
        sd_w = tf.constant(1, dtype=tf.int64) if sd_sample_weight else tf.constant(0, dtype=tf.int64)
        rp_w = tf.constant(1, dtype=tf.int64)
        if relnet:
            question = tf.constant([1, 0], dtype=tf.int64) if sd_sample_weight else tf.constant([0, 1], dtype=tf.int64)
            rp_w = tf.constant(0, dtype=tf.int64) if sd_sample_weight else tf.constant(1, dtype=tf.int64)
            return (image, question), (label, rel_pos), (sd_w, rp_w)
        else:
            return image, (label, rel_pos), (sd_w, rp_w)
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

def get_master_dataset(
    autotune_settings, 
    batch_size,
    ds_dir,
    dont_include,
    process_img=True
    ):
    """Builds dataset that samples each batch from one of the training datasets
    assiging a same-different sample weight of 1 for all but the dont_include
    dataset and a relative-position sample weight 1 for all other datasets."""
    
    # Make datasets and append if it is not the dont_include dataset.
    datasets = []
    
    # Original
    original_sw_w = False if dont_include == 'Original' else True
    ds_original = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/original_train.tfrecords',
        autotune_settings=autotune_settings,
        is_training=True,
        process_img=process_img,
        sd_sample_weight=original_sw_w)
    datasets.append(ds_original)

    # Regular
    regular_sw_w = False if dont_include == 'Regular' else True
    ds_regular = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/regular_train.tfrecords',
        autotune_settings=autotune_settings,
        is_training=True,
        process_img=process_img,
        sd_sample_weight=regular_sw_w
        )
    datasets.append(ds_regular)

    # Irregular
    irregular_sw_w = False if dont_include == 'Irregular' else True
    ds_irregular = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/irregular_train.tfrecords',
        autotune_settings=autotune_settings,
        is_training=True,
        process_img=process_img,
        sd_sample_weight=irregular_sw_w
        )
    datasets.append(ds_irregular)

    # Open
    open_sw_w = False if dont_include == 'Open' else True
    ds_open = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/open_train.tfrecords',
        autotune_settings=autotune_settings,
        is_training=True,
        process_img=process_img,
        sd_sample_weight=open_sw_w
        )
    datasets.append(ds_open)

    # Wider line
    wider_sw_w = False if dont_include == 'Wider line' else True
    ds_wider = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/wider_line_train.tfrecords',
        autotune_settings=autotune_settings,
        is_training=True,
        process_img=process_img,
        sd_sample_weight=wider_sw_w
        )
    datasets.append(ds_wider)

    # Scrambled
    scrambled_sw_w = False if dont_include == 'Scrambled' else True
    ds_scrambled = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/scrambled_train.tfrecords',
        autotune_settings=autotune_settings,
        is_training=True,
        process_img=process_img,
        sd_sample_weight=scrambled_sw_w
        )
    datasets.append(ds_scrambled)

    # Random color
    random_sw_w = False if dont_include == 'Random color' else True
    ds_random = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/random_color_train.tfrecords',
        autotune_settings=autotune_settings,
        is_training=True,
        process_img=process_img,
        sd_sample_weight=random_sw_w
        )
    datasets.append(ds_random)

    # Filled
    filled_sw_w = False if dont_include == 'Filled' else True
    ds_filled = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/filled_train.tfrecords',
        autotune_settings=autotune_settings,
        is_training=True,
        process_img=process_img,
        sd_sample_weight=filled_sw_w
        )
    datasets.append(ds_filled)

    # Lines
    lines_sw_w = False if dont_include == 'Lines' else True
    ds_lines = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/lines_train.tfrecords',
        autotune_settings=autotune_settings,
        is_training=True,
        process_img=process_img,
        sd_sample_weight=lines_sw_w
        )
    datasets.append(ds_lines)

    # Arrows
    arrows_sw_w = False if dont_include == 'Arrows' else True
    ds_arrows = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/arrows_train.tfrecords',
        autotune_settings=autotune_settings,
        is_training=True,
        process_img=process_img,
        sd_sample_weight=arrows_sw_w
        )
    datasets.append(ds_arrows)
    
    # Note. No need to oversample the original dataset in simulation 4 as
    # the model is trained on the same/different task in all datasets except
    # on the one that is going to be tested.
    choice_dataset = tf.data.Dataset.range(len(datasets)).repeat()
    
    return tf.data.Dataset.choose_from_datasets(datasets, choice_dataset)

# Training functions
def fine_tune_model(
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
    lr=0.0001,
    base_weights='imagenet',
    gap=True
    ):
    with strategy.scope():
        for i in range(n):
            # Define model
            base_model, model = make_ResNet50_sd_classifier(
                base_weights=base_weights,
                base_trainable=False,
                gap=gap,
                secondary_outputs=True
                )
            # Compile
            model.compile(
                optimizer=tf.keras.optimizers.Adam(0.0003),
                loss={'sd': 'binary_crossentropy', 'rel_pos': 'binary_crossentropy'},
                metrics={'sd': 'binary_accuracy', 'rel_pos': 'binary_accuracy'}
                )
            # Train
            filename = save_name + model_name +'_run_' + str(i) + '_log.csv'
            history_logger = tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)
            model.fit(
                train_ds,
                epochs=epochs_top,
                verbose=0,
                steps_per_epoch=steps_per_epoch,
                validation_data=val_ds,
                validation_steps=validation_steps,
                callbacks=[history_logger]
                )
            # Unfreeze Resnet50
            base_model.trainable = True
            # Re-compile
            model.compile(
                optimizer=tf.keras.optimizers.Adam(lr),
                loss={'sd': 'binary_crossentropy', 'rel_pos': 'binary_crossentropy'},
                metrics={'sd': 'binary_accuracy', 'rel_pos': 'binary_accuracy'})
            # Train
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
            weights_name = save_name + model_name + '_instance_' + str(i) + '.hdf5'
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
    batch_size=64):
    # Define all master datasets
    no_irregular_ds = get_master_dataset(batch_size=batch_size, dont_include='Irregular', process_img=True, autotune_settings=autotune_settings, ds_dir=ds_dir)
    no_regular_ds = get_master_dataset(batch_size=batch_size, dont_include='Regular', process_img=True, autotune_settings=autotune_settings, ds_dir=ds_dir)
    no_open_ds = get_master_dataset(batch_size=batch_size, dont_include='Open', process_img=True, autotune_settings=autotune_settings, ds_dir=ds_dir)
    no_wider_ds = get_master_dataset(batch_size=batch_size, dont_include='Wider line', process_img=True, autotune_settings=autotune_settings, ds_dir=ds_dir)
    no_scrambled_ds = get_master_dataset(batch_size=batch_size, dont_include='Scrambled', process_img=True, autotune_settings=autotune_settings, ds_dir=ds_dir)
    no_random_ds = get_master_dataset(batch_size=batch_size, dont_include='Random color', process_img=True, autotune_settings=autotune_settings, ds_dir=ds_dir)
    no_filled_ds = get_master_dataset(batch_size=batch_size, dont_include='Filled', process_img=True, autotune_settings=autotune_settings, ds_dir=ds_dir)
    no_lines_ds = get_master_dataset(batch_size=batch_size, dont_include='Lines', process_img=True, autotune_settings=autotune_settings, ds_dir=ds_dir)
    no_arrows_ds = get_master_dataset(batch_size=batch_size, dont_include='Arrows', process_img=True, autotune_settings=autotune_settings, ds_dir=ds_dir)

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
        fine_tune_model(
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
            lr=lr,
            base_weights='imagenet',
            gap=True
            )
    return

# Test functions
def test_model_auc(
    ds, 
    weights_dir,
    model,
    model_name, 
    condition
    ):
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
        metrics = model.evaluate(ds, verbose=2)
        # model.metrics_names = ['loss', 'sd_loss', 'rel_pos_loss', 'sd_auc_1', 'rel_pos_auc_1']
        models_data.append([model_name, condition, 'Same-Different', metrics[3]])
        models_data.append([model_name, condition, 'Relative position', metrics[4]])
    
    return models_data

def test_models_all_ds_auc(
    strategy,
    autotune_settings,
    batch_size, 
    sim_dir, 
    ds_dir
    ):
    # Load same/different datasets.
    datasets = []
    
    # Regular
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/regular_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(ds)
    
    # Irregular
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/irregular_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(ds)
    
    # Open
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/open_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(ds)

    # Wider line
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/wider_line_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(ds)
    
    # Scrambled
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/scrambled_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(ds)
    
    # Random color
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/random_color_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(ds)
    
    # Filled
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/filled_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(ds)
    
    # Lines
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/lines_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(ds)
    
    # Arrows
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/arrows_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(ds)
    
    conditions = [
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
    with strategy.scope():
        # Define model architecture
        base_model, model = make_ResNet50_sd_classifier(base_weights=None, base_trainable=True, gap=True)
        # Compile
        model.compile(
                optimizer=tf.keras.optimizers.Adam(0.0003),
                loss='binary_crossentropy',
                metrics=[tf.keras.metrics.AUC()]
                )
        results = []
        for ds, condition, w_dir in zip(datasets, conditions, weights_dirs):
            # Fine-tune, imagenet, GAP.
            ft_im_gap = test_model_auc(
                ds=ds,
                weights_dir=w_dir,
                model=model,
                model_name='ResNet152',
                condition=condition
                )
            results.extend(ft_im_gap)
    
    return pd.DataFrame(results, columns=['Model', 'Condition', 'Task', 'AUC'])

# Validation functions
def test_model_val_auc(ds_val, ds_name, weights_dir, model, model_name, condition):
    """
    Tests 10 versions of a single model in a single condition using the area under the ROC curve. 
    Args:
        ds_val: single dataset with cases from the 'same' and 'different' classes.
        ds_name: name of the dataset being tested. String.
        weights_dir: directory of models weights to test.
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
        metrics_val = model.evaluate(ds_val, verbose=2)
        # model.metrics_names = ['loss', 'sd_loss', 'rel_pos_loss', 'sd_auc_1', 'rel_pos_auc_1']
        models_data.append([model_name, condition, ds_name, 'Same-Different', metrics_val[3]])
        models_data.append([model_name, condition, ds_name, 'Relative position', metrics_val[4]])

    return models_data

def test_model_val_auc_on_datastets(
    names_and_datasets,
    weights_dir,
    condition,
    model,
    model_name='ResNet152'
    ):
    """
    Test 10 models in weights_dir on each dataset in names_and_datasets.
    Args:
        names_and_datasets: list of tuples (name, dataset).
        weights_dir: directory with the weights of 10 runs of a model.
        condition: name of the dataset the model was not trained on.
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
            model_name='ResNet152', 
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

    with strategy.scope():
        # Define model
        base_model, model = make_ResNet50_sd_classifier(base_weights=None, base_trainable=True, gap=True)
        # Compile: set metric to auc (default is area under the ROC curve)
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
            print(ds_to_ignore)
            # condition_names_and_ds = [x for x in names_and_ds if x[0] != ds_to_ignore]
            condition_data = test_model_val_auc_on_datastets(
                names_and_datasets=names_and_ds,
                weights_dir=w_dir,
                condition=ds_to_ignore,
                model=model,
                model_name='ResNet152'
                )
            results.extend(condition_data)
    return pd.DataFrame(results, columns=['Model', 'Condition', 'Testing data', 'Task', 'AUC'])

if __name__ == '__main__':
    # Training hyperparameters.
    EPOCHS_TOP = 5
    EPOCHS = 15
    BATCH_SIZE = 64
    STEPS_PER_EPOCH = 28000 // BATCH_SIZE
    VALIDATION_STEPS = 2800 // BATCH_SIZE

    # Train ResNet152 classifier
    AUTO = tf.data.experimental.AUTOTUNE
    strategy = tf.distribute.get_strategy()

    train_in_all_datasets(
        strategy=strategy,
        autotune_settings=AUTO,
        ds_dir='data',
        sim_dir='simulation_4',
        epochs_top=EPOCHS_TOP,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=None,
        n=10,
        lr=0.0001,
        batch_size=BATCH_SIZE
        )
    # Test
    df = test_models_all_ds_auc(
        strategy=strategy,
        autotune_settings=AUTO,
        batch_size=560, 
        sim_dir='simulation_4', 
        ds_dir='data'
        )
    df.to_csv('simulation_4/sim_4_ResNet152_relpos_test_auc.csv')

    # Validation
    df = test_all_conditions_val_auc(
        strategy=strategy,
        autotune_settings=AUTO,
        batch_size=560, 
        ds_dir='data',
        sim_dir='simulation_4',
        relnet=False
        )
    df.to_csv('simulation_4/sim_4_ResNet152_relpos_val_auc.csv')
    