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
def make_ResNet152_sd_classifier(
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
    else:
        raise ValueError('unrecognized base weights!')
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
            base_model, model = make_ResNet152_sd_classifier(
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
            # Unfreeze ResNet152
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

def extra_train_model(
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
    base_weights='imagenet',
    gap=True
    ):
    with strategy.scope():
        for i in range(n):
            # Define model
            base_model, model = make_ResNet152_sd_classifier(
                base_weights=base_weights,
                base_trainable=True,
                gap=gap,
                secondary_outputs=True
                )
            # Load weights
            weights_name = save_name + model_name + '_instance_' + str(i) + '.hdf5'
            model.load_weights(weights_name)
            # Compile
            model.compile(
                optimizer=tf.keras.optimizers.Adam(lr),
                loss={'sd': 'binary_crossentropy', 'rel_pos': 'binary_crossentropy'},
                metrics={'sd': 'binary_accuracy', 'rel_pos': 'binary_accuracy'})
            # Train
            filename = save_name + model_name +'_run_' + str(i) + '_log.csv'
            history_logger = tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)
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

def train_in_master_ds(
    strategy,
    autotune_settings,
    ds_dir,
    save_name,
    epochs_top,
    epochs,
    steps_per_epoch,
    validation_steps,
    n=10,
    lr=0.0001,
    batch_size=64
    ):
    # Define all master datasets
    all_dss = get_master_dataset(autotune_settings=autotune_settings, batch_size=batch_size, ds_dir=ds_dir, dont_include=None, process_img=True)

    # Train model in master dataset 10 times.    
    fine_tune_model(
        strategy=strategy,
        train_ds=all_dss,
        val_ds=None,
        save_name=save_name,
        model_name='ResNet152',
        epochs_top=epochs_top,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=None,
        n=10,
        lr=0.0001,
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
        fname = w_path.split('/')[-1]
        model_run = fname.split('_')[-1]
        model_run = model_run.split('.')[0]
        model.load_weights(w_path)
        metrics = model.evaluate(ds)
        # model.metrics_names = ['loss', 'sd_loss', 'rel_pos_loss', 'sd_auc_1', 'rel_pos_auc_1']
        models_data.append([model_name, model_run, condition, 'Same-Different', metrics[3]])
        models_data.append([model_name, model_run, condition, 'Relative position', metrics[4]])
    return models_data

def test_models_new_dss_auc(    
    strategy,
    autotune_settings,
    batch_size, 
    weights_dir, 
    ds_dir
    ):
    # Load same/different datasets.
    datasets = []
    
    # Original
    original_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/original_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(original_ds)
    
    # Regular
    regular_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/regular_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(regular_ds)
    
    # Irregular
    irregular_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/irregular_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(irregular_ds)
    
    # Open
    open_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/open_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(open_ds)

    # Wider line
    wider_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/wider_line_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(wider_ds)
    
    # Scrambled
    scrambled_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/scrambled_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(scrambled_ds)
    
    # Random color
    random_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/random_color_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(random_ds)
    
    # Filled
    filled_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/filled_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(filled_ds)
    
    # Lines
    lines_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/lines_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(lines_ds)
    
    # Arrows
    arrows_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/arrows_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(arrows_ds)

    # Rectangles
    rectangles_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/rectangles_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(rectangles_ds)
    
    # Straight Lines
    straight_lines = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/straight_lines_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True)
    datasets.append(straight_lines)
    
    # Connected squares
    conected_squares_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/connected_squares_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(conected_squares_ds)
    
    # Circles
    circles_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/connected_circles_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(circles_ds)
    
    conditions = [
                  'Original', 
                  'Regular', 
                  'Irregular', 
                  'Open', 
                  'Wider Line', 
                  'Scrambled', 
                  'Random Color',
                  'Filled', 
                  'Lines', 
                  'Arrows',
                  'Rectangles', 
                  'Straight Lines', 
                  'Conected Squares', 
                  'Circles'
                  ]    
    with strategy.scope():
        # Define model
        base_model, model = make_ResNet152_sd_classifier(
            base_weights='imagenet',
            base_trainable=True,
            gap=True,
            secondary_outputs=True
            )
        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.0003),
            loss={'sd': 'binary_crossentropy', 'rel_pos': 'binary_crossentropy'},
            metrics=[tf.keras.metrics.AUC()]
            )
    results = []
    for ds, condition in zip(datasets, conditions):
        # Fine-tune, imagenet, GAP.
        ft_im_gap = test_model_auc(
            ds=ds,
            weights_dir=weights_dir,
            model=model,
            model_name='ResNet152',
            condition=condition
            )
        results.extend(ft_im_gap)
        
    return pd.DataFrame(results, columns=['Model', 'Run', 'Condition', 'Task', 'AUC'])

# Validation functions
def validate_models_trained_dss_auc(    
    strategy,
    autotune_settings,
    batch_size, 
    weights_dir, 
    ds_dir
    ):
    # Load same/different datasets.
    datasets = []
    
    # Original
    original_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/original_val.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(original_ds)
    
    # Regular
    regular_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/regular_val.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(regular_ds)
    
    # Irregular
    irregular_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/irregular_val.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(irregular_ds)
    
    # Open
    open_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/open_val.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(open_ds)

    # Wider line
    wider_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/wider_line_val.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(wider_ds)
    
    # Scrambled
    scrambled_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/scrambled_val.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(scrambled_ds)
    
    # Random color
    random_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/random_color_val.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(random_ds)
    
    # Filled
    filled_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/filled_val.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(filled_ds)
    
    # Lines
    lines_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/lines_val.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(lines_ds)
    
    # Arrows
    arrows_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/arrows_val.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(arrows_ds)

    conditions = [
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
    with strategy.scope():
        # Define model
        base_model, model = make_ResNet152_sd_classifier(
            base_weights='imagenet',
            base_trainable=True,
            gap=True,
            secondary_outputs=True
            )
        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.0003),
            loss={'sd': 'binary_crossentropy', 'rel_pos': 'binary_crossentropy'},
            metrics=[tf.keras.metrics.AUC()]
            )
        results = []
        for ds, condition in zip(datasets, conditions):
            # Fine-tune, imagenet, GAP.
            ft_im_gap = test_model_auc(
                ds=ds,
                weights_dir=weights_dir,
                model=model,
                model_name='ResNet152',
                condition=condition
                )
            results.extend(ft_im_gap)
        
    return pd.DataFrame(results, columns=['Model', 'Run', 'Condition', 'Task', 'AUC'])

# Density estimate ds building
def get_model_predictions(
    ds, 
    weights_path,
    steps,
    model, 
    model_name,
    run,
    condition,
    batch_size
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
    density_data = []
    model.load_weights(weights_path)
    for x, y, z in ds.take(steps):
        y_sd, y_rp = model.predict_on_batch(x)
        p_same = np.squeeze(y_sd)
        y_np = y[0].numpy()
        for i in range(batch_size):
            density_data.append([condition, model_name, run, y_np[i], p_same[i]])
    
    return density_data

def get_model_predictions_all_ds(
    strategy,
    autotune_settings,
    batch_size,
    steps,
    weights_path,
    run,
    ds_dir
    ):
    # Load same/different datasets.
    datasets = []
    
    # Original
    original_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/original_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(original_ds)
    
    # Regular
    regular_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/regular_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(regular_ds)
    
    # Irregular
    irregular_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/irregular_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(irregular_ds)
    
    # Open
    open_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/open_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(open_ds)

    # Wider line
    wider_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/wider_line_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(wider_ds)
    
    # Scrambled
    scrambled_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/scrambled_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(scrambled_ds)
    
    # Random color
    random_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/random_color_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(random_ds)
    
    # Filled
    filled_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/filled_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(filled_ds)
    
    # Lines
    lines_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/lines_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(lines_ds)
    
    # Arrows
    arrows_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/arrows_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(arrows_ds)

    # Rectangles
    rectangles_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/rectangles_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(rectangles_ds)
    
    # Straight Lines
    straight_lines = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/straight_lines_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True)
    datasets.append(straight_lines)
    
    # Connected squares
    conected_squares_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/connected_squares_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(conected_squares_ds)
    
    # Circles
    circles_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/connected_circles_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True,
        sd_sample_weight=True
        )
    datasets.append(circles_ds)
    
    conditions = [
                  'Original', 
                  'Regular', 
                  'Irregular', 
                  'Open', 
                  'Wider Line', 
                  'Scrambled', 
                  'Random Color',
                  'Filled', 
                  'Lines', 
                  'Arrows',
                  'Rectangles', 
                  'Straight Lines', 
                  'Conected Squares', 
                  'Circles'
                  ]
    with strategy.scope():
        # Define model
        base_model, model = make_ResNet152_sd_classifier(
            base_weights='imagenet',
            base_trainable=True,
            gap=True,
            secondary_outputs=True
            )
        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.0003),
            loss={'sd': 'binary_crossentropy', 'rel_pos': 'binary_crossentropy'},
            metrics={'sd': 'binary_accuracy', 'rel_pos': 'binary_accuracy'}
            )
    results = []
    for ds, condition in zip(datasets, conditions):
        density_data = get_model_predictions(
            ds=ds, 
            weights_path=weights_path,
            steps=steps,
            model=model, 
            model_name='ResNet152',
            run=run,
            condition=condition,
            batch_size=batch_size
            )
        results.extend(density_data)

    return pd.DataFrame(results, columns=['Condition', 'Model', 'Run', 'y_label', 'p_same'])

if __name__ == '__main__':
    # Training hyperparameters.
    EPOCHS_TOP = 5
    EPOCHS = 20
    BATCH_SIZE = 64
    STEPS_PER_EPOCH = 28000 // BATCH_SIZE
    VALIDATION_STEPS = 2800 // BATCH_SIZE
    TEST_STEPS = 11200 // BATCH_SIZE

    # Train
    AUTO = tf.data.experimental.AUTOTUNE
    strategy = tf.distribute.get_strategy()

    # Train
    train_in_master_ds(
        strategy=strategy,
        autotune_settings=AUTO,
        ds_dir='data',
        save_name='simulation_5/ResNet152_instances/',
        epochs_top=EPOCHS_TOP,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=None,
        n=10,
        lr=0.0001,
        batch_size=64
        )
    # Test
    df = test_models_new_dss_auc(
        strategy=strategy,
        autotune_settings=AUTO,
        batch_size=1120, 
        weights_dir='simulation_5/ResNet152_instances/', 
        ds_dir='data'
        )
    df.to_csv('simulation_5/sim_5_ResNet152_test_auc.csv')
    # Validate
    df = validate_models_trained_dss_auc(    
        strategy=strategy,
        autotune_settings=AUTO,
        batch_size=1120, 
        weights_dir='simulation_5/ResNet152_instances/', 
        ds_dir='data'
        )
    df.to_csv('simulation_5/sim_5_ResNet152_val_auc.csv')
    # Get predictions for density plot
    df = get_model_predictions_all_ds(
        strategy=strategy,
        autotune_settings=AUTO,
        batch_size=BATCH_SIZE,
        steps=TEST_STEPS,
        weights_path='simulation_5/ResNet152_instances/ResNet152_instance_0.hdf5',
        run=6,
        ds_dir='data'
        )
    df.to_csv('simulation_5/sim_5_ResNet152_density.csv')