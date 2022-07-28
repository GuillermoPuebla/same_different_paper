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
        raise ValueError('Unrecognized base weights argument!')
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

def get_master_dataset(autotune_settings, batch_size, ds_dir, process_img=True):
    """Builds dataset that samples each batch from one of the training datasets
    assiging a same-different sample weight of 1 only for the original
    condition and a relative-position sample weight 1 for all conitions."""
    
    # Make datasets and append if it is not the dont_include dataset.
    datasets = []
    
    # Original.
    ds_original = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/original_train.tfrecords',
        autotune_settings=autotune_settings,
        is_training=True,
        process_img=process_img,
        sd_sample_weight=True)
    datasets.append(ds_original)
    
    # Regular.
    ds_regular = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/regular_train.tfrecords',
        autotune_settings=autotune_settings,
        is_training=True,
        process_img=process_img,
        sd_sample_weight=False)
    datasets.append(ds_regular)
    
    # Irregular.
    ds_irregular = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/irregular_train.tfrecords',
        autotune_settings=autotune_settings,
        is_training=True,
        process_img=process_img,
        sd_sample_weight=False)
    datasets.append(ds_irregular)
    
    # Open.
    ds_open = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/open_train.tfrecords',
        autotune_settings=autotune_settings,
        is_training=True,
        process_img=process_img,
        sd_sample_weight=False)
    datasets.append(ds_open)

    # Wider line.
    ds_wider = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/wider_line_train.tfrecords',
        autotune_settings=autotune_settings,
        is_training=True,
        process_img=process_img,
        sd_sample_weight=False)
    datasets.append(ds_wider)

    # Scrambled.
    ds_scrambled = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/scrambled_train.tfrecords',
        autotune_settings=autotune_settings,
        is_training=True,
        process_img=process_img,
        sd_sample_weight=False)
    datasets.append(ds_scrambled)
    
    # Random color.
    ds_random = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/random_color_train.tfrecords',
        autotune_settings=autotune_settings,
        is_training=True,
        process_img=process_img,
        sd_sample_weight=False)
    datasets.append(ds_random)
    
    # Filled.
    ds_filled = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/filled_train.tfrecords',
        autotune_settings=autotune_settings,
        is_training=True,
        process_img=process_img,
        sd_sample_weight=False)
    datasets.append(ds_filled)
    
    # Lines.
    ds_lines = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/lines_train.tfrecords',
        autotune_settings=autotune_settings,
        is_training=True,
        process_img=process_img,
        sd_sample_weight=False)
    datasets.append(ds_lines)
    
    # Arrows.
    ds_arrows = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/arrows_train.tfrecords',
        autotune_settings=autotune_settings,
        is_training=True,
        process_img=process_img,
        sd_sample_weight=False)
    datasets.append(ds_arrows)
    
    # Oversample the original dataset (50%) because I'm samplig tasks (same-diff, rel-pos)
    choice_tensor = tf.constant(value=[0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0, 9], dtype=tf.int64)
    choice_dataset = tf.data.Dataset.from_tensor_slices(choice_tensor).repeat()
    
    return tf.data.Dataset.choose_from_datasets(datasets, choice_dataset)

# Training function
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
    gap=True):
    # Train n model instances
    with strategy.scope():
        for i in range(n):
            # Define model
            base_model, model = make_ResNet152_sd_classifier(
                base_weights=base_weights,
                base_trainable=False,
                gap=gap,
                secondary_outputs=True)
            # Compile
            model.compile(
                optimizer=tf.keras.optimizers.Adam(0.0003),
                loss={'sd': 'binary_crossentropy', 'rel_pos': 'binary_crossentropy'},
                metrics={'sd': 'binary_accuracy', 'rel_pos': 'binary_accuracy'}
                )
            # Pretrain classifier
            filename = save_name + '/' + model_name +'_run_' + str(i) + '_log.csv'
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
            # Unfreeze ResNet layers
            base_model.trainable = True
            # Re-compile
            model.compile(
                optimizer=tf.keras.optimizers.Adam(lr),
                loss={'sd': 'binary_crossentropy', 'rel_pos': 'binary_crossentropy'},
                metrics={'sd': 'binary_accuracy', 'rel_pos': 'binary_accuracy'}
                )
            # Train whole model
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

# Test functions
def test_model_auc(ds, weights_dir, model, model_name, condition):
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
        models_data.append([model_name, condition, 'Same-Different', metrics[3]])
        models_data.append([model_name, condition, 'Relative position', metrics[4]])
    return models_data

def test_models_all_ds_auc(
    strategy,
    autotune_settings,
    batch_size, 
    weights_dir, 
    ds_dir
    ):
    # Load same/different datasets
    datasets = []
    # Original
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/original_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    # Regular
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/regular_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    # Irregular
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/irregular_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    # Open
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/open_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    # Wider line
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/wider_line_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    # Scrambled
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/scrambled_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    # Random color
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/random_color_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    # Filled
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/filled_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    # Lines
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/lines_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    # Arrows
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/arrows_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    
    conditions = ['Original', 'Regular', 'Irregular', 'Open', 'Wider Line', 'Scrambled', 'Random Color', 'Filled', 'Lines', 'Arrows']
    
    # Define model architecture
    with strategy.scope():
        base_model, model = make_ResNet152_sd_classifier(
            base_weights='imagenet',
            base_trainable=True,
            gap=True,
            secondary_outputs=True)
        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
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
        
    return pd.DataFrame(results, columns=['Model', 'Condition', 'Task', 'AUC'])

# Validation functions
def validate_models_all_ds_auc(
    strategy,
    autotune_settings,
    batch_size, 
    weights_dir, 
    ds_dir
    ):
    # Load same/different datasets
    datasets = []
    # Original
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/original_val.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    # Regular
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/regular_val.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    # Irregular
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/irregular_val.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    # Open
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/open_val.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    # Wider line
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/wider_line_val.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    # Scrambled
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/scrambled_val.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    # Random color
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/random_color_val.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    # Filled
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/filled_val.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    # Lines
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/lines_val.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    # Arrows
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/arrows_val.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    
    conditions = ['Original', 'Regular', 'Irregular', 'Open', 'Wider Line', 'Scrambled', 'Random Color', 'Filled', 'Lines', 'Arrows']
    
    # Define model architecture
    with strategy.scope():
        base_model, model = make_ResNet152_sd_classifier(
            base_weights='imagenet',
            base_trainable=True,
            gap=True,
            secondary_outputs=True)
        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
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
        
    return pd.DataFrame(results, columns=['Model', 'Condition', 'Task', 'AUC'])

if __name__ == '__main__':
    # Training hyperparameters.
    EPOCHS_TOP = 5
    EPOCHS = 15
    BATCH_SIZE = 64
    STEPS_PER_EPOCH = 28000 // BATCH_SIZE
    VALIDATION_STEPS = 2800 // BATCH_SIZE

    # Train ResNet classifier
    AUTO = tf.data.experimental.AUTOTUNE
    strategy = tf.distribute.get_strategy()

    train_ds = get_master_dataset(
        autotune_settings=AUTO, 
        batch_size=BATCH_SIZE, 
        ds_dir='data', 
        process_img=True
        )
    fine_tune_model(
        strategy=strategy,
        train_ds=train_ds,
        val_ds=None,
        save_name='simulation_3/relative_position/ResNet152_instances',
        model_name='ResNet',
        epochs_top=EPOCHS_TOP,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=None,
        n=10,
        lr=0.0001,
        base_weights='imagenet',
        gap=True
        )
    # Test ResNet152 classifier
    df = test_models_all_ds_auc(
        strategy=strategy,
        autotune_settings=AUTO,
        batch_size=560, 
        weights_dir='simulation_3/relative_position/ResNet152_instances', 
        ds_dir='data'
        )
    df.to_csv('simulation_3/relative_position/sim_3_ResNet152_relpos_test_auc.csv')
    
    # Validation
    df = validate_models_all_ds_auc(
        strategy=strategy,
        autotune_settings=AUTO,
        batch_size=560, 
        weights_dir='simulation_3/relative_position/ResNet152_instances', 
        ds_dir='data'
        )
    df.to_csv('simulation_3/relative_position/sim_3_ResNet152_relpos_val_auc.csv')
    