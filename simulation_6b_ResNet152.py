# Imports
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Concatenate, Input

# Model
def make_ResNet50_2chanels(base_trainable=False):
    # Define input tensors
    image1 = Input((128, 128, 3))
    image2 = Input((128, 128, 3))
    # Base pre-trained model
    base_model = ResNet152(
        weights='imagenet', 
        include_top=False, 
        input_shape=(128, 128, 3)
        )
    # Freeze the if necesarry base_model
    base_model.trainable = base_trainable
    # Add a global spatial average pooling layer
    x1 = base_model(image1)
    x1 = GlobalAveragePooling2D()(x1)
    x2 = base_model(image2)
    x2 = GlobalAveragePooling2D()(x2)
    x = Concatenate()([x1, x2])
    # Add a two fully-connected layer
    x = Dense(2048, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    # Add logistic layer
    predictions = Dense(1, activation='sigmoid', name='sd')(x)
    # This is the model we will train
    model = Model(inputs=[image1, image2], outputs=predictions)
    return base_model, model

# Dataset
def get_dataset(
    batch_size, 
    tfrecord_dir,
    autotune_settings,
    is_training=True, 
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
            'image1_raw': tf.io.FixedLenFeature([], tf.string),
            'image2_raw': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
            }
        # Parse example
        example = tf.io.parse_single_example(
            serialized_example, 
            feature_description
            )
        # Get images
        image1 = tf.io.decode_png(example['image1_raw'], channels=3)
        image2 = tf.io.decode_png(example['image2_raw'], channels=3)
        # Ensure shape dimensions are constant
        image1 = tf.reshape(image1, [128, 128, 3])
        image2 = tf.reshape(image2, [128, 128, 3])
        # Preprocess images
        if process_img:
            image1 = tf.cast(image1, tf.float64)
            image1 /= 255.0
            # Sample-wise center image.
            mean1 = tf.reduce_mean(image1)
            image1 -= mean1
            # Sample-wise std normalization.
            std1 = tf.math.reduce_std(image1)
            image1 /= std1
            image2 = tf.cast(image2, tf.float64)
            image2 /= 255.0
            # Sample-wise center image.
            mean2 = tf.reduce_mean(image2)
            image2 -= mean2
            # Sample-wise std normalization.
            std2 = tf.math.reduce_std(image2)
            image2 /= std2
        # Cast label to int64
        label = example['label']
        label = tf.cast(label, tf.int64)
        return (image1, image2), label
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

# Train functions
def fine_tune_model(
    strategy,
    train_ds,
    save_name,
    model_name,
    epochs_top,
    epochs,
    steps_per_epoch,
    val_ds=None,
    validation_steps=None,
    n=10,
    lr=0.0001
    ):    
    with strategy.scope():
        for i in range(n):
            # Define model
            base_model, model = make_ResNet50_2chanels(base_trainable=False)
            # Compile
            model.compile(
                optimizer=tf.keras.optimizers.Adam(0.0003),
                loss={'sd': 'binary_crossentropy'},
                metrics={'sd': 'binary_accuracy'}
                )
            # Train
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
            # Unfreeze Resnet50
            base_model.trainable = True
            # Re-compile
            model.compile(
                optimizer=tf.keras.optimizers.Adam(lr),
                loss={'sd': 'binary_crossentropy'},
                metrics={'sd': 'binary_accuracy'}
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
            weights_name = save_name + '/' + model_name + '_instance_' + str(i) + '.hdf5'
            model.save_weights(weights_name)
    return

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
    
    # Regular
    ds_regular = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/regular_two_chanels_train.tfrecords',
        autotune_settings=autotune_settings,
        is_training=True,
        process_img=process_img        
        )
    if dont_include != 'Regular':
        datasets.append(ds_regular)
    
    # Irregular
    ds_irregular = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/irregular_two_chanels_train.tfrecords',
        autotune_settings=autotune_settings,
        is_training=True,
        process_img=process_img
        )
    if dont_include != 'Irregular':
        datasets.append(ds_irregular)
    
    # Open
    ds_open = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/open_two_chanels_train.tfrecords',
        autotune_settings=autotune_settings,
        is_training=True,
        process_img=process_img
        )
    if dont_include != 'Open':
        datasets.append(ds_open)

    # Wider line
    ds_wider = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/wider_line_two_chanels_train.tfrecords',
        autotune_settings=autotune_settings,
        is_training=True,
        process_img=process_img
        )
    if dont_include != 'Wider line':
        datasets.append(ds_wider)

    # Scrambled
    ds_scrambled = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/scrambled_two_chanels_train.tfrecords',
        autotune_settings=autotune_settings,
        is_training=True,
        process_img=process_img
        )
    if dont_include != 'Scrambled':
        datasets.append(ds_scrambled)
    
    # Random color
    ds_random = get_dataset(
    batch_size=batch_size,
    tfrecord_dir=f'{ds_dir}/random_color_two_chanels_train.tfrecords',
    autotune_settings=autotune_settings,
    is_training=True,
    process_img=process_img
    )
    if dont_include != 'Random color':
        datasets.append(ds_random)
    
    # Filled
    ds_filled = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/filled_two_chanels_train.tfrecords',
        autotune_settings=autotune_settings,
        is_training=True,
        process_img=process_img
        )
    if dont_include != 'Filled':
        datasets.append(ds_filled)
    
    # Lines
    ds_lines = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/lines_two_chanels_train.tfrecords',
        autotune_settings=autotune_settings,
        is_training=True,
        process_img=process_img
        )
    if dont_include != 'Lines':
        datasets.append(ds_lines)
    
    # Arrows
    ds_arrows = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/arrows_two_chanels_train.tfrecords',
        autotune_settings=autotune_settings,
        is_training=True,
        process_img=process_img
        )
    if dont_include != 'Arrows':
        datasets.append(ds_arrows)
    
    # Define uniform dataset sampling
    choice_dataset = tf.data.Dataset.range(len(datasets)).repeat()
    
    return tf.data.Dataset.choose_from_datasets(datasets, choice_dataset)

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
        metrics = model.evaluate(ds)
        models_data.append([model_name, condition, metrics[1]])
    
    return models_data

def test_all_models_all_conditions_auc(
    strategy,
    autotune_settings,
    batch_size, 
    weights_dir, 
    ds_dir
    ):
    # Load same/different datasets.
    datasets = []
    
    # Irregular
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/irregular_two_chanels_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    
    # Regular
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/regular_two_chanels_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    
    # Open
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/open_two_chanels_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True
        )
    datasets.append(ds)

    # Wider line
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/wider_line_two_chanels_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    
    # Scrambled
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/scrambled_two_chanels_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    
    # Random color
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/random_color_two_chanels_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    
    # Filled
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/filled_two_chanels_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    
    # Lines
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/lines_two_chanels_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    
    # Arrows
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/arrows_two_chanels_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True
        )
    datasets.append(ds)

    # Rectangles
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/rectangles_two_chanels_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    
    # Straight lines
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/straight_lines_two_chanels_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    
    # Connected squares
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/connected_squares_two_chanels_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    
    # Connected circles
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/connected_circles_two_chanels_test.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    
    conditions = [
                  'Irregular',
                  'Regular',
                  'Open',
                  'Wider Line',
                  'Scrambled',
                  'Random Color',
                  'Filled', 
                  'Lines',
                  'Arrows',
                  'Rectangles',
                  'Straight Lines',
                  'Connected Squares',
                  'Connected Circles'
                  ]
    with strategy.scope():
        # Define model architecture
        base_model, model = make_ResNet50_2chanels(base_trainable=True)
        # Compile
        model.compile(
                optimizer=tf.keras.optimizers.Adam(0.0003),
                loss='binary_crossentropy',
                metrics=[tf.keras.metrics.AUC()]
                )
        results = []
        for ds, condition in zip(datasets, conditions):
            data = test_model_auc(
                ds=ds, 
                weights_dir=weights_dir,
                model=model,
                model_name='Siamese_ResNet152', 
                condition=condition
                )
            results.extend(data)
    # Make pandas dataframe and return
    return pd.DataFrame(results, columns=['Model', 'Condition', 'AUC'])

# Validation functions
def validate_all_models_all_conditions_auc(
    strategy,
    autotune_settings,
    batch_size, 
    weights_dir, 
    ds_dir
    ):
    # Load same/different datasets.
    datasets = []

    # Irregular
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/irregular_two_chanels_val.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    
    # Regular
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/regular_two_chanels_val.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    
    # Open
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/open_two_chanels_val.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True
        )
    datasets.append(ds)

    # Wider line
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/wider_line_two_chanels_val.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    
    # Scrambled
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/scrambled_two_chanels_val.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    
    # Random color
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/random_color_two_chanels_val.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    
    # Filled
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/filled_two_chanels_val.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    
    # Lines
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/lines_two_chanels_val.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    
    # Arrows
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=f'{ds_dir}/arrows_two_chanels_val.tfrecords',
        autotune_settings=autotune_settings,
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    
    conditions = [
        'Irregular',
        'Regular', 
        'Open', 
        'Wider Line', 
        'Scrambled',
        'Random Color', 
        'Filled', 
        'Lines', 
        'Arrows'
        ]
    with strategy.scope():
        # Define model architecture
        base_model, model = make_ResNet50_2chanels(base_trainable=True)
        # Compile
        model.compile(
                optimizer=tf.keras.optimizers.Adam(0.0003),
                loss='binary_crossentropy',
                metrics=[tf.keras.metrics.AUC()]
                )
        results = []
        for ds, condition in zip(datasets, conditions):
            data = test_model_auc(
                ds=ds, 
                weights_dir=weights_dir,
                model=model,
                model_name='Siamese_ResNet152', 
                condition=condition
                )
            results.extend(data)
    # Make pandas dataframe and return
    return pd.DataFrame(results, columns=['Model', 'Condition', 'AUC'])


if __name__ == '__main__':
    # Training hyperparameters
    EPOCHS_TOP = 4
    EPOCHS = 6
    BATCH_SIZE = 64
    STEPS_PER_EPOCH = 28000 // BATCH_SIZE

    # Train
    AUTO = tf.data.experimental.AUTOTUNE
    strategy = tf.distribute.get_strategy()

    # Training data
    train_ds = get_master_dataset(
        autotune_settings=AUTO,
        batch_size=BATCH_SIZE,
        ds_dir='data',
        dont_include=None,
        process_img=True
        )
    val_ds = get_dataset(
        batch_size=BATCH_SIZE,
        tfrecord_dir='data/irregular_two_chanels_val.tfrecords',
        autotune_settings=AUTO,
        is_training=False,
        process_img=True
        )
    # Train
    fine_tune_model(
        strategy=strategy,
        train_ds=train_ds,
        save_name='simulation_6b/siamese_ResNet152_instances',
        model_name='siamese',
        epochs_top=EPOCHS_TOP,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        val_ds=val_ds,
        validation_steps=None,
        n=10,
        lr=0.0001
        )
    # Test
    df = test_all_models_all_conditions_auc(
        strategy=strategy,
        autotune_settings=AUTO,
        batch_size=560, 
        weights_dir='simulation_6b/siamese_ResNet152_instances', 
        ds_dir='data'
        )
    df.to_csv('simulation_6b/sim_6b_siamese_ResNet152_test_auc.csv')

    # Validate
    df = validate_all_models_all_conditions_auc(
        strategy=strategy,
        autotune_settings=AUTO,
        batch_size=560, 
        weights_dir='simulation_6b/siamese_ResNet152_instances', 
        ds_dir='data'
        )
    df.to_csv('simulation_6b/sim_6b_siamese_ResNet152_val_auc.csv')