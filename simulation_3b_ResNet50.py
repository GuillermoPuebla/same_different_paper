# Imports
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Input

# Model builder
def make_ResNet50_sd_classifier_with_ds_label(
    base_weights='imagenet',
    base_trainable=False,
    gap=True,
    secondary_outputs=True
    ):
    # Define input tensor.
    image = Input((128, 128, 3))
    
    # Base pre-trained model.
    if base_weights == 'imagenet':
        base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=image)
    elif base_weights is None:
        base_model = ResNet50(weights=None, include_top=False, input_tensor=image)
    elif base_weights == 'TU-Berlin':
        # Get weights from sketch classifier base.
        sketch_b, sketch_m = make_ResNet50_sketch_classifier(weights=None, trainable=True)
        sketch_m.load_weights('cogsci_sims/sketches/ResNet50_classifier_sketches_weights.10-2.92.hdf5')
        base_weights = []
        for layer in sketch_b.layers:
            base_weights.append(layer.get_weights())
        # Set weights of base model.
        base_model = ResNet50(weights=None, include_top=False, input_tensor=image)
        i = 0
        for layer in base_model.layers:
            layer.set_weights(base_weights[i])
            i += 1
            
    # Freeze the if necesarry base_model.
    base_model.trainable = base_trainable

    # Add a global spatial average pooling layer.
    x = base_model.output
    if gap:
        x = GlobalAveragePooling2D()(x)
    else:
        x = Flatten()(x)

    # Add a fully-connected layer.
    x = Dense(1024, activation='relu')(x)

    # Add logistic layer.
    predictions = Dense(1, activation='sigmoid', name='sd')(x)
    
    if secondary_outputs:
        ds_class = Dense(10, activation='sigmoid', name='ds_class')(x)
        # This is the model we will train.
        model = Model(inputs=base_model.input, outputs=[predictions, ds_class])
    else:
        # This is the model we will train.
        model = Model(inputs=base_model.input, outputs=predictions)

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
    process_img=True
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
        tc_sample_weight=True
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
        tc_sample_weight=True
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
        tc_sample_weight=True
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
        tc_sample_weight=True
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
        tc_sample_weight=True
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
        tc_sample_weight=True
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
        tc_sample_weight=True
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
        tc_sample_weight=True
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
        tc_sample_weight=True
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
        tc_sample_weight=True
        )
    tc_datasets.append(ds_arrows_tc)
    # no 'weights' argument gives you uniform distribution
    task_classification_dataset = tf.data.Dataset.sample_from_datasets(
        tc_datasets,
        weights=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        )
    task_classification_dataset = task_classification_dataset.batch(batch_size).prefetch(autotune_settings)
    return task_classification_dataset

def get_master_ds_with_ds_labels(autotune_settings, batch_size, ds_dir, process_img=True):
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
        tc_sample_weight=False
        )    
    # All others (task-classification)
    ds_task_classification = get_task_classification_dataset(
        autotune_settings=AUTO,
        batch_size=batch_size, 
        ds_dir=ds_dir, 
        ds_type='train', 
        process_img=process_img
        )
    all_datasets = [ds_original_sd, ds_task_classification]
    # Sample task (same-diff, task classification)
    choice_tensor = tf.constant(value=[0, 1], dtype=tf.int64)
    choice_dataset = tf.data.Dataset.from_tensor_slices(choice_tensor).repeat()
    
    return tf.data.Dataset.choose_from_datasets(all_datasets, choice_dataset)

# Training
def fine_tune_model_with_ds_label(
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
            base_model, model = make_ResNet50_sd_classifier_with_ds_label(
                base_weights=base_weights,
                base_trainable=False,
                gap=gap,
                secondary_outputs=True
                )
            # Compile
            model.compile(
                optimizer=tf.keras.optimizers.Adam(0.0003),
                loss={'sd': 'binary_crossentropy', 'ds_class': 'categorical_crossentropy'},
                metrics={'sd': 'binary_accuracy', 'ds_class': 'categorical_accuracy'}
                )
            # Train on same-different and task classification
            filename = save_name + '/' + model_name +'_run_' + str(i) + '_log.csv'
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
            # Unfreeze ResNet layers
            base_model.trainable = True
            # Re-compile
            model.compile(
                optimizer=tf.keras.optimizers.Adam(lr),
                loss={'sd': 'binary_crossentropy', 'ds_class': 'categorical_crossentropy'},
                metrics={'sd': 'binary_accuracy', 'ds_class': 'categorical_accuracy'}
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

# Testing
def test_model_auc_with_ds_label(model, ds, weights_dir, model_name, condition):
    """Tests 10 versions of a single model in a single condition using the area under the ROC curve. 
    Args:
        ds: dataset with cases from the 'same' and 'different' classes.
        weights_dir: directory of models weights to test.
        model_name: model name. String.
        condition: condition (dataset) name. String.
    Returns:
        A list with test data: model_name, condition, area under the ROC curve.
    """
    
    # Get list of weights from path
    weights_list = [f for f in listdir(weights_dir) if f.endswith('.hdf5')]
    weights_paths = [join(weights_dir, f) for f in weights_list if isfile(join(weights_dir, f))]
    
    # Test each model.
    models_data = []
    for w_path in weights_paths:
        model.load_weights(w_path)
        metrics = model.evaluate(ds)
        # model.metrics_names = ['loss', 'sd_loss', 'ds_class_loss', 'sd_auc_1', 'ds_class_auc_1']
        models_data.append([model_name, condition, 'Same-Different', metrics[3]])
        models_data.append([model_name, condition, 'Condition classification', metrics[4]])
    
    return models_data

def test_models_all_ds_auc_with_ds_label(
    strategy,
    autotune_settings,
    batch_size, 
    weights_dir, 
    ds_dir
    ):
    # Load same/different datasets.
    datasets = []
    
    # Original
    ds = get_ds_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/original_test.tfrecords',
        autotune_settings=autotune_settings,
        ds_name='original',
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    # Regular
    ds = get_ds_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/regular_test.tfrecords',
        autotune_settings=autotune_settings,
        ds_name='regular',
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    # Irregular
    ds = get_ds_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/irregular_test.tfrecords',
        autotune_settings=autotune_settings,
        ds_name='irregular',
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    # Open
    ds = get_ds_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/open_test.tfrecords',
        autotune_settings=autotune_settings,
        ds_name='open',
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    # Wider line
    ds = get_ds_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/wider_line_test.tfrecords',
        autotune_settings=autotune_settings,
        ds_name='wider',
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    # Scrambled
    ds = get_ds_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/scrambled_test.tfrecords',
        autotune_settings=autotune_settings,
        ds_name='scrambled',
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    # Random color
    ds = get_ds_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/random_color_test.tfrecords',
        autotune_settings=autotune_settings,
        ds_name='random',
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    # Filled
    ds = get_ds_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/filled_test.tfrecords',
        autotune_settings=autotune_settings,
        ds_name='filled',
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    # Lines
    ds = get_ds_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/lines_test.tfrecords',
        autotune_settings=autotune_settings,
        ds_name='lines',
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    # Arrows
    ds = get_ds_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/arrows_test.tfrecords',
        autotune_settings=autotune_settings,
        ds_name='arrows',
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    
    conditions = ['Original', 'Regular', 'Irregular', 'Open', 'Wider Line', 'Scrambled', 'Random Color', 'Filled', 'Lines', 'Arrows']
    
    # Define model architecture
    with strategy.scope():
        base_model, model = make_ResNet50_sd_classifier_with_ds_label(base_weights=None, base_trainable=True, gap=True)
        
        # Compile: set metric to auc (default is area under the ROC curve)
        model.compile(
                optimizer=tf.keras.optimizers.Adam(0.0003),
                loss={'sd': 'binary_crossentropy', 'ds_class': 'categorical_crossentropy'},
                metrics=[tf.keras.metrics.AUC()]
                )
        results = []
        for ds, condition in zip(datasets, conditions):
            # Fine-tune, imagenet, GAP.
            ft_im_gap = test_model_auc_with_ds_label(
                model=model,
                ds=ds,
                weights_dir=weights_dir,
                model_name='ResNet',
                condition=condition)
            results.extend(ft_im_gap)
        
    return pd.DataFrame(results, columns=['Model', 'Condition', 'Task', 'AUC'])

# Validation
def validate_models_all_ds_auc_with_ds_label(
    strategy,
    autotune_settings,
    batch_size, 
    weights_dir, 
    ds_dir
    ):
    # Load same/different datasets.
    datasets = []
    
    # Original
    ds = get_ds_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/original_val.tfrecords',
        autotune_settings=autotune_settings,
        ds_name='original',
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    # Regular
    ds = get_ds_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/regular_val.tfrecords',
        autotune_settings=autotune_settings,
        ds_name='regular',
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    # Irregular
    ds = get_ds_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/irregular_val.tfrecords',
        autotune_settings=autotune_settings,
        ds_name='irregular',
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    # Open
    ds = get_ds_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/open_val.tfrecords',
        autotune_settings=autotune_settings,
        ds_name='open',
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    # Wider line
    ds = get_ds_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/wider_line_val.tfrecords',
        autotune_settings=autotune_settings,
        ds_name='wider',
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    # Scrambled
    ds = get_ds_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/scrambled_val.tfrecords',
        autotune_settings=autotune_settings,
        ds_name='scrambled',
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    # Random color
    ds = get_ds_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/random_color_val.tfrecords',
        autotune_settings=autotune_settings,
        ds_name='random',
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    # Filled
    ds = get_ds_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/filled_val.tfrecords',
        autotune_settings=autotune_settings,
        ds_name='filled',
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    # Lines
    ds = get_ds_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/lines_val.tfrecords',
        autotune_settings=autotune_settings,
        ds_name='lines',
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    # Arrows
    ds = get_ds_with_ds_label(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/arrows_val.tfrecords',
        autotune_settings=autotune_settings,
        ds_name='arrows',
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    
    conditions = ['Original', 'Regular', 'Irregular', 'Open', 'Wider Line', 'Scrambled', 'Random Color', 'Filled', 'Lines', 'Arrows']
    
    # Define model architecture
    with strategy.scope():
        base_model, model = make_ResNet50_sd_classifier_with_ds_label(base_weights=None, base_trainable=True, gap=True)
        
        # Compile: set metric to auc (default is area under the ROC curve)
        model.compile(
                optimizer=tf.keras.optimizers.Adam(0.0003),
                loss={'sd': 'binary_crossentropy', 'ds_class': 'categorical_crossentropy'},
                metrics=[tf.keras.metrics.AUC()]
                )
        results = []
        for ds, condition in zip(datasets, conditions):
            # Fine-tune, imagenet, GAP.
            ft_im_gap = test_model_auc_with_ds_label(
                model=model,
                ds=ds,
                weights_dir=weights_dir,
                model_name='ResNet',
                condition=condition)
            results.extend(ft_im_gap)
        
    return pd.DataFrame(results, columns=['Model', 'Condition', 'Task', 'AUC'])

if __name__ == '__main__':
    # Training hyperparameters.
    EPOCHS_TOP = 5
    EPOCHS = 15
    BATCH_SIZE = 64
    STEPS_PER_EPOCH = 28000 // BATCH_SIZE
    VALIDATION_STEPS = 2800 // BATCH_SIZE

    # Train ResNet50 classifier
    AUTO = tf.data.experimental.AUTOTUNE
    strategy = tf.distribute.get_strategy()

    train_ds = get_master_ds_with_ds_labels(
        autotune_settings=AUTO,
        batch_size=BATCH_SIZE,
        ds_dir='data',
        process_img=True
        )
    fine_tune_model_with_ds_label(
        strategy=strategy,
        train_ds=train_ds,
        val_ds=None,
        save_name='simulation_3/task_classification/ResNet_instances',
        model_name='ResNet',
        epochs_top=EPOCHS_TOP,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        n=10,
        lr=0.0001,
        base_weights='imagenet',
        gap=True
        )
    # Test ResNet50 classifier
    df = test_models_all_ds_auc_with_ds_label(
        strategy=strategy,
        autotune_settings=AUTO,
        batch_size=560,
        weights_dir='simulation_3/task_classification/ResNet_instances', 
        ds_dir='data'
        )
    df.to_csv('simulation_3/task_classification/sim_3_ResNet50_taskclass_test_auc.csv')
    
    # Validate ResNet50 classifier
    df = validate_models_all_ds_auc_with_ds_label(
        strategy=strategy,
        autotune_settings=AUTO,
        batch_size=560,
        weights_dir='simulation_3/task_classification/ResNet_instances', 
        ds_dir='data'
        )
    df.to_csv('simulation_3/task_classification/sim_3_ResNet50_taskclass_val_auc.csv')