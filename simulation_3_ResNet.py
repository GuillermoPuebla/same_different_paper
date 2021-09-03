from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Input

def make_ResNet50_sketch_classifier(weights=None, trainable=True):
    # Inputs.
    image = Input((128, 128, 3))

    # Get CNN features.
    base_model = ResNet50(weights=weights, include_top=False, input_tensor=image)

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

def make_ResNet50_sd_classifier(base_weights='imagenet',
                                base_trainable=False,
                                gap=True,
                                secondary_outputs=True):
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
        rel_pos = Dense(1, activation='sigmoid', name='rel_pos')(x)
        # This is the model we will train.
        model = Model(inputs=base_model.input, outputs=[predictions, rel_pos])
    else:
        # This is the model we will train.
        model = Model(inputs=base_model.input, outputs=predictions)

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
        raw_image_dataset = tf.data.TFRecordDataset(tfrecord_dir, num_parallel_reads=2)
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

def get_master_dataset(batch_size, process_img=True):
    """Builds dataset that samples each batch from one of the training datasets
    assiging a same-different sample weight of 1 only for the original
    condition and a relative-position sample weight 1 for all conitions."""
    
    # Make datasets and append if it is not the dont_include dataset.
    datasets = []
    
    # Original.
    ds_original = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/original_train.tfrecords',
        is_training=True,
        process_img=process_img,
        sd_sample_weight=True)
    datasets.append(ds_original)
    
    # Regular.
    ds_regular = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/regular_train.tfrecords',
        is_training=True,
        process_img=process_img,
        sd_sample_weight=False)
    datasets.append(ds_regular)
    
    # Irregular.
    ds_irregular = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/irregular_train.tfrecords',
        is_training=True,
        process_img=process_img,
        sd_sample_weight=False)
    datasets.append(ds_irregular)
    
    # Open.
    ds_open = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/open_train.tfrecords',
        is_training=True,
        process_img=process_img,
        sd_sample_weight=False)
    datasets.append(ds_open)

    # Wider line.
    ds_wider = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/wider_line_train.tfrecords',
        is_training=True,
        process_img=process_img,
        sd_sample_weight=False)
    datasets.append(ds_wider)

    # Scrambled.
    ds_scrambled = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/scrambled_train.tfrecords',
        is_training=True,
        process_img=process_img,
        sd_sample_weight=False)
    datasets.append(ds_scrambled)
    
    # Random color.
    ds_random = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/random_color_train.tfrecords',
        is_training=True,
        process_img=process_img,
        sd_sample_weight=False)
    datasets.append(ds_random)
    
    # Filled.
    ds_filled = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/filled_train.tfrecords',
        is_training=True,
        process_img=process_img,
        sd_sample_weight=False)
    datasets.append(ds_filled)
    
    # Lines.
    ds_lines = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/lines_train.tfrecords',
        is_training=True,
        process_img=process_img,
        sd_sample_weight=False)
    datasets.append(ds_lines)
    
    # Arrows.
    ds_arrows = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/arrows_train.tfrecords',
        is_training=True,
        process_img=process_img,
        sd_sample_weight=False)
    datasets.append(ds_arrows)
    
    # Oversample the original dataset (50%) because I'm samplig tasks (same-diff, rel-pos)
    choice_tensor = tf.constant(value=[0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0, 9], dtype=tf.int64)
    choice_dataset = tf.data.Dataset.from_tensor_slices(choice_tensor).repeat()
    
    return tf.data.experimental.choose_from_datasets(datasets, choice_dataset)

def fine_tune_model(
    train_ds,
    val_ds,
    save_name,
    epochs_top,
    epochs,
    steps_per_epoch,
    validation_steps,
    n=10,
    lr=0.0001,
    base_weights='imagenet',
    gap=True):
    
    for i in range(n):
        # Define model.
        base_model, model = make_ResNet50_sd_classifier(
            base_weights=base_weights,
            base_trainable=False,
            gap=gap,
            secondary_outputs=True)
        
        # Compile.
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.0003),
            loss={'sd': 'binary_crossentropy', 'rel_pos': 'binary_crossentropy'},
            metrics={'sd': 'binary_accuracy', 'rel_pos': 'binary_accuracy'})

        # Train.
        model.fit(train_ds,
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
        model.fit(train_ds,
                  epochs=epochs,
                  steps_per_epoch=steps_per_epoch,
                  validation_data=val_ds,
                  validation_steps=validation_steps)

        # Save weights.
        weights_name = save_name + 'instance_' + str(i) + '.hdf5'
        model.save_weights(weights_name)
        
    return

def test_model_auc(ds, weights_dir, gap, model_name, condition):
    """Tests 10 versions of a single model in a single condition using the area under the ROC curve. 
    Args:
        ds: dataset with cases from the 'same' and 'different' classes.
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
    weights_paths = [join(weights_dir, f) for f in listdir(weights_dir) if isfile(join(weights_dir, f))]
    
    # Test each model.
    models_data = []
    for w_path in weights_paths:
        model.load_weights(w_path)
        metrics = model.evaluate(ds)
        # model.metrics_names = ['loss', 'sd_loss', 'rel_pos_loss', 'sd_auc_1', 'rel_pos_auc_1']
        models_data.append([model_name, condition, 'Same-Different', metrics[3]])
        models_data.append([model_name, condition, 'Relative position', metrics[4]])
    
    return models_data

def test_models_all_ds_auc(batch_size):
    # Load same/different datasets.
    datasets = []
    
    # Original.
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/original_test.tfrecords',
        is_training=False,
        process_img=True)
    
    datasets.append(ds)
    
    # Regular.
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/regular_test.tfrecords',
        is_training=False,
        process_img=True)
    
    datasets.append(ds)
    
    # Irregular.
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/irregular_test.tfrecords',
        is_training=False,
        process_img=True)
    
    datasets.append(ds)
    
    # Open.
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/open_test.tfrecords',
        is_training=False,
        process_img=True)
    
    datasets.append(ds)

    # Wider line.
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/wider_line_test.tfrecords',
        is_training=False,
        process_img=True)
    
    datasets.append(ds)
    
    # Scrambled.
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/scrambled_test.tfrecords',
        is_training=False,
        process_img=True)
    
    datasets.append(ds)
    
    # Random color.
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/random_color_test.tfrecords',
        is_training=False,
        process_img=True)
    
    datasets.append(ds)
    
    # Filled.
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/filled_test.tfrecords',
        is_training=False,
        process_img=True)
    
    datasets.append(ds)
    
    # Lines.
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/lines_test.tfrecords',
        is_training=False,
        process_img=True)
    
    datasets.append(ds)
    
    # Arrows.
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/arrows_test.tfrecords',
        is_training=False,
        process_img=True)
    
    datasets.append(ds)
    
    conditions = ['Original', 'Regular', 'Irregular', 'Open', 'Wider Line', 'Scrambled', 'Random Color', 'Filled', 'Lines', 'Arrows']
    
    results = []
    for ds, condition in zip(datasets, conditions):
        # Fine-tune, imagenet, GAP.
        ft_im_gap = test_model_auc(
            ds=ds,
            weights_dir='simulation_3/ResNet_instances',
            gap=True,
            model_name='ResNet',
            condition=condition)
        results.extend(ft_im_gap)
        
    return pd.DataFrame(results, columns=['Model', 'Condition', 'Task', 'AUC'])

if __name__ == '__main__':
    # Training hyperparameters.
    EPOCHS_TOP = 5
    EPOCHS = 13
    BATCH_SIZE = 64
    STEPS_PER_EPOCH = 28000 // BATCH_SIZE
    VALIDATION_STEPS = 2800 // BATCH_SIZE

    # Train ResNet50 classifier.
    train_ds = get_master_dataset(batch_size=BATCH_SIZE, process_img=True)

    val_ds = get_dataset(
        batch_size=BATCH_SIZE,
        tfrecord_dir='data/original_val.tfrecords',
        is_training=False,
        process_img=True,
        sd_sample_weight=True)

    fine_tune_model(
        train_ds=train_ds,
        val_ds=val_ds,
        save_name='simulation_3/ResNet_instances',
        epochs_top=EPOCHS_TOP,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        n=10,
        lr=0.0001,
        base_weights='imagenet',
        gap=True)

    # Test ResNet50 classifier.
    df = test_models_all_ds_auc(batch_size=112)
    df.to_csv('simulation_3/sim_3_auc.csv')

