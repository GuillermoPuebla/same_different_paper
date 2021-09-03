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

def make_ResNet50_sd_classifier(base_weights='imagenet', base_trainable=False, gap=True):
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
    predictions = Dense(1, activation='sigmoid')(x)

    # This is the model we will train.
    model = Model(inputs=base_model.input, outputs=predictions)

    return base_model, model

def get_dataset(batch_size, tfrecord_dir, is_training=True, process_img=True, relnet=False):
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
        
        if relnet:
            question = tf.constant([1, 0], dtype=tf.int64)
            return (image, question), label
        else:
            return image, label
            
    # Parse dataset.
    dataset = raw_image_dataset.map(read_tfrecord)
    
    # Always shuffle for simplicity.
    dataset = dataset.shuffle(5600)
    
    if is_training:
        # Infinite dataset to avoid the potential last partial batch in each epoch.
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)
    
    return dataset

def get_master_dataset(batch_size, dont_include, process_img=True, relnet=False):
    """Builds dataset that samples each batch from one of the training
    datasets except 'dont_include'."""
    
    # Make datasets and append if it is not the dont_include dataset.
    datasets = []
    
    # Original.
    ds_original = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/original_train.tfrecords',
        is_training=True,
        process_img=process_img,
        relnet=relnet)
    
    if dont_include != 'Original':
        datasets.append(ds_original)
    
    # Regular.
    ds_regular = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/regular_train.tfrecords',
        is_training=True,
        process_img=process_img,
        relnet=relnet)
    
    if dont_include != 'Regular':
        datasets.append(ds_regular)
    
    # Irregular.
    ds_irregular = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/irregular_train.tfrecords',
        is_training=True,
        process_img=process_img,
        relnet=relnet)
    
    if dont_include != 'Irregular':
        datasets.append(ds_irregular)
    
    # Open.
    ds_open = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/open_train.tfrecords',
        is_training=True,
        process_img=process_img,
        relnet=relnet)
    
    if dont_include != 'Open':
        datasets.append(ds_open)

    # Wider line.
    ds_wider = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/wider_line_train.tfrecords',
        is_training=True,
        process_img=process_img,
        relnet=relnet)
    
    if dont_include != 'Wider line':
        datasets.append(ds_wider)

    # Scrambled.
    ds_scrambled = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/scrambled_train.tfrecords',
        is_training=True,
        process_img=process_img,
        relnet=relnet)

    if dont_include != 'Scrambled':
        datasets.append(ds_scrambled)
    
    # Random color.
    ds_random = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/random_color_train.tfrecords',
        is_training=True,
        process_img=process_img,
        relnet=relnet)
    
    if dont_include != 'Random color':
        datasets.append(ds_random)
    
    # Filled.
    ds_filled = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/filled_train.tfrecords',
        is_training=True,
        process_img=process_img,
        relnet=relnet)
    
    if dont_include != 'Filled':
        datasets.append(ds_filled)
    
    # Lines.
    ds_lines = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/lines_train.tfrecords',
        is_training=True,
        process_img=process_img,
        relnet=relnet)
    
    if dont_include != 'Lines':
        datasets.append(ds_lines)
    
    # Arrows.
    ds_arrows = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/arrows_train.tfrecords',
        is_training=True,
        process_img=process_img,
        relnet=relnet)
    
    if dont_include != 'Arrows':
        datasets.append(ds_arrows)

    # Define uniform dataset sampling.
    choice_dataset = tf.data.Dataset.range(len(datasets)).repeat()
    
    return tf.data.experimental.choose_from_datasets(datasets, choice_dataset)

def train_best_sim1_model(
    train_ds,
    val_ds,
    save_name,
    epochs_top,
    epochs,
    steps_per_epoch,
    validation_steps,
    n=10,
    lr=0.0001):
    
    for i in range(n):
        # Define best model sim1: ImageNet & fine-tune & GAP.
        base_model, model = make_ResNet50_sd_classifier(base_weights='imagenet',
                                                        base_trainable=False,
                                                        gap=True)
        # Compile.
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.0003),
            loss='binary_crossentropy',
            metrics=['accuracy'])

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
            loss='binary_crossentropy',
            metrics=['accuracy'])
        
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

def train_in_all_datasets(
    epochs_top,
    epochs,
    steps_per_epoch,
    validation_steps,
    n=10,
    lr=0.0001,
    batch_size=64):
    
    # Define all master datasets.
    no_irregular_ds = get_master_dataset(batch_size=batch_size, dont_include='Irregular', process_img=True)
    no_regular_ds = get_master_dataset(batch_size=batch_size, dont_include='Regular', process_img=True)
    no_open_ds = get_master_dataset(batch_size=batch_size, dont_include='Open', process_img=True)
    no_wider_ds = get_master_dataset(batch_size=batch_size, dont_include='Wider line', process_img=True)
    no_scrambled_ds = get_master_dataset(batch_size=batch_size, dont_include='Scrambled', process_img=True)
    no_random_ds = get_master_dataset(batch_size=batch_size, dont_include='Random color', process_img=True)
    no_filled_ds = get_master_dataset(batch_size=batch_size, dont_include='Filled', process_img=True)
    no_lines_ds = get_master_dataset(batch_size=batch_size, dont_include='Lines', process_img=True)
    no_arrows_ds = get_master_dataset(batch_size=batch_size, dont_include='Arrows', process_img=True)
    
    train_datasets = [no_irregular_ds, no_regular_ds, no_open_ds,
                      no_wider_ds, no_scrambled_ds, no_random_ds,
                      no_filled_ds, no_lines_ds, no_arrows_ds]
    
    # Validation dataset.
    val_ds = get_dataset(
    batch_size=batch_size,
    tfrecord_dir='sd_data/original_val.tfrecords',
    is_training=False,
    process_img=True)
    
    # Train model in each dataset 10 times.
    ds_and_names = [(no_irregular_ds, 'simulation_2/no_irregular/'),
                    (no_regular_ds, 'simulation_2/no_regular/'),
                    (no_open_ds, 'simulation_2/no_open/'),
                    (no_wider_ds, 'simulation_2/no_wider/'),
                    (no_scrambled_ds, 'simulation_2/no_scrambled/'),
                    (no_random_ds, 'simulation_2/no_random/'),
                    (no_filled_ds, 'simulation_2/no_filled/'),
                    (no_lines_ds, 'simulation_2/no_lines/'),
                    (no_arrows_ds, 'simulation_2/no_arrows/')]
    
    for ds, name in ds_and_names:
        train_best_sim1_model(
            train_ds=ds,
            val_ds=val_ds,
            save_name=name,
            epochs_top=epochs_top,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            n=n,
            lr=lr)
    return

def get_master_test_dataset(batch_size, dont_include, relnet=False):
    """Returns test dataset with all the s/d files except the ones
    corresponding to the dont_include condition."""
    
    # Define all condition records.
    all_records = []

    if dont_include != 'Original':
        all_records.append('data/original_test.tfrecords')
        
    if dont_include != 'Regular':
        all_records.append('data/regular_test.tfrecords')
        
    if dont_include != 'Irregular':
        all_records.append('data/irregular_test.tfrecords')
        
    if dont_include != 'Open':
        all_records.append('data/open_test.tfrecords')
    
    if dont_include != 'Wider line':
        all_records.append('data/wider_line_test.tfrecords')
        
    if dont_include != 'Scrambled':
        all_records.append('data/scrambled_test.tfrecords')
        
    if dont_include != 'Random color':
        all_records.append('data/random_color_test.tfrecords')
        
    if dont_include != 'Filled':
        all_records.append('data/filled_test.tfrecords')
        
    if dont_include != 'Lines':
        all_records.append('data/lines_test.tfrecords')
    
    if dont_include != 'Arrows':
        all_records.append('data/arrows_test.tfrecords')
    
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=all_records,
        is_training=False,
        process_img=True,
        relnet=relnet)
    
    return ds

def test_model_auc(ds_train, ds_untrained, weights_dir, gap, model_name, condition):
    """Tests 10 versions of a single model in a single condition using the area under the ROC curve. 
    Args:
        ds_train: dataset with cases from the 'same' and 'different' classes from all trained conditions.
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
    weights_paths = [join(weights_dir, f) for f in listdir(weights_dir) if isfile(join(weights_dir, f))]
    
    # Test each model.
    models_data = []
    for w_path in weights_paths:
        model.load_weights(w_path)
        metrics_trained = model.evaluate(ds_train)
        metrics_untrained = model.evaluate(ds_untrained)
        models_data.append([model_name, condition, 'Trained', metrics_trained[1]])
        models_data.append([model_name, condition, 'Untrained', metrics_untrained[1]])
    
    return models_data

def test_all_ds_auc(batch_size):
    # Load same/different datasets.
    untrained_dss = []
    trained_dss = []

    # Regular.
    regular_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/regular_test.tfrecords',
        is_training=False,
        process_img=True)
    
    all_but_regular_ds = get_master_test_dataset(batch_size=batch_size, dont_include='Regular')
    
    untrained_dss.append(regular_ds)
    trained_dss.append(all_but_regular_ds)
    
    # Irregular.
    irregular_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/irregular_test.tfrecords',
        is_training=False,
        process_img=True)
    
    all_but_irregular_ds = get_master_test_dataset(batch_size=batch_size, dont_include='Irregular')
    
    untrained_dss.append(irregular_ds)
    trained_dss.append(all_but_irregular_ds)
    
    # Open.
    open_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/open_test.tfrecords',
        is_training=False,
        process_img=True)
    
    all_but_open_ds = get_master_test_dataset(batch_size=batch_size, dont_include='Open')
    
    untrained_dss.append(open_ds)
    trained_dss.append(all_but_open_ds)

    # Wider line.
    wider_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/wider_line_test.tfrecords',
        is_training=False,
        process_img=True)
    
    all_but_wider_ds = get_master_test_dataset(batch_size=batch_size, dont_include='Wider line')
    
    untrained_dss.append(wider_ds)
    trained_dss.append(all_but_wider_ds)
    
    # Scrambled.
    scrambled_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/scrambled_test.tfrecords',
        is_training=False,
        process_img=True)
    
    all_but_scrambled_ds = get_master_test_dataset(batch_size=batch_size, dont_include='Scrambled')
    
    untrained_dss.append(scrambled_ds)
    trained_dss.append(all_but_scrambled_ds)
    
    # Random color.
    random_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/random_color_test.tfrecords',
        is_training=False,
        process_img=True)
    
    all_but_random_ds = get_master_test_dataset(batch_size=batch_size, dont_include='Random color')
    
    untrained_dss.append(random_ds)
    trained_dss.append(all_but_random_ds)
    
    # Filled.
    filled_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/filled_test.tfrecords',
        is_training=False,
        process_img=True)
    
    all_but_filled_ds = get_master_test_dataset(batch_size=batch_size, dont_include='Filled')
    
    untrained_dss.append(filled_ds)
    trained_dss.append(all_but_filled_ds)
    
    # Lines.
    lines_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/lines_test.tfrecords',
        is_training=False,
        process_img=True)
    
    all_but_lines_ds = get_master_test_dataset(batch_size=batch_size, dont_include='Lines')
    
    untrained_dss.append(lines_ds)
    trained_dss.append(all_but_lines_ds)
    
    # Arrows.
    arrows_ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/arrows_test.tfrecords',
        is_training=False,
        process_img=True)
    
    all_but_arrows_ds = get_master_test_dataset(batch_size=batch_size, dont_include='Arrows')
    
    untrained_dss.append(arrows_ds)
    trained_dss.append(all_but_arrows_ds)
    
    conditions = ['Regular', 'Irregular', 'Open', 'Wider Line', 'Scrambled', 'Random Color', 'Filled', 'Lines', 'Arrows']
    
    weights_dirs = ['simulation_2/no_regular', 'simulation_2/no_irregular', 'simulation_2/no_open',
                    'simulation_2/no_wider', 'simulation_2/no_scrambled', 'simulation_2/no_random',
                    'simulation_2/no_filled', 'simulation_2/no_lines', 'simulation_2/no_arrows']
    
    results = []
    for ds1, ds2, condition, w_dir in zip(trained_dss, untrained_dss, conditions, weights_dirs):
        data = test_model_auc(
            ds_train=ds1,
            ds_untrained=ds2,
            weights_dir=w_dir,
            gap=True,
            model_name='ResNet',
            condition=condition)
        results.extend(data)
    
    return pd.DataFrame(results, columns=['Model', 'Condition', 'Testing data', 'AUC'])

if __name__ == '__main__':
    # Training hyperparameters.
    EPOCHS_TOP = 5
    EPOCHS = 13
    BATCH_SIZE = 64
    STEPS_PER_EPOCH = 28000 // BATCH_SIZE
    VALIDATION_STEPS = 2800 // BATCH_SIZE

    # Train instances in each training condition.
    train_in_all_datasets(
        epochs_top=EPOCHS_TOP,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        n=10,
        lr=0.0001,
        batch_size=BATCH_SIZE)
    
    # Test and save.
    df = test_all_ds_auc(batch_size=112)
    df.to_csv('simulation_2/sim_2_resnet_auc.csv')
    print('All model instances trained and tested!')



