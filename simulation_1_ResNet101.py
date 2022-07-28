from os import listdir
from os.path import isfile, join
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Input
import pandas as pd

# Model
def make_ResNet101_sketch_classifier(weights=None, trainable=True):
    # Inputs.
    image = Input((128, 128, 3))

    # Get CNN features.
    base_model = ResNet101(include_top=False, weights=None, input_tensor=image)

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

def make_ResNet101_sd_classifier(base_weights='imagenet', tuberlin_path=None, base_trainable=False, gap=True):
    # Define input tensor.
    image = Input((128, 128, 3))
    
    # Base pre-trained model.
    if base_weights == 'imagenet':
        base_model = ResNet101(weights='imagenet', include_top=False, input_tensor=image)
    elif base_weights is None:
        base_model = ResNet101(weights=None, include_top=False, input_tensor=image)
        
    elif base_weights == 'TU-Berlin':
        # Get weights from sketch classifier base.
        sketch_b, sketch_m = make_ResNet101_sketch_classifier(weights=None, trainable=True)
        sketch_m.load_weights(tuberlin_path)
        # Note that loading the weights to the full model affects the base model too.
        base_weights = []
        for layer in sketch_b.layers:
            base_weights.append(layer.get_weights())
            
        # Set weights of base model.
        base_model = ResNet101(weights=None, include_top=False, input_tensor=image)
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

# Dataset
def get_dataset(batch_size, tfrecord_dir, is_training=True, process_img=True, relnet=False):
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

# Train function
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
    gap=True,
    tuberlin_path=None
    ):
    
    for i in range(n):
        # Define model
        base_model, model = make_ResNet101_sd_classifier(
            base_weights=base_weights,
            tuberlin_path=tuberlin_path,
            base_trainable=False,
            gap=gap
            )
        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.0003),
            loss='binary_crossentropy',
            metrics=['accuracy'])
        # Train
        model.fit(train_ds,
                  epochs=epochs_top,
                  steps_per_epoch=steps_per_epoch,
                  validation_data=val_ds,
                  validation_steps=validation_steps)
        # Unfreeze Resnet50
        base_model.trainable = True
        # Re-compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss='binary_crossentropy',
            metrics=['accuracy']
            )
        # Train
        model.fit(train_ds,
                  epochs=epochs,
                  steps_per_epoch=steps_per_epoch,
                  validation_data=val_ds,
                  validation_steps=validation_steps)
        # Save weights
        weights_name = save_name + 'ResNet101_instance_' + str(i) + '.hdf5'
        model.save_weights(weights_name)
        
    return

# Test functions
def test_model_auc(ds, weights_dir, gap, model_name, condition):
    """Tests 10 instances of a single model in a single condition using the area under the ROC curve. 
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
    base_model, model = make_ResNet101_sd_classifier(base_weights=None, base_trainable=True, gap=gap)
    
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
        models_data.append([model_name, condition, metrics[1]])
    
    return models_data

def test_all_models_auc(ds, condition, weights_dir):
    """Test all models in a single dataset. Condition is the name of the test dataset."""

    results = []
    
    # Fine-tune, imagenet, GAP.
    ft_im_gap = test_model_auc(
        ds=ds,
        weights_dir=weights_dir+'/ResNet101_ImageNet_GAP',
        gap=True,
        model_name='ResNet101_ImageNet_GAP',
        condition=condition)
    results.extend(ft_im_gap)
   
    # Fine-tune, imagenet, flatten.
    ft_im_flatten = test_model_auc(
        ds=ds,
        weights_dir=weights_dir+'/ResNet101_ImageNet_flatten',
        gap=False,
        model_name='ResNet101_ImageNet_flatten',
        condition=condition)
    results.extend(ft_im_flatten)
    
    # Fine-tune, TU-Berlin, GAP.
    ft_tu_gap = test_model_auc(
        ds=ds,
        weights_dir=weights_dir+'/ResNet101_TUBerlin_GAP',
        gap=True,
        model_name='ResNet101_TUBerlin_GAP',
        condition=condition)
    results.extend(ft_tu_gap)
    
    # Fine-tune, TU-Berlin, flatten.
    ft_im_gap = test_model_auc(
        ds=ds,
        weights_dir=weights_dir+'/ResNet101_TUBerlin_flatten',
        gap=False,
        model_name='ResNet101_TUBerlin_flatten',
        condition=condition)
    results.extend(ft_im_gap)
        
    # Make pandas dataframe and return.
    return pd.DataFrame(results, columns=['Model', 'Condition', 'AUC'])

def test_all_models_all_ds_auc(batch_size, ds_dir, weights_dir):
    # Load same/different datasets.
    datasets = []

    # Original
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/original_test.tfrecords',
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    
    # Regular
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/regular_test.tfrecords',
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    
    # Irregular
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/irregular_test.tfrecords',
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    
    # Open
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/open_test.tfrecords',
        is_training=False,
        process_img=True
        )
    datasets.append(ds)

    # Wider line
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/wider_line_test.tfrecords',
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    
    # Scrambled
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/scrambled_test.tfrecords',
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    
    # Random color
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/random_color_test.tfrecords',
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    
    # Filled
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/filled_test.tfrecords',
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    
    # Lines
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/lines_test.tfrecords',
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    
    # Arrows
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir=ds_dir+'/arrows_test.tfrecords',
        is_training=False,
        process_img=True
        )
    datasets.append(ds)
    
    conditions = ['Original', 'Regular', 'Irregular', 'Open', 'Wider Line', 'Scrambled', 'Random Color', 'Filled', 'Lines', 'Arrows']
    
    dfs = []
    for ds, condition in zip(datasets, conditions):
        df = test_all_models_auc(ds, condition, weights_dir)
        dfs.append(df)
    
    return pd.concat(dfs)

if __name__ == '__main__':
    # Training hyperparameters.
    EPOCHS_TOP = 5
    EPOCHS = 10
    BATCH_SIZE = 64
    STEPS_PER_EPOCH = 28000 // BATCH_SIZE
    VALIDATION_STEPS = 2800 // BATCH_SIZE

    # Training data.
    train_ds = get_dataset(
        batch_size=BATCH_SIZE,
        tfrecord_dir='data/original_train.tfrecords',
        is_training=True,
        process_img=True)

    val_ds = get_dataset(
        batch_size=BATCH_SIZE,
        tfrecord_dir='data/original_val.tfrecords',
        is_training=False,
        process_img=True)

    # Train: imagenet with GAP
    fine_tune_model(
        train_ds=train_ds,
        val_ds=val_ds,
        save_name='simulation_1/ResNet101_ImageNet_GAP/',
        epochs_top=EPOCHS_TOP,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        n=10,
        lr=0.0001,
        base_weights='imagenet',
        gap=True)

    # Test all models and save results.
    df = test_all_models_all_ds_auc(
        batch_size=112,
        ds_dir='data',
        weights_dir='simulation_1'
        )
    df.to_csv('simulation_1/sim_1_resnet101_auc.csv')
    