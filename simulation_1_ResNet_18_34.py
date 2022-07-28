from os import listdir
from os.path import isfile, join
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Input
import pandas as pd

# From https://github.com/breadbread1984/resnet18-34
def ResnetBlock(in_channels, out_channels, down_sample = False):
    inputs = tf.keras.Input((None, None, in_channels)); # inputs.shape = (batch, height, width, in_channels)
    if down_sample:
        shortcut = tf.keras.layers.Conv2D(
            out_channels, kernel_size = (1,1), 
            strides = (2,2), padding = 'same', 
            kernel_initializer = tf.keras.initializers.HeNormal()
            )(inputs)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    else:
        shortcut = inputs
    results = tf.keras.layers.Conv2D(
        out_channels, 
        kernel_size = (3,3), 
        strides = (2,2) if down_sample else (1,1), 
        padding = 'same', 
        kernel_initializer = tf.keras.initializers.HeNormal()
        )(inputs)
    results = tf.keras.layers.BatchNormalization()(results)
    results = tf.keras.layers.ReLU()(results)
    results = tf.keras.layers.Conv2D(
        out_channels, 
        kernel_size = (3,3), 
        strides = (1,1), 
        padding = 'same', 
        kernel_initializer = tf.keras.initializers.HeNormal()
        )(results)
    results = tf.keras.layers.BatchNormalization()(results)
    results = tf.keras.layers.Add()([results, shortcut])
    results = tf.keras.layers.ReLU()(results)
    return tf.keras.Model(inputs = inputs, outputs = results)

def ResNet18(**kwargs):
    inputs = tf.keras.Input((None, None, 3));
    results = tf.keras.layers.Conv2D(
        64, 
        kernel_size=(7,7), 
        strides=(2,2), 
        padding='same', 
        kernel_initializer = tf.keras.initializers.HeNormal()
        )(inputs)
    results = tf.keras.layers.BatchNormalization()(results)
    results = tf.keras.layers.ReLU()(results)
    results = tf.keras.layers.MaxPool2D(
        pool_size = (3,3), 
        strides = (2,2), 
        padding = 'same'
        )(results)
    results = ResnetBlock(64, 64)(results)
    results = ResnetBlock(64, 64)(results)
    results = ResnetBlock(64, 128, down_sample=True)(results)
    results = ResnetBlock(128, 128)(results)
    results = ResnetBlock(128, 256, down_sample=True)(results)
    results = ResnetBlock(256, 256)(results)
    results = ResnetBlock(256, 512, down_sample=True)(results)
    results = ResnetBlock(512, 512)(results)
    results = tf.keras.layers.GlobalAveragePooling2D()(results); # results.shape = (batch, 512)
    return tf.keras.Model(inputs = inputs, outputs = results, **kwargs)

def ResNet34(**kwargs):
    inputs = tf.keras.Input((None, None, 3))
    results = tf.keras.layers.Conv2D(
        64, 
        kernel_size=(7,7), 
        strides=(2,2), 
        padding='same', 
        kernel_initializer = tf.keras.initializers.HeNormal()
        )(inputs)
    results = tf.keras.layers.BatchNormalization()(results)
    results = tf.keras.layers.ReLU()(results)
    results = tf.keras.layers.MaxPool2D(
        pool_size = (3,3), 
        strides = (2,2), padding='same'
        )(results)
    results = ResnetBlock(64, 64)(results)
    results = ResnetBlock(64, 64)(results)
    results = ResnetBlock(64, 64)(results)
    results = ResnetBlock(64, 128, down_sample=True)(results)
    results = ResnetBlock(128, 128)(results)
    results = ResnetBlock(128, 128)(results)
    results = ResnetBlock(128, 128)(results)
    results = ResnetBlock(128, 256, down_sample=True)(results)
    results = ResnetBlock(256, 256)(results)
    results = ResnetBlock(256, 256)(results)
    results = ResnetBlock(256, 256)(results)
    results = ResnetBlock(256, 256)(results)
    results = ResnetBlock(256, 256)(results)
    results = ResnetBlock(256, 512, down_sample=True)(results)
    results = ResnetBlock(512, 512)(results)
    results = ResnetBlock(512, 512)(results)
    results = tf.keras.layers.GlobalAveragePooling2D()(results) # results.shape = (batch, 512)
    return tf.keras.Model(inputs = inputs, outputs = results, **kwargs)

# Models
def make_ResNet18_sketch_classifier(weights=None, trainable=True):
    # Inputs
    image = Input((128, 128, 3))

    # Get CNN features.
    base_model = ResNet18()
    base_model.load_weights(weights)

    # Freeze the base_model.
    base_model.trainable = trainable
    
    # Global average pooling is included in the base model
    x = base_model(image)

    # Add a fully-connected layer
    x = Dense(512, activation='relu')(x)
    
    # Add a logistic layer.
    predictions = Dense(250, activation='softmax')(x)
    
    # Model to train.
    model = Model(inputs=base_model.input, outputs=predictions)

    return base_model, model

def make_ResNet18_sd_classifier(
    weights_path=None, 
    base_trainable=False 
    ):
    # Define input tensor
    image = Input((128, 128, 3))
    
    # Base pre-trained model
    base_model = ResNet18()
    if weights_path:
        base_model.load_weights(weights_path)
    
    # Freeze base_model if necesarry 
    base_model.trainable = base_trainable

    # Global average pooling is included in the base model
    x = base_model(image)

    # Add a fully-connected layer
    x = Dense(512, activation='relu')(x)

    # Add logistic layer
    predictions = Dense(1, activation='sigmoid')(x)

    # This is the model we will train.
    model = Model(inputs=image, outputs=predictions)

    return base_model, model

def make_ResNet34_sketch_classifier(weights=None, trainable=True):
    # Inputs
    image = Input((128, 128, 3))

    # Get CNN features.
    base_model = ResNet34()
    base_model.load_weights(weights)

    # Freeze the base_model.
    base_model.trainable = trainable
    
    # Global average pooling is included in the base model
    x = base_model(image)

    # Add a fully-connected layer
    x = Dense(512, activation='relu')(x)
    
    # Add a logistic layer.
    predictions = Dense(250, activation='softmax')(x)
    
    # Model to train.
    model = Model(inputs=base_model.input, outputs=predictions)

    return base_model, model

def make_ResNet34_sd_classifier(
    weights_path=None, 
    base_trainable=False 
    ):
    # Define input tensor
    image = Input((128, 128, 3))
    
    # Base pre-trained model
    base_model = ResNet34()
    if weights_path:
        base_model.load_weights(weights_path)
    
    # Freeze base_model if necesarry 
    base_model.trainable = base_trainable

    # Global average pooling is included in the base model
    x = base_model(image)

    # Add a fully-connected layer
    x = Dense(512, activation='relu')(x)

    # Add logistic layer
    predictions = Dense(1, activation='sigmoid')(x)

    # This is the model we will train.
    model = Model(inputs=image, outputs=predictions)

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
    model_name,
    epochs_top,
    epochs,
    steps_per_epoch,
    validation_steps,
    weights_path,
    n=10,
    lr=0.0001,
    gap=True
    ):
    # Define make_model function
    if model_name == 'ResNet18':
        make_model_fn = make_ResNet18_sd_classifier
    elif model_name == 'ResNet34':
        make_model_fn = make_ResNet34_sd_classifier
    else:
        raise ValueError('Unrecognized model!')
    # Train n runs
    for i in range(n):
        # Define model
        base_model, model = make_model_fn(
            weights_path=weights_path,
            base_trainable=False
            )
        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.0003),
            loss='binary_crossentropy',
            metrics=['accuracy'])
        # Train
        filename = save_name + model_name +'_run_' + str(i) + '_log.csv'
        history_logger = tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)
        model.fit(
            train_ds,
            epochs=epochs_top,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,
            validation_steps=validation_steps,
            callbacks=[history_logger]
            )
        # Unfreeze Resnet
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
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,
            validation_steps=validation_steps,
            callbacks=[history_logger]
            )
        # Save weights
        weights_name = save_name + model_name + '_instance_' + str(i) + '.hdf5'
        model.save_weights(weights_name)
        
    return

# Test functions
def test_model_auc(ds, weights_dir, model_name, condition):
    """Tests 10 instances of a single model in a single condition using the area under the ROC curve. 
    Args:
        ds: dataset with cases from the 'same' and 'different' classes.
        weights_dir: directory of models weights to test.
        model_name: model name. String.
        condition: condition name. String.
    Returns:
        A list with test data: model_name, condition, area under the ROC curve.
    """
    # Define make_model function
    if model_name == 'ResNet18':
        make_model_fn = make_ResNet18_sd_classifier
    elif model_name == 'ResNet34':
        make_model_fn = make_ResNet34_sd_classifier
    else:
        raise ValueError('Unrecognized model!')
    
    # Define model architecture.
    base_model, model = make_model_fn(
            weights_path=None,
            base_trainable=True
            )
    
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
        metrics = model.evaluate(ds)
        models_data.append([model_name, condition, metrics[1]])
    
    return models_data

def test_all_models_auc(ds, condition, weights_dir, model_name):
    """Test all models in a single dataset. Condition is the name of the test dataset."""

    results = []
    
    # Fine-tune, imagenet, GAP.
    ft_im_gap = test_model_auc(
        ds=ds,
        weights_dir=f'{weights_dir}/{model_name}_ImageNet_GAP',
        model_name=model_name,
        condition=condition)
    results.extend(ft_im_gap)
        
    # Make pandas dataframe and return.
    return pd.DataFrame(results, columns=['Model', 'Condition', 'AUC'])

def test_all_models_all_ds_auc(batch_size, ds_dir, weights_dir, model_name):
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
        df = test_all_models_auc(ds, condition, weights_dir=weights_dir, model_name=model_name)
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

    # Train: ResNet18 imagenet with GAP
    fine_tune_model(
        train_ds=train_ds,
        val_ds=val_ds,
        save_name='simulation_1/ResNet18_ImageNet_GAP/',
        model_name='ResNet18',
        epochs_top=EPOCHS_TOP,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        n=10,
        lr=0.0001,
        gap=True,
        weights_path='imagenet_weights/resnet18_weights.h5')

    # Train: Resnet34 imagenet with GAP
    fine_tune_model(
        train_ds=train_ds,
        val_ds=val_ds,
        save_name='simulation_1/ResNet34_ImageNet_GAP/',
        model_name='ResNet34',
        epochs_top=EPOCHS_TOP,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        n=10,
        lr=0.0001,
        gap=True,
        weights_path='imagenet_weights/resnet34_weights.h5')

    # Test ResNet18 models and save results.
    df = test_all_models_all_ds_auc(
        batch_size=112,
        ds_dir='data',
        weights_dir='simulation_1',
        model_name='ResNet18'
        )
    df.to_csv('simulation_1/sim_1_resnet18_auc.csv')

    # Test ResNet34 models and save results.
    df = test_all_models_all_ds_auc(
        batch_size=112,
        ds_dir='data',
        weights_dir='simulation_1',
        model_name='ResNet34',
        )
    df.to_csv('simulation_1/sim_1_resnet34_auc.csv')