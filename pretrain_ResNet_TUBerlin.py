import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import ResNet50

def make_ResNet50_classifier(weights=None, trainable=True):
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

# Dataset reading function.
def get_dataset(batch_size, tfrecord_dir, is_training=True, process_img=True):
    # Load dataset.
    raw_image_dataset = tf.data.TFRecordDataset(tfrecord_dir)
    
    # Define example reading function.
    def read_tfrecord(serialized_example):
        # Create a dictionary describing the features.
        feature_description = {
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)}
        
        # Parse example.
        example = tf.io.parse_single_example(serialized_example, feature_description)

        # Get image.
        image = tf.io.decode_png(example['image_raw'], channels=3)

        # Ensure shape dimensions are constant.
        image = tf.reshape(image, [128, 128, 3])

        # Preprocess image.
        if process_img:
            image = tf.cast(image, tf.float64)
            image /= 255.0
            # Sample-wise center image.
            mean = tf.reduce_mean(image)
            image -= mean
            # Sample-wise std normalization.
            std = tf.math.reduce_std(image)
            image /= std
        
        # Cast label to int64
        label = example['label']
        
        # Get one-hot label.
        label = tf.one_hot(label, 250)
        label = tf.cast(label, tf.int64)
        
        return image, label
        
    # Parse dataset.
    dataset = raw_image_dataset.map(read_tfrecord)
    
    if is_training:
        # I'm shuffling the datasets at creation time, so this is no necesarry for now.
        dataset = dataset.shuffle(10000) # half test dataset size.
        # Infinite dataset to avoid the potential last partial batch in each epoch.
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)
    
    return dataset

if __name__ == '__main__':
    # Training parameters.
    BATCH_SIZE = 64
    EPOCHS = 10
    STEPS_PER_EPOCH = 960000 // BATCH_SIZE  # train dataset size: 0.8*60*80*250=960,000
    LR = 0.0003

    # Get datasets.
    train_ds = get_dataset(batch_size=BATCH_SIZE,
                        tfrecord_dir='data/TUBerlin_train.tfrecords',
                        is_training=True,
                        process_img=True)

    test_ds = get_dataset(batch_size=BATCH_SIZE,
                        tfrecord_dir='data/TUBerlin_test.tfrecords',
                        is_training=False,
                        process_img=True)

    # Get model.
    base_model, model = make_ResNet50_classifier(weights=None, trainable=True)

    # Compile.
    opt = tf.keras.optimizers.Adam(learning_rate=LR)
    model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])

    # Train and save logs and weights.
    log_name = 'simulation1/ResNet50_TUBerlin_weights_log.csv'
    history_logger = tf.keras.callbacks.CSVLogger(log_name, separator=",", append=True)

    checkpoint_filepath = 'simulation1/ResNet50_TUBerlin_weights.hdf5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=False)

    model_history = model.fit(train_ds,
                            steps_per_epoch=STEPS_PER_EPOCH,
                            epochs=EPOCHS,
                            validation_data=test_ds,
                            callbacks=[history_logger, model_checkpoint_callback])

