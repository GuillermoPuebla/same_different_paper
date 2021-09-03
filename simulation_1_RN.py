from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate, Lambda, Average, Dropout

# Relation Network functions.
def get_dense(n, units):
    r = []
    for k in range(n):
        r.append(Dense(units, activation='relu'))
    return r

def get_MLP(n, denses):
    def g(x):
        d = x
        for k in range(n):
            d = denses[k](d)
        return d
    return g

def dropout_dense(x, units):
    y = Dense(units, activation='relu')(x)
    y = Dropout(0.5)(y)
    return y

def build_tag(conv):
    d = K.int_shape(conv)[2]
    tag = np.zeros((d,d,2))
    for i in range(d):
        for j in range(d):
            tag[i,j,0] = float(int(i%d))/(d-1)*2-1
            tag[i,j,1] = float(int(j%d))/(d-1)*2-1
    tag = K.variable(tag)
    tag = K.expand_dims(tag, axis=0)
    batch_size = K.shape(conv)[0]
    tag = K.tile(tag, [batch_size,1,1,1])
    return tag

def slice_1(t):
    return t[:, 0, :, :]

def slice_2(t):
    return t[:, 1:, :, :]

def slice_3(t):
    return t[:, 0, :]

def slice_4(t):
    return t[:, 1:, :]

def make_ResNet50_relnet(dataset='SVRT',
                         resnet_layer='last_size_8',
                         trainable=False):
    # Inputs.
    image = Input((128, 128, 3))
    
    if dataset=='sort-of-clevr':
        question = Input((11,))
    elif dataset=='SVRT':
        question = Input((2,)) # same-different ([1, 0]) or relative position ([0, 1]).
    else:
        raise ValueError('dataset not supported!')
    
    # Get CNN features.
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=image)
        
    if resnet_layer=='last_size_4':
        layer_name = 'conv5_block3_out' # shape: (None, 4, 4, 2048)
    elif resnet_layer=='last_size_8':
        layer_name = 'conv4_block6_out' # shape: (None, 8, 8, 1024)
    else:
        raise ValueError('layer not supported!')
    cnn_features = base_model.get_layer(layer_name).output
    
    # Freeze the base_model.
    base_model.trainable = trainable
    
    # Make tag and append to cnn features.
    tag = build_tag(cnn_features)
    cnn_features = Concatenate()([cnn_features, tag])
    
    # Make list with objects.
    shapes = cnn_features.shape
    w, h = shapes[1], shapes[2]
    
    slice_layer1 = Lambda(slice_1)
    slice_layer2 = Lambda(slice_2)
    slice_layer3 = Lambda(slice_3)
    slice_layer4 = Lambda(slice_4)

    features = []
    for k1 in range(w):
        features1 = slice_layer1(cnn_features)
        cnn_features = slice_layer2(cnn_features)
        for k2 in range(h):
            features2 = slice_layer3(features1)
            features1 = slice_layer4(features1)
            features.append(features2)

    # Make list with all combinations of objects.
    relations = []
    concat = Concatenate()
    for feature1 in features:
        for feature2 in features:
            relations.append(concat([feature1, feature2, question]))
    
    # g function.
    g_MLP = get_MLP(4, get_dense(4, units=512))
    mid_relations = []
    for r in relations:
        mid_relations.append(g_MLP(r))
    combined_relation = Average()(mid_relations)

    # f function.
    rn = Dense(512, activation='relu')(combined_relation)
    rn = dropout_dense(rn, units=512)
    
    # Output.
    if dataset == 'sort-of-clevr':
        output_units = 10
        answer = Dense(output_units, activation='softmax')(rn)
    elif dataset == 'SVRT':
        output_units = 1
        answer = Dense(output_units, activation='sigmoid')(rn)
    
    model = Model(inputs=[image, question], outputs=answer)
    
    return base_model, model

def fine_tune_relnet(train_ds,
                     val_ds,
                     save_name,
                     resnet_layer,
                     epochs_top,
                     epochs,
                     steps_per_epoch,
                     validation_steps,
                     n=10,
                     lr=0.0001):
    
    # Repate trainign 10 times.
    for i in range(n):
        # Define model.
        base_model, model = make_ResNet50_relnet(
            dataset='SVRT',
            resnet_layer=resnet_layer,
            trainable=False)
        
        # Compile.
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.0003),
            loss='binary_crossentropy',
            metrics=['accuracy'])
        
        # Train.
        model.fit(
            train_ds,
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
        model.fit(
            train_ds,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,
            validation_steps=validation_steps)
        
        # Save weights.
        weights_name = save_name + 'model_' + str(i) + '.hdf5'
        model.save_weights(weights_name)
        
    return

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

def test_relnet_auc(ds, weights_dir, resnet_layer, model_name, condition):
    """Tests 10 versions of a single relation net model in a single condition using the area under the ROC curve. 
    Args:
        ds: dataset with cases from the 'same' and 'different' classes.
        weights_dir: directory of models weights to test.
        resnet_layer: layer of ResNet50 to use.
        model_name: model name. String.
        condition: condition name. String.
    Returns:
        A list with test data: model_name, condition, area under the ROC curve.
    """
    
    # Define model architecture.
    base_model, model = make_ResNet50_relnet(
        dataset='SVRT',
        resnet_layer=resnet_layer,
        trainable=False)
    
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

def test_all_relnets_auc(ds, condition):
    results = []
    
    # last_size_4.
    last_size_4_data = test_relnet_auc(
        ds=ds, 
        weights_dir='simulation_1/RN_4x4', 
        resnet_layer='last_size_4', 
        model_name='RN_4x4', 
        condition=condition)
    results.extend(last_size_4_data)
    
    # last_size_8.
    last_size_8_data = test_relnet_auc(
        ds=ds, 
        weights_dir='simulation_1/RN_8x8', 
        resnet_layer='last_size_8', 
        model_name='RN_8x8', 
        condition=condition)
    results.extend(last_size_8_data)
    
        
    # Make pandas dataframe and return.
    return pd.DataFrame(results, columns=['Model', 'Condition', 'AUC'])

def test_all_relnets_all_ds_auc(batch_size):
    # Load same/different datasets.
    datasets = []

    # Original.
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/original_test.tfrecords',
        is_training=False,
        process_img=True,
        relnet=True)
    datasets.append(ds)
    
    # Regular.
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/regular_test.tfrecords',
        is_training=False,
        process_img=True,
        relnet=True)
    datasets.append(ds)
    
    # Irregular.
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/irregular_test.tfrecords',
        is_training=False,
        process_img=True,
        relnet=True)
    datasets.append(ds)
    
    # Open.
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/open_test.tfrecords',
        is_training=False,
        process_img=True,
        relnet=True)
    datasets.append(ds)

    # Wider line.
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/wider_line_test.tfrecords',
        is_training=False,
        process_img=True,
        relnet=True)
    datasets.append(ds)
    
    # Scrambled.
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/scrambled_test.tfrecords',
        is_training=False,
        process_img=True,
        relnet=True)
    datasets.append(ds)
    
    # Random color.
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/random_color_test.tfrecords',
        is_training=False,
        process_img=True,
        relnet=True)
    datasets.append(ds)
    
    # Filled.
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/filled_test.tfrecords',
        is_training=False,
        process_img=True,
        relnet=True)
    datasets.append(ds)
    
    # Lines.
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/lines_test.tfrecords',
        is_training=False,
        process_img=True,
        relnet=True)
    datasets.append(ds)
    
    # Arrows.
    ds = get_dataset(
        batch_size=batch_size,
        tfrecord_dir='data/arrows_test.tfrecords',
        is_training=False,
        process_img=True,
        relnet=True)
    datasets.append(ds)
    
    conditions = ['Original', 'Regular', 'Irregular', 'Open', 'Wider Line', 'Scrambled', 'Random Color', 'Filled', 'Lines', 'Arrows']
    
    dfs = []
    for ds, condition in zip(datasets, conditions):
        df = test_all_relnets_auc(ds, condition)
        dfs.append(df)
    
    return pd.concat(dfs)

if __name__ == '__main__':
    # Training hyperparameters.
    EPOCHS_TOP = 5
    EPOCHS = 10
    BATCH_SIZE = 64
    STEPS_PER_EPOCH = 28000 // BATCH_SIZE
    VALIDATION_STEPS = 2800 // BATCH_SIZE

    # Get datasets.
    train_ds = get_dataset(
        batch_size=BATCH_SIZE,
        tfrecord_dir='data/original_train.tfrecords',
        is_training=True, 
        process_img=True,
        relnet=True)

    val_ds = get_dataset(
        batch_size=BATCH_SIZE,
        tfrecord_dir='data/original_val.tfrecords',
        is_training=False,
        process_img=True,
        relnet=True)

    # Train: imagenet, filter size 4x4.
    fine_tune_relnet(
        train_ds=train_ds,
        val_ds=val_ds,
        save_name='simulation_1/RN_4x4/',
        resnet_layer='last_size_4',
        epochs_top=EPOCHS_TOP,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        n=10,
        lr=0.0001,
        base_weights='imagenet',
        gap=True)
    
    # Train: imagenet, filter size 8x8.
    fine_tune_relnet(
        train_ds=train_ds,
        val_ds=val_ds,
        save_name='simulation_1/RN_8x8/',
        resnet_layer='last_size_8',
        epochs_top=EPOCHS_TOP,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        n=10,
        lr=0.0001,
        base_weights='imagenet',
        gap=True)

    df = test_all_relnets_all_ds_auc(batch_size=112)
    df.to_csv('simulation_1/sim_1_relnet_auc.csv')

print('All models trained and tested!')