import tensorflow as tf
import os
import glob
import re
import numpy as np


# ----- dataset format conversion -----
# Used in tensorflow's dataset.map() function
def sparse2dense(sparse):
    return tf.sparse.to_dense(sparse)

def labelformatting_dense(matrices, labels):
    # Sorting labels in order to match (m,n) with (n,m)
    return tf.sparse.to_dense(matrices), tf.sort(tf.sparse.to_dense(labels)[:2])

def labelformatting_cnn(matrices, labels):
    labels = tf.sort(tf.sparse.to_dense(labels))
    lb0 = labels[0]
    lb1 = labels[1]
    return tf.sparse.to_dense(matrices), tf.convert_to_tensor(lb0*41+lb1) # 41 corresponds to the maximal m or n label+1

def labelformatting_toric(labels):
    return labels # change if needed
# -------------------------------------

# ----- custom keras layers/metrics -----
@tf.function
def flipTensorComponents(tensor):
    #reorder rows
    tensor = tf.random.shuffle(tensor)
    #reorder columns
    tensor = tf.transpose(tensor,perm=[1,0,2])
    tensor = tf.random.shuffle(tensor)
    #reinstate original component ordering
    tensor = tf.transpose(tensor,perm=[1,0,2])

    # wz[0] -> exchange z <-> w if wz[0] > 0
    # wz[1] -> exchange z <-> 1/z if wz[1] > 0
    # wz[2] -> exchange w <-> 1/w if wz[2] > 0
    wz = tf.random.uniform((1,3), minval=-1, maxval=3, dtype=tf.dtypes.int32)[0] #WARNING dtype.float causes wz to remain unchanged throughout training
    
    if wz[0] > 0: # w <-> z
        w_mat = tensor[:,:,1]
        winv_mat = tensor[:,:,2]
        z_mat = tensor[:,:,3]
        zinv_mat = tensor[:,:,4]
        wz_mat = tensor[:,:,5]
        zwinv_mat = tensor[:,:,6]
        wzinv_mat = tensor[:,:,7]
        winvzinv_mat = tensor[:,:,8]

        tensor = tf.concat([
            tensor[:,:,:1],
            z_mat[:,:,tf.newaxis],
            zinv_mat[:,:,tf.newaxis],
            w_mat[:,:,tf.newaxis],
            winv_mat[:,:,tf.newaxis],
            wz_mat[:,:,tf.newaxis],
            wzinv_mat[:,:,tf.newaxis],
            zwinv_mat[:,:,tf.newaxis],
            winvzinv_mat[:,:,tf.newaxis]
        ], axis=2)
    if wz[1] > 0: # z <-> 1/z
        w_mat = tensor[:,:,1]
        winv_mat = tensor[:,:,2]
        z_mat = tensor[:,:,3]
        zinv_mat = tensor[:,:,4]
        wz_mat = tensor[:,:,5]
        zwinv_mat = tensor[:,:,6]
        wzinv_mat = tensor[:,:,7]
        winvzinv_mat = tensor[:,:,8]

        tensor = tf.concat([
            tensor[:,:,:1],
            w_mat[:,:,tf.newaxis],
            winv_mat[:,:,tf.newaxis],
            zinv_mat[:,:,tf.newaxis],
            z_mat[:,:,tf.newaxis],
            wzinv_mat[:,:,tf.newaxis],
            winvzinv_mat[:,:,tf.newaxis],
            wz_mat[:,:,tf.newaxis],
            zwinv_mat[:,:,tf.newaxis]
        ], axis=2)
    if wz[2] > 0: # w <-> 1/w
        w_mat = tensor[:,:,1]
        winv_mat = tensor[:,:,2]
        z_mat = tensor[:,:,3]
        zinv_mat = tensor[:,:,4]
        wz_mat = tensor[:,:,5]
        zwinv_mat = tensor[:,:,6]
        wzinv_mat = tensor[:,:,7]
        winvzinv_mat = tensor[:,:,8]

        tensor = tf.concat([
            tensor[:,:,:1],
            winv_mat[:,:,tf.newaxis],
            w_mat[:,:,tf.newaxis],
            z_mat[:,:,tf.newaxis],
            zinv_mat[:,:,tf.newaxis],
            zwinv_mat[:,:,tf.newaxis],
            wz_mat[:,:,tf.newaxis],
            winvzinv_mat[:,:,tf.newaxis],
            wzinv_mat[:,:,tf.newaxis],
        ], axis=2)
    return tensor

@tf.keras.utils.register_keras_serializable()
class RandomFlipLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(RandomFlipLayer, self).__init__()

    def call(self, inputs, training=None):
        if training:
            inputs = tf.map_fn(flipTensorComponents, inputs)
        return inputs

@tf.function
def flipTensorComponentsNoWZ(tensor):
    #reorder rows
    tensor = tf.random.shuffle(tensor)
    #reorder columns
    tensor = tf.transpose(tensor,perm=[1,0,2])
    tensor = tf.random.shuffle(tensor)
    #reinstate original component ordering
    tensor = tf.transpose(tensor,perm=[1,0,2])
    return tensor

@tf.keras.utils.register_keras_serializable()
class RandomFlipLayerNoWZ(tf.keras.layers.Layer):
    def __init__(self):
        super(RandomFlipLayerNoWZ, self).__init__()

    def call(self, inputs, training=None):
        if training:
            inputs = tf.map_fn(flipTensorComponentsNoWZ, inputs)
        return inputs

@tf.keras.utils.register_keras_serializable()
class EvaluateHolesCallback(tf.keras.callbacks.Callback):
    def __init__(self, holes_dataset, logdir):
        super(EvaluateHolesCallback, self).__init__()
        self.holes = holes_dataset
        self.logdir = logdir
        self.test_summary_writer = tf.summary.create_file_writer(logdir)

    def on_epoch_end(self, epoch, logs=None):
        # Evaluate the model on the excised dataset
        loss, mae = self.model.evaluate(self.holes, verbose=0)
        tf.print("Holes loss: ", loss, "Holes mae: ", mae)
        
        # Log the accuracy to TensorBoard
        with self.test_summary_writer.as_default():
            tf.summary.scalar('Holes_mae', mae, step=epoch)
            tf.summary.scalar('Holes_loss', loss, step=epoch)
    
    def get_config(self):
        return {
            "holes": self.holes,
            "logdir": self.logdir,
            "test_summary_writer": self.test_summary_writer
        }

@tf.keras.utils.register_keras_serializable()
class R2(tf.keras.metrics.Metric):
    def __init__(self, name='r2', **kwargs):
        super(R2, self).__init__(name=name, **kwargs)
        self.sse = self.add_weight(name='sse', initializer='zeros')
        self.sst = self.add_weight(name='sst', initializer='zeros')
        self.mean = self.add_weight(name='mean', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        error = y_true - y_pred
        self.sse.assign_add(tf.reduce_sum(tf.square(error)))

        mean = tf.reduce_mean(y_true)
        self.sst.assign_add(tf.reduce_sum(tf.square(y_true - mean)))

    def result(self):
        return 1 - (self.sse / (self.sst + tf.keras.backend.epsilon()))

    def reset_states(self):
        self.sse.assign(0.0)
        self.sst.assign(0.0)
        self.mean.assign(0.0)
        self.count.assign(0.0)
    
    def get_config(self):
        config = super(R2, self).get_config()
        config.update({
            "name": self.name
        })
        return config

@tf.keras.utils.register_keras_serializable()
class SkipLayer(tf.keras.layers.Layer):
    def __init__(self, resize):
        super(SkipLayer, self).__init__()
        self.resize = resize

    def call(self, inputs):
        if self.resize == 0:
            # keep skip layer shape unchanged if size difference is zero
            return inputs
        elif self.resize > 0:
            # pad skip layer shape if size difference is positive
            return tf.pad(inputs, [[0,0,],[0,0,],[0,0,],[0,self.resize,]])
        else:
            # trim skip layer shape if size difference is negative
            return inputs[:,:,:,:-self.resize]

    def get_config(self):
        return {
            "resize": self.resize
        }
# ---------------------------------------

# ----- ResNet definitions -----
def residual_block(input_tensor, filters, resize, activation):
    # Initial skip layer
    skip = SkipLayer(resize=resize)(input_tensor)

    # Two conv blocks in parallel to the skip layer
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Recombine skip layer with cnn blocks
    x = tf.keras.layers.Add()([x, skip])
    x = tf.keras.layers.Activation(activation)(x)

    return x

def ResNet(input_shape, num_classes, shape=[64,64,64], activation='relu', noWZ=False):
    input_tensor = tf.keras.layers.Input(shape=input_shape)
    
    # Random Flip Layer
    if noWZ:
        x = RandomFlipLayerNoWZ()(input_tensor)
    else:
        x = RandomFlipLayer()(input_tensor)

    # Initial Conv Layer
    x = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Create residual layers
    prev_size = 64
    for size in shape:
        resize = size - prev_size
        x = residual_block(x, size, resize, activation=activation)
        prev_size = size

    # Global Average Pooling and Output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    output_tensor = tf.keras.layers.Dense(num_classes)(x)

    # Create the model
    model = tf.keras.models.Model(inputs=input_tensor, outputs=output_tensor)

    return model
# ------------------------------

# ----- checkpoint functions -----
def extract_epoch_number(filename):
    # Read filename and find epoch number in string
    match = re.search(r'_e(\d+)', filename)
    if match:
        return int(match.group(1))  # Return the epoch number as an integer
    return -1

def checkpoint_loader(path, model):
    checkpoint_path = path + '_e{epoch:04d}.weights.h5'
    weight_files = glob.glob(path + '*.weights.h5')
    # Sort files by the extracted epoch number
    sorted_weight_files = sorted(weight_files, key=extract_epoch_number)

    # Load the latest weights file (last one in the sorted list)
    if sorted_weight_files:
        latest_weights = sorted_weight_files[-1]
        latest_epoch = extract_epoch_number(latest_weights)
        print(f"Loading checkpoint file: {latest_weights}")
        model.load_weights(latest_weights)
    else:
        latest_epoch = 0
        print("No checkpoint files found. Starting from epoch 0")
        model.save_weights(checkpoint_path.format(epoch=0))
    return latest_epoch
# --------------------------------

# ----- dataset import -----
def import_toric(path='datasets/y60', shuffle=1):

    dataset_data = None
    
    dataset_sparse = tf.data.Dataset.load(path + "/mat", compression="GZIP")
    dataset_data = tf.data.Dataset.load(path + "/dat", compression="GZIP")

    print("Dataset loaded with " + str(dataset_sparse.cardinality().numpy()) + " elements")

    # Format data
    dataset_dense = dataset_sparse.map(sparse2dense)
    dataset_data = dataset_data.map(labelformatting_toric) # POTENTIALLY CHANGE THIS

    dataset = tf.data.Dataset.zip(dataset_dense, dataset_data)

    # Determine the size of the dataset
    dataset_size = dataset.cardinality().numpy()
    train_size = int(0.8 * dataset_size)

    # Shuffle the dataset
    dataset = dataset.shuffle(buffer_size=int(dataset_size*shuffle))

    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)

    return train_dataset, test_dataset

def import_dataset_path(path, ms, it, folder):
    return tf.data.Dataset.load(path + "/ms" + str(ms) + "_it" + str(it) + "/" + folder, compression="GZIP")

def import_dataset(MATRIX_SIZE=64, ITERATIONS=1000, path='datasets', num_classes=None, hole_size=0, keep_sparse=False):
    # Initialise datasets
    dataset_train = None
    dataset_validation = None

    if hole_size > 0:
        # Initialize excised dataset
        dataset_holes = None
    
    # Loop through all dataset folders and append them
    for folder in os.listdir(path + "/ms" + str(MATRIX_SIZE) + "_it" + str(ITERATIONS)):
        try:
            M = int(folder.split("_")[0])
            N = int(folder.split("_")[1])
        except:
            M = 0
            N = 0
        
        # Load excised dataset
        if M-N < hole_size and M-N > -hole_size:
            if dataset_holes == None:
                dataset_holes = import_dataset_path(path, MATRIX_SIZE, ITERATIONS, folder)
            else:
                dataset_holes = dataset_holes.concatenate(import_dataset_path(path, MATRIX_SIZE, ITERATIONS, folder))
        else:
            # Load training dataset
            if "train" in folder:
                if dataset_train == None:
                    dataset_train = import_dataset_path(path, MATRIX_SIZE, ITERATIONS, folder)
                else:
                    dataset_train = dataset_train.concatenate(import_dataset_path(path, MATRIX_SIZE, ITERATIONS, folder))
            # Load validation dataset
            elif "validation" in folder:
                if dataset_validation == None:
                    dataset_validation = import_dataset_path(path, MATRIX_SIZE, ITERATIONS, folder)
                else:
                    dataset_validation = dataset_validation.concatenate(import_dataset_path(path, MATRIX_SIZE, ITERATIONS, folder))

    print("Training dataset loaded with " + str(dataset_train.cardinality().numpy()) + " elements")
    print("Validation dataset loaded with " + str(dataset_validation.cardinality().numpy()) + " elements")

    # Format labels to either remove depth or convert to unique integer
    # Both convert the dataset to tensorflow dense tensor
    if not keep_sparse:
        if num_classes:
            dataset_train = dataset_train.map(labelformatting_cnn)
            dataset_validation = dataset_validation.map(labelformatting_cnn)
        else:
            dataset_train = dataset_train.map(labelformatting_dense)
            dataset_validation = dataset_validation.map(labelformatting_dense)

    # Format labels to either remove depth or convert to unique integer for the excised dataset
    # Both convert the dataset to tensorflow dense tensor
    if hole_size > 0:
        print("Holes loaded with " + str(dataset_holes.cardinality().numpy()) + " elements")
        if not keep_sparse:
            if num_classes:
                dataset_holes = dataset_holes.map(labelformatting_cnn)
            else:
                dataset_holes = dataset_holes.map(labelformatting_dense)

        return dataset_train, dataset_validation, dataset_holes
    else:
        return dataset_train, dataset_validation
# --------------------------

if __name__ == "__main__":
    a, b, c = import_dataset(MATRIX_SIZE=128, ITERATIONS=1000, path='datasets', hole_size=10)
    print(a.cardinality().numpy(), b.cardinality().numpy(), c.cardinality().numpy())