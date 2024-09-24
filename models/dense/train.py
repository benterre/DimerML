import tensorflow as tf
import sys

sys.path.append('.')
from custom_functions import *


if __name__ == "__main__":

    # Input constants used to identify correct dataset
    MATRIX_SIZE = 128
    ITERATIONS = 1000

    # Network hyperparameters
    HP_DENSE_SHAPE = 64
    HP_BATCH_SIZE = 64
    HP_ACTIVATION = 'relu'

    METRIC_ACCURACY = 'mae'

    # Training constants
    EPOCHS = 100
    CHECKPOINT = 1

    train_dataset, test_dataset = import_dataset(MATRIX_SIZE, ITERATIONS, path='datasets')

    # Make batches and enable prefetching for better performance
    train_dataset = train_dataset.batch(HP_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(HP_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    train_dataset_size = train_dataset.cardinality().numpy()

    # Define model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(MATRIX_SIZE,MATRIX_SIZE,9)),
        RandomFlipLayer(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(HP_DENSE_SHAPE, activation=HP_ACTIVATION),
        tf.keras.layers.Dense(2)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=[METRIC_ACCURACY])
    model.build()
    model.summary()

    logdir = 'ms' + str(MATRIX_SIZE) + '_b' + str(HP_BATCH_SIZE) + '_ds' + str(HP_DENSE_SHAPE) + '_ac' + HP_ACTIVATION

    # Define callbacks
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir= 'models/dense/logs/' + logdir + '/',
        histogram_freq=1,
        write_images=False
    )
    checkpoint_path = 'models/dense/checkpoints/' + logdir + '_e{epoch:04d}.weights.h5'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_freq=int(CHECKPOINT*train_dataset_size),
        verbose=1,
        save_weights_only=True
    )

    latest_epoch = checkpoint_loader(path='models/dense/checkpoints/'+logdir, model=model)

    model.fit(
        train_dataset,
        epochs=EPOCHS,
        batch_size=HP_BATCH_SIZE,
        callbacks=[tensorboard_callback,checkpoint_callback],
        validation_data=test_dataset,
        initial_epoch=latest_epoch)