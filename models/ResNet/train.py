import tensorflow as tf
import sys

sys.path.append('.')
from custom_functions import *


if __name__ == "__main__":

    # Input constants used to identify correct dataset
    MATRIX_SIZE = 18 # 18 for unbalanced y60, 22 for y40, 36 for balanced y60
    NUM_CLASSES = 16 # 5 for y40, 16 for y60, 4 for balanced y60

    # Network hyperparameters
    HP_BATCH_SIZE = 128
    HP_ACTIVATION = 'relu'

    METRIC_ACCURACY = 'accuracy'

    # Training constants
    EPOCHS = 100
    CHECKPOINT = 100

    # Import dataset files and shuffle
    train_dataset, test_dataset = import_toric(path='datasets/y60', shuffle=1)

    # Make batches and enable prefetching for better performance
    train_dataset = train_dataset.batch(HP_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(HP_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    train_dataset_size = train_dataset.cardinality().numpy()

    # Build the ResNet model
    model = ResNet(input_shape=(MATRIX_SIZE,MATRIX_SIZE,9), num_classes=NUM_CLASSES, activation=HP_ACTIVATION)

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[METRIC_ACCURACY])
    model.build(input_shape=(MATRIX_SIZE,MATRIX_SIZE,9))
    model.summary()

    logdir = 'y60' + '_b' + str(HP_BATCH_SIZE) + '_ac' + HP_ACTIVATION

    # Define callbacks
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir= 'models/ResNet/logs/' + logdir + '/',
        histogram_freq=1,
        write_images=False
    )
    checkpoint_path = 'models/ResNet/checkpoints/' + logdir + '_e{epoch:04d}.weights.h5'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_freq=int(CHECKPOINT*train_dataset_size),
        verbose=1,
        save_weights_only=True
    )

    latest_epoch = checkpoint_loader(path='models/ResNet/checkpoints/'+logdir, model=model)

    model.fit(
        train_dataset,
        epochs=EPOCHS,
        batch_size=HP_BATCH_SIZE,
        callbacks=[tensorboard_callback,checkpoint_callback],
        validation_data=test_dataset,
        initial_epoch=latest_epoch)
    model.save('models/ResNet/models/' + logdir + '_e' + str(EPOCHS) + '.keras')