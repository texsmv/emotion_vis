
import sys
sys.path.insert(1, '/home/texs/Documents/AirQuality/repositories/peax/experiments')
import h5py
import os
import logging
import numpy as np
from keras.models import Model, load_model
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tensorflow.contrib.tensorboard.plugins import projector
# from utils import createDir

def createDir(dirName):
    """Creates the directory if not already exists

    Args:
        dirName (str): directory path
    """
    if not os.path.isdir(dirName):
        os.makedirs(dirName)
        logging.info(f"Created output directory {dirName}")


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config));
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# LOG_DIR = "./tensorboard-logs"
IMAGE_SIZE = (45,45)
SPRITES_FILE = "sprites.png"
FEATURE_VECTORS = "feature_vectors.npy"
# METADATA_FILE = "metadata.tsv"
# METADATA_PATH = os.path.join(LOG_DIR, METADATA_FILE)

# MAX_NUMBER_SAMPLES = 40
# MAX_NUMBER_SAMPLES = 4000
MAX_NUMBER_SAMPLES = 692



def create_sprite(data):
    """
    Tile images into sprite image. 
    Add any necessary padding
    """
    
    # For B&W or greyscale images
    if len(data.shape) == 3:
        data = np.tile(data[...,np.newaxis], (1,1,1,3))

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0), (0, 0), (0, 0))
    data = np.pad(data, padding, mode='constant',
            constant_values=0)
    
    # Tile images into sprite
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3, 4))
    # print(data.shape) => (n, image_height, n, image_width, 3)
    
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    # print(data.shape) => (n * image_height, n * image_width, 3) 
    return data

def plot_all_windows(
    data: np.array,
    model_name: str,
    log_dir: str,
    base: str = ".",
    batch_size: int = 10,
    make_image = True,
):
    
    SPRITES_PATH = os.path.join(log_dir, SPRITES_FILE)
    CHECKPOINT_FILE = os.path.join(log_dir, "features.ckpt")

    createDir(log_dir)
    # np.random.shuffle(data)
    N, L, _ = data.shape
    print("N: {} L: {}".format(N, L))

    autoencoder_filepath = os.path.join(
        base, "savedData", model_name
    )
    autoencoder = load_model(autoencoder_filepath)
    encoder = Model(autoencoder.input, autoencoder.get_layer('embed').output)
    # encoder = Model(autoencoder.input, autoencoder.output)

    img_data = []
    features = []

    counter = 0
    for batch_start in np.arange(0, N, batch_size):
        print(counter)
        if (make_image) and (counter >= MAX_NUMBER_SAMPLES):
            break
        
        # batch_data = np.squeeze(
        #     f[data_type][batch_start: batch_start + batch_size], axis=2
        # )
        batch_data = data[batch_start: batch_start + batch_size]
        batch_data_reconstructed = autoencoder.predict(
            batch_data
        )
        batch_data_encoding = encoder.predict(
            batch_data
        )
        size = batch_data_encoding.shape[0]
        # for i in range(batch_size):
        for i in range(size):
        # print(batch_data_encoding.shape[0])
            if make_image and counter >= MAX_NUMBER_SAMPLES:
                break
            features = features + [batch_data_encoding[i]]
            if make_image:
                fig = plt.figure()
                alpha = 0.7
                plt.fill_between(list(range(L)), np.squeeze(batch_data[i], axis=1))
                plt.fill_between(list(range(L)), np.squeeze(
                    batch_data_reconstructed[i], axis=1), alpha=alpha)
                fig.canvas.draw()
                img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

                input_img_resize = cv2.resize(img, IMAGE_SIZE)
                img_data.append(input_img_resize)


                plt.clf()


            # plt.show()

            counter = counter + 1
    
    if make_image:
        img_data = np.array(img_data)
        sprite = create_sprite(img_data)
        cv2.imwrite(SPRITES_PATH, sprite)

    features = np.array(features)
    with open(os.path.join(log_dir + '/features.npy'), 'wb') as f:
        np.save(f,features)
    
    features = tf.Variable(features, name='features')
    with tf.Session() as sess:
        saver = tf.train.Saver([features])

        sess.run(features.initializer)
        saver.save(sess, CHECKPOINT_FILE)

        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = features.name
        # embedding.metadata_path = METADATA_FILE

        # This adds the sprite images
        embedding.sprite.image_path = SPRITES_FILE
        embedding.sprite.single_image_dim.extend(IMAGE_SIZE)
        projector.visualize_embeddings(tf.summary.FileWriter(log_dir), config)

if __name__ == "__main__":
    DATA_FILEPATH = f'savedData/data.npy'
    
    with open(DATA_FILEPATH, 'rb') as f:
        train_data = np.load(f, allow_pickle=True)
#         train_data_labels = np.load(f, allow_pickle=True)
        test_data = np.load(f, allow_pickle=True)
#         test_data_labels = np.load(f, allow_pickle=True)
        val_data = np.load(f, allow_pickle=True)
#         val_data_labels = np.load(f, allow_pickle=True)
    model_name = f"test.h5"
    plot_all_windows(
        train_data,
        model_name,
        log_dir = f"./log_test",
        base=".",
        batch_size=100,
        make_image=True,
    )