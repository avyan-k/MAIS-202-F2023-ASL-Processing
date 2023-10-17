
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import opendatasets as od
import matplotlib.pyplot as plt # for testing

def load_data():
    if not os.path.exists(r"synthetic-asl-alphabet"):
        od.download("https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet/data")

def load_data_to_tensor() -> (tf.data.Dataset,tf.data.Dataset):
    load_data()
    train_ds = tf.keras.utils.image_dataset_from_directory(
    r"synthetic-asl-alphabet/Train_Alphabet",
    seed=1209,
    image_size=(513, 512),
    batch_size=32) 
    test_ds = tf.keras.utils.image_dataset_from_directory(
    r"synthetic-asl-alphabet/Test_Alphabet",
    seed=1209,
    image_size=(513, 512),
    batch_size=32) 
    return test_ds,train_ds

cache = dict()

def get_ds(ds):
    print("Getting dataset")
    if ds not in cache:
        cache[ds] = (ds)
    return cache[ds]

def show_images(dataset : tf.data.Dataset) -> None :
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(dataset.class_names[labels[i]])
            plt.axis("off")
    plt.show()

show_images(load_data_to_tensor()[0])

testing,training=load_data_to_tensor()
training_cache=get_ds(training)