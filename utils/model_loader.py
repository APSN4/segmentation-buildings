from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

from matplotlib import pyplot as plt
from patchify import patchify
from PIL import Image
import segmentation_models as sm

from keras.metrics import MeanIoU

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()

from utils.image_processing import rgb_to_2D_label

seed=24
batch_size=16
n_classes=6

# Определяем функцию-генератор для обучения
def trainGenerator(train_img_path, train_mask_path, num_class):

    # Задаем аргументы для генератора данных: включаем горизонтальное и вертикальное отражение
    img_data_gen_args = dict(horizontal_flip=True,
                      vertical_flip=True)

    # Создаем генераторы данных для изображений и масок с заданными аргументами
    image_datagen = ImageDataGenerator(**img_data_gen_args)
    mask_datagen = ImageDataGenerator(**img_data_gen_args)

    # Генераторы читают изображения и маски из указанных директорий
    image_generator = image_datagen.flow_from_directory(
        train_img_path,
        class_mode = None,
        batch_size = batch_size,
        color_mode = 'rgb',
        target_size=(256, 256),
        seed = seed)

    mask_generator = mask_datagen.flow_from_directory(
        train_mask_path,
        class_mode = None,
        batch_size = batch_size,
        color_mode = 'rgb',
        target_size=(256, 256),
        seed = seed)

    # Объединяем генераторы изображений и масок в один генератор
    train_generator = zip(image_generator, mask_generator)

    # Для каждой пары (изображение, маска) выполняем предварительную обработку и возвращаем результат
    for (img, mask) in train_generator:
        img, mask = preprocess_data(img, mask, num_class)
        yield (img, mask)


# Импортируем MinMaxScaler из библиотеки sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler
# Импортируем функцию to_categorical из библиотеки keras.utils
from keras.utils import to_categorical

# Создаем объект MinMaxScaler для нормализации данных
scaler = MinMaxScaler()

# Задаем архитектуру сети для передачи обучения
BACKBONE = 'resnet34'

# Получаем функцию предварительной обработки для выбранной архитектуры сети
preprocess_input = sm.get_preprocessing(BACKBONE)

# Определяем функцию для предварительной обработки данных
def preprocess_data(img, mask, num_class):
    # Масштабируем пиксели изображения в диапазон от 0 до 1
    img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)

    # Применяем функцию предварительной обработки к изображению
    img = preprocess_input(img)

    # Приводим изображение к float32
    img = img.astype('float32')

    # Преобразуем маску из формата RGB в 2D-метку (256, 256, 3) -> (256, 256, 1)
    mask = rgb_to_2D_label(mask)

    # Преобразуем 2D-метку в бинарную матрицу (256, 256, 1) -> (256, 256, 6)
    mask = to_categorical(mask, num_class)

    # Приводим маску к float32
    mask = mask.astype('float32')

    # Возвращаем предварительно обработанное изображение и маску
    return (img, mask)
