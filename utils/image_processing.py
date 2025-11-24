import cv2
import numpy as np
from PIL import Image
import io
from config import SEGMENTATION_COLORS


def load_image_from_bytes(image_bytes):
    try:
        img_pil = Image.open(io.BytesIO(image_bytes))
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')
        return np.array(img_pil)
    except:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Не удалось загрузить изображение")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def label_to_rgb(predicted_image):
    colors = np.array(SEGMENTATION_COLORS, dtype=np.uint8)
    return colors[predicted_image]


def create_response_image(prediction_rgb):
    result_img = Image.fromarray(prediction_rgb)
    img_byte_arr = io.BytesIO()
    result_img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr


def rgb_to_2D_label(label):
    """
    Замена значений каждого пикселя маски в формате RGB на целое число
    """

    # Если маска трехмерная, оставляем только первые три канала (RGB)
    if len(label.shape) == 3:
        label = label[:,:,:3]
    # Если маска четырехмерная, оставляем только первые три канала (RGB) каждого изображения
    if len(label.shape) == 4:
        label = label[:,:,:,:3]

    # Создаем новую маску той же формы, что и исходная
    label_seg = np.zeros(label.shape,dtype=np.uint8)

    # Присваиваем каждому пикселю значение в зависимости от его цвета
    label_seg [np.all(label == COLOR_SCHEME['roads'],axis=-1)] = 0
    label_seg [np.all(label==COLOR_SCHEME['buildings'],axis=-1)] = 1
    label_seg [np.all(label==COLOR_SCHEME['low_veg'],axis=-1)] = 2
    label_seg [np.all(label==COLOR_SCHEME['trees'],axis=-1)] = 3
    label_seg [np.all(label==COLOR_SCHEME['cars'],axis=-1)] = 4
    label_seg [np.all(label==COLOR_SCHEME['clutter'],axis=-1)] = 5

    if len(label.shape) == 3:
        label_seg = label_seg[:,:,0]

    if len(label.shape) == 4:
        label_seg = label_seg[:,:,:,0]

    # Возвращаем преобразованную маску
    return label_seg

