import os
os.environ['SM_FRAMEWORK'] = 'tf.keras'

import cv2
import keras
import numpy as np
from matplotlib import pyplot as plt
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
import segmentation_models as sm

from config import APP_TITLE, CORS_CONFIG
from utils.model_loader import scaler, trainGenerator
from utils.prediction import jacard_coef, weighted_loss
from utils.image_processing import load_image_from_bytes, label_to_rgb, create_response_image, rgb_to_2D_label

app = FastAPI(title=APP_TITLE)
app.add_middleware(CORSMiddleware, **CORS_CONFIG)

# Загружаем модель
model = keras.models.load_model(
    'models/best_segmentation_model.h5',
    custom_objects={
        'jacard_coef': jacard_coef,
        'loss': weighted_loss
    }
)


@app.get("/")
def read_root():
    return {"сообщение": "API модели сегментации работает"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Загружаем изображение из байтов
        contents = await file.read()
        img = load_image_from_bytes(contents)
        
        # Проверка размера изображения
        max_pixels = 4096 * 4096
        if img.shape[0] * img.shape[1] > max_pixels:
            raise HTTPException(status_code=400, detail="Изображение слишком большое")

        # Приводим изображение к нужному размеру (256x256)
        img_resized = cv2.resize(img, (256, 256))

        # Предобработка изображения (как в генераторе)
        img_processed = scaler.fit_transform(img_resized.reshape(-1, img_resized.shape[-1])).reshape(img_resized.shape)

        # Применяем preprocessing для ResNet34
        try:
            preprocess_input = sm.get_preprocessing('resnet34')
            img_processed = preprocess_input(img_processed)
        except:
            pass

        # Добавляем batch dimension
        img_processed = np.expand_dims(img_processed, axis=0)

        # Получаем предсказание модели
        y_pred = model.predict(img_processed, verbose=0)
        mask_pred = np.argmax(y_pred[0], axis=-1)

        # Конвертируем маску в цветное RGB изображение
        prediction_rgb = label_to_rgb(mask_pred)
        
        # Возвращаем цветное изображение через API
        response_image = create_response_image(prediction_rgb)
        return StreamingResponse(response_image, media_type="image/png")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
