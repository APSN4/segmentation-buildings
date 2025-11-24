BACKBONE = 'resnet34'
DEFAULT_MODEL_PATH = 'models/best_segmentation_model.h5'
ENV_CONFIG = {'SM_FRAMEWORK': 'tf.keras'}

APP_TITLE = "API модели сегментации"

CORS_CONFIG = {
    "allow_origins": ["*"],
    "allow_credentials": True,
    "allow_methods": ["*"],
    "allow_headers": ["*"],
}

CLASS_NAMES = ['roads', 'buildings', 'low_veg', 'trees', 'cars', 'clutter']

CLASS_NAMES_RU = [
    'Дороги',
    'Здания',
    'Низкая растительность',
    'Деревья',
    'Машины',
    'Прочее'
]

SEGMENTATION_COLORS = [
    [255, 255, 255],  # roads
    [0, 0, 255],      # buildings
    [0, 255, 255],    # low_veg
    [0, 255, 0],      # trees
    [255, 255, 0],    # cars
    [255, 0, 0]       # clutter
]

DEFAULT_SUBDIVISIONS = 2
DEFAULT_NUM_CLASSES = 6
