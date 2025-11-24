import tensorflow.keras.backend as K
import tensorflow as tf


# ============================================================
# Расчет весов классов для борьбы с дисбалансом
# ============================================================
def calculate_class_weights(class_percentages):
    """
    Рассчитывает веса классов обратно пропорционально их частоте
    
    Args:
        class_percentages: словарь {class_id: percentage}
        Например: {0: 28.64, 1: 27.32, 2: 23.85, 3: 12.62, 4: 1.80, 5: 5.78}
    
    Returns:
        dict: словарь весов {class_id: weight}
    
    Формула: weight = total / (n_classes * class_percentage)
    Это дает больший вес редким классам (например, cars: 1.80%)
    """
    total = sum(class_percentages.values())
    class_weights = {}
    
    print("📊 Распределение классов из EDA:")
    for class_id, pct in class_percentages.items():
        print(f"  Класс {class_id}: {pct}%")
    
    for class_id, pct in class_percentages.items():
        # Формула: weight = total / (n_classes * class_percentage)
        weight = total / (len(class_percentages) * pct)
        class_weights[class_id] = weight
    
    print("\n⚖️ Рассчитанные веса классов:")
    class_names = ['roads', 'buildings', 'low_veg', 'trees', 'cars', 'clutter']
    for class_id, weight in class_weights.items():
        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        print(f"  {class_name:12} (класс {class_id}): {weight:.3f}")
    
    print(f"\n🎯 Класс 'cars' получил вес {class_weights[4]:.3f} (в ~{class_weights[4]/class_weights[0]:.1f}x больше чем roads)")
    print("   Это заставит модель уделять больше внимания редкому классу!")
    
    return class_weights


# Используем результаты анализа из EDA:
# roads: 28.64%, buildings: 27.32%, low_veg: 23.85%,
# trees: 12.62%, clutter: 5.78%, cars: 1.80%
CLASS_PERCENTAGES = {
    0: 28.64,  # roads
    1: 27.32,  # buildings
    2: 23.85,  # low_veg
    3: 12.62,  # trees
    4: 1.80,   # cars - РЕДКИЙ КЛАСС!
    5: 5.78    # clutter
}

# Рассчитываем веса классов
class_weights = calculate_class_weights(CLASS_PERCENTAGES)


# Функция для вычисления коэффициента Жаккара (IoU)
def jacard_coef(y_true, y_pred):
    """
    Jaccard coefficient (IoU) - метрика для оценки качества сегментации
    """
    # Преобразование меток в одномерные массивы
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    # Вычисление пересечения между истинными и предсказанными метками
    intersection = K.sum(y_true_f * y_pred_f)

    # Вычисление коэффициента Жаккара
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

# Weighted Categorical Crossentropy Loss
def weighted_categorical_crossentropy(class_weights_dict):
    """
    Создает взвешенную categorical crossentropy loss функцию
    Веса классов встроены в функцию потерь для работы с генераторами
    """
    # Конвертируем словарь в тензор (явно float32)
    weights = tf.constant([class_weights_dict[i] for i in range(len(class_weights_dict))], dtype=tf.float32)

    def loss(y_true, y_pred):
        # Приводим все к float32 для согласованности типов
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Применяем веса к каждому классу
        weights_per_pixel = tf.reduce_sum(y_true * weights, axis=-1)

        # Categorical crossentropy с numerical stability
        epsilon = tf.constant(K.epsilon(), dtype=tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        crossentropy = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)

        # Применяем веса
        weighted_crossentropy = crossentropy * weights_per_pixel

        return tf.reduce_mean(weighted_crossentropy)

    return loss

# Создаем взвешенную loss функцию с нашими весами
weighted_loss = weighted_categorical_crossentropy(class_weights)
