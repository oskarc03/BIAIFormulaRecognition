import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, Model, Input
from PIL import Image

def fix_images(folder):
    for subdir, _, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(subdir, file)
            try:
                with Image.open(file_path) as img:
                    img = img.convert('RGB')  # usunięcie przezroczystości itp.
                    new_path = os.path.splitext(file_path)[0] + ".jpg"
                    img.save(new_path, 'JPEG')
                    if new_path != file_path:
                        os.remove(file_path)  # usuń oryginał jeśli zmieniono rozszerzenie
            except Exception as e:
                print(f"Nie udało się naprawić: {file_path}, błąd: {e}")
                os.remove(file_path)  # opcjonalnie: usuń tylko gdy całkowicie nieczytelny


DATA_DIR = "Formula One Cars"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
#fix_images(DATA_DIR)
datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)
class_names = train_gen.class_indices


val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

inputs = layers.Input(shape=(224, 224, 3), name='input_layer')

base_model = MobileNetV2(include_top=False)
base_model.trainable = False

x = base_model(inputs, training=False)
x=layers.GlobalAveragePooling2D()(x)
num_classes=len(class_names)
outputs=layers.Dense(num_classes,activation='softmax',dtype=tf.float32)(x)

model = Model(inputs, outputs)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_gen, validation_data=val_gen, epochs=5)
model.save("model/f1_model.h5")