import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, Model, Input, models
from PIL import Image
from main import model_acc_plt, model_loss_plt, conf_matrix
import json
import numpy as np

DATA_DIR = "Formula One Cars"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)

train_gen = tf.keras.utils.image_dataset_from_directory( 
	DATA_DIR, 
	validation_split=0.2, 
	subset="training", 
	seed=123, 
	image_size=(224, 224), 
	batch_size=32) 

class_names = train_gen.class_names


val_gen = tf.keras.utils.image_dataset_from_directory( 
	DATA_DIR, 
	validation_split=0.2, 
	subset="validation", 
	seed=123, 
	image_size=(224, 224), 
	batch_size=32) 

val_gen_pred = tf.keras.utils.image_dataset_from_directory( 
	DATA_DIR, 
	validation_split=0.2, 
	subset="validation", 
	seed=123, 
	image_size=(224, 224), 
	batch_size=32,
    shuffle = False) 

num_classes=len(class_names)


model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(224, 224, 3)),
    
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(224, activation='relu'),
    #layers.Dropout(0.5),
    layers.Dense(num_classes)
])

model.compile(optimizer='adam', 
			loss=tf.keras.losses.SparseCategoricalCrossentropy( 
				from_logits=True), 
			metrics=['accuracy']) 
model.summary()

history = model.fit(train_gen, validation_data=val_gen, epochs=10)
model.save("model/my_f1_model.h5")

history_dict = model.history.history
json.dump(history_dict, open('model/my_model_history_dict.json', 'w'))

y_pred_probs = model.predict(val_gen_pred)
y_pred = y_pred = np.argmax(y_pred_probs, axis=1)
y_true = val_gen_pred.class_names
class_labels = list(val_gen_pred.class_indices.keys())

model_acc_plt(history_dict)
model_loss_plt(history_dict)
conf_matrix(y_true, y_pred, class_labels)