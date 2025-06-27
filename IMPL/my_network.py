import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, Model, Input, models
from PIL import Image
import matplotlib.pyplot as plt
#from main import model_acc_plt, model_loss_plt, conf_matrix
import json
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
from sklearn.utils.class_weight import compute_class_weight

DATA_DIR = "Formula One Cars"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)

def model_acc_plt(history_dict):
    plt.plot(history_dict['accuracy'], label='train accuracy')
    plt.plot(history_dict['val_accuracy'], label='val accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Model Accuracy')
    plt.show()

def model_loss_plt(history_dict):
    plt.plot(history_dict['loss'], label='train loss')
    plt.plot(history_dict['val_loss'], label='val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Model Loss')
    plt.show()

def conf_matrix(y_true, y_pred, class_labels_show):
    cm=confusion_matrix(y_true,y_pred)
    plt.figure(figsize=(14,10))
    sns.heatmap(cm,annot=True,cmap='Blues',fmt='.0f')
    plt.ylabel("True values",size=15)
    plt.xlabel('Predicted values',size=15)
    plt.xticks(ticks=np.arange(len(class_labels_show))+0.5,labels=class_labels_show,rotation=60)
    plt.yticks(ticks=np.arange(len(class_labels_show))+0.5,labels=class_labels_show,rotation=0)
    plt.show()

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

val_gen_pred = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# val_gen_pred = datagen.flow_from_directory(
#     DATA_DIR,
#     target_size=IMAGE_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     subset='validation',
#     shuffle=False
# ) 

num_classes=len(class_names)


model = models.Sequential([    
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.GlobalAveragePooling2D(),
    layers.Dense(224, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# class_weights = compute_class_weight(
#     class_weight='balanced',
#     classes=np.unique(train_gen.classes),
#     y=train_gen.classes
# )
# class_weights = dict(enumerate(class_weights))

# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
# 			loss='categorical_crossentropy', 
# 			metrics=['accuracy']) 
# model.summary()

# history = model.fit(train_gen, validation_data=val_gen, epochs=30, class_weight=class_weights)
# model.save("model/my_f1_model.h5")
# #model.trainable = True
# history_dict = model.history.history
# json.dump(history_dict, open('model/my_model_history_dict.json', 'w'))

loaded_model = load_model("model/my_f1_model.h5")
loaded_model.summary()
with open("model/my_model_history_dict.json", "r") as f:
    history_dict_json = json.load(f)
    
y_pred_probs = loaded_model.predict(val_gen_pred)
y_pred = y_pred = np.argmax(y_pred_probs, axis=1)
y_true = val_gen.classes
class_labels = list(val_gen.class_indices.keys())

model_acc_plt(history_dict_json)
model_loss_plt(history_dict_json)
conf_matrix(y_true, y_pred, class_labels)

