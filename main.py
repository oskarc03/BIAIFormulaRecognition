import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.models import load_model
from PIL import Image
tf.config.list_physical_devices('GPU')

def fix_images(folder):
    for subdir, _, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(subdir, file)
            try:
                with Image.open(file_path) as img:
                    img = img.convert('RGB')
                    new_path = os.path.splitext(file_path)[0] + ".jpg"
                    img.save(new_path, 'JPEG')
                    if new_path != file_path:
                        os.remove(file_path)
            except Exception as e:
                print(f"Nie udało się naprawić: {file_path}, błąd: {e}")
                os.remove(file_path)


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

val_gen_pred = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

inputs = layers.Input(shape=(224, 224, 3), name='input_layer')

base_model = MobileNetV2(include_top=False)
base_model.trainable = False

x = base_model(inputs, training=False)
x=layers.GlobalAveragePooling2D()(x)
num_classes=len(class_names)
outputs=layers.Dense(num_classes,activation='softmax',dtype=tf.float32)(x)

model = Model(inputs, outputs)

#model.compile(
    #optimizer='adam',
    #loss='categorical_crossentropy',
    #metrics=['accuracy']
#)

#model.fit(train_gen, validation_data=val_gen, epochs=10)
#model.save("model/f1_model.h5")

#fine-tuning
base_model.trainable = True
#model.compile(
    #optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    #loss='categorical_crossentropy',
    #metrics=['accuracy']
#)
#model.fit(train_gen, validation_data=val_gen, epochs=10)
#model.save("model/f1_model_fine_tuned.h5")

#save data for later
#history_dict=model.history.history
#json.dump(history_dict, open('model/history_dict.json', 'w'))

trained_model = load_model("model/f1_model.h5")
trained_model.summary()

#loss and metrics
#results = trained_model.evaluate(val_gen)

with open("model/history_dict.json", "r") as f:
    history_dict_json = json.load(f)

y_pred_probs = trained_model.predict(val_gen_pred)
y_pred = np.argmax(y_pred_probs, axis=1)

y_true = val_gen_pred.classes
class_labels = list(val_gen_pred.class_indices.keys())


#basic plots    
model_acc_plt(history_dict_json)
model_loss_plt(history_dict_json)
conf_matrix(y_true, y_pred, class_labels)