import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, ResNet101
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
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 10))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', 
                xticklabels=class_labels_show, 
                yticklabels=class_labels_show)
    plt.ylabel("True Labels", fontsize=14)
    plt.xlabel("Predicted Labels", fontsize=14)
    plt.title("Confusion Matrix", fontsize=16)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def predict_image(image_path, model, class_labels):
    from tensorflow.keras.preprocessing import image
    
    img = image.load_img(image_path, target_size=(224, 224))
    
    img_array = image.img_to_array(img)
    
    img_array = img_array / 255.0
    
    img_batch = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_batch)
    predicted_index = np.argmax(predictions)
    predicted_label = class_labels[predicted_index]
    confidence = predictions[0][predicted_index]
    
    return predicted_label, confidence


def show_prediction(image_path, model, class_labels):
    label, confidence = predict_image(image_path, model, class_labels)
    
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title(f"{label} ({confidence*100:.1f}%)")
    plt.axis('off')
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

base_model = ResNet101(include_top=False, classes=len(class_names))
base_model.trainable = False

x = base_model(inputs, training=False)
x=layers.GlobalAveragePooling2D()(x)
num_classes=len(class_names)
outputs=layers.Dense(num_classes,activation='softmax',dtype=tf.float32)(x)

model = Model(inputs, outputs)
print("TEST123123")
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# model.fit(train_gen, validation_data=val_gen, epochs=5)
# model.save("model/f1_model_ResNet101.h5")
# print("model saved")
#fine-tuning
print("=======FINE TUNING=======")
base_model.trainable = True
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )
# model.fit(train_gen, validation_data=val_gen, epochs=10)
# model.save("model/f1_model_ResNet101_fine_tuned.h5")

# #save data for later
# history_dict=model.history.history
# json.dump(history_dict, open('model/history_dict.json', 'w'))

#trained_model = load_model("model/f1_model_MobileNetV2.h5")
#trained_model.summary()

# #loss and metrics
# #results = trained_model.evaluate(val_gen)

with open("model/history_dict.json", "r") as f:
    history_dict_json = json.load(f)

# y_pred_probs = trained_model.predict(val_gen_pred)
# y_pred = np.argmax(y_pred_probs, axis=1)

# y_true = val_gen_pred.classes
# class_labels = list(val_gen_pred.class_indices.keys())


# #basic plots    
model_acc_plt(history_dict_json)
model_loss_plt(history_dict_json)
# conf_matrix(y_true, y_pred, class_labels)

#img_path = "D:\\BIAI\\Formula One Cars\\Renault F1 car\\00000098.jpg"
#show_prediction(img_path, trained_model, class_labels)

# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Rozpoznawanie obrazu F1 z wybranym modelem")
#     parser.add_argument("--model_path", type=str, required=True, help="Ścieżka do pliku .h5 z modelem")
#     parser.add_argument("--image_path", type=str, required=True, help="Ścieżka do obrazu do predykcji")

#     args = parser.parse_args()

#     trained_model = load_model(args.model_path)
#     print("Model załadowany:", args.model_path)
#     trained_model.summary()

#     dummy_gen = datagen.flow_from_directory(
#         DATA_DIR,
#         target_size=IMAGE_SIZE,
#         batch_size=BATCH_SIZE,
#         class_mode='categorical',
#         subset='validation'
#     )
#     class_labels = list(dummy_gen.class_indices.keys())


#     show_prediction(args.image_path, trained_model, class_labels)