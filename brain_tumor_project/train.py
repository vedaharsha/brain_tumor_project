import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import plot_training, save_labels

# --------------------------
# UPDATED DATASET PATHS ✔
# --------------------------
train_dir = r"C:\DL_PROJECT\brain_tumor_project\dataset\train"
val_dir   = r"C:\DL_PROJECT\brain_tumor_project\dataset\val"

# --------------------------
# Model settings
# --------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15
LR = 1e-4
NUM_CLASSES = 4

# --------------------------
# Data generators
# --------------------------
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    shear_range=0.1,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# --------------------------
# CREATE MODELS FOLDER BEFORE SAVING LABELS ✔
# --------------------------
os.makedirs("models", exist_ok=True)

save_labels(train_data, "models/class_indices.npy")

# --------------------------
# Build MobileNetV2 Model
# --------------------------
base = MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights="imagenet"
)
base.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)

output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base.input, outputs=output)

model.compile(
    optimizer=Adam(LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --------------------------
# Callbacks
# --------------------------
os.makedirs("models", exist_ok=True)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    ModelCheckpoint("models/best_model.h5", save_best_only=True)
]

# --------------------------
# Train model
# --------------------------
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data,
    callbacks=callbacks
)

plot_training(history, "models/training.png")

# --------------------------
# Fine-tune last few layers
# --------------------------
base.trainable = True
for layer in base.layers[:-40]:  # unfreeze last 40 layers
    layer.trainable = False

model.compile(optimizer=Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])

history_ft = model.fit(
    train_data,
    epochs=5,
    validation_data=val_data,
    callbacks=callbacks
)

plot_training(history_ft, "models/finetune.png")

model.save("models/best_model.h5")
print("Model saved in models/best_model.h5")
plot_training(history_ft, "models/finetune.png")