import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image

def plot_training(history, save_path=None):
    """Plot training/validation accuracy and loss."""
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, acc, label='Train Acc')
    plt.plot(epochs, val_acc, label='Val Acc')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.title('Loss')
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def show_augmentations(datagen, sample_image_path, save_path=None, n=6, target_size=(224,224)):
    """Show augmented versions of one sample image."""
    img = image.load_img(sample_image_path, target_size=target_size)
    x = image.img_to_array(img)
    x = x.reshape((1,) + x.shape)

    i = 0
    plt.figure(figsize=(12,6))
    for batch in datagen.flow(x, batch_size=1):
        plt.subplot(2, n//2, i+1)
        plt.imshow(image.array_to_img(batch[0].astype('uint8')))
        plt.axis('off')
        i += 1
        if i >= n:
            break
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def save_labels(generator, out_path='class_indices.npy'):
    """Save class indices mapping to disk for later use."""
    np.save(out_path, generator.class_indices)

def load_image_for_prediction(img_path, target_size=(224,224)):
    img = image.load_img(img_path, target_size=target_size)
    arr = image.img_to_array(img)
    arr = arr/255.0
    return np.expand_dims(arr, axis=0)

def grad_cam(model, img_path, last_conv_layer_name=None, preprocess_size=(224,224), upsample_size=None, show=True, save_path=None):
    """Generate Grad-CAM heatmap for a binary classification model.

    - model: keras Model
    - img_path: path to image
    - last_conv_layer_name: name of last conv layer in model. If None, auto-search.
    """
    # Load and preprocess image
    img = image.load_img(img_path, target_size=preprocess_size)
    img_arr = image.img_to_array(img) / 255.0
    input_arr = np.expand_dims(img_arr, axis=0)

    if last_conv_layer_name is None:
        # find last conv layer
        for layer in reversed(model.layers):
            if len(layer.output_shape) == 4:
                last_conv_layer_name = layer.name
                break
    if last_conv_layer_name is None:
        raise ValueError("Couldn't find a conv layer in the model. Provide last_conv_layer_name.")

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_arr)
        # for binary sigmoid output predictions[:,0]; for softmax use argmax
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    guided_grads = grads[0]

    conv_outputs = conv_outputs[0].numpy()
    weights = np.mean(guided_grads.numpy(), axis=(0, 1))

    cam = np.zeros(conv_outputs.shape[0:2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * conv_outputs[:, :, i]

    cam = np.maximum(cam, 0)
    if cam.max() != 0:
        cam = cam / cam.max()

    if upsample_size is None:
        upsample_size = preprocess_size

    cam = cv2.resize(cam, upsample_size)
    heatmap = (cam * 255).astype("uint8")
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    orig = cv2.cvtColor(np.uint8(img_arr*255), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

    if show:
        plt.figure(figsize=(8,4))
        plt.subplot(1,3,1)
        plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
        plt.title("Original")
        plt.axis('off')

        plt.subplot(1,3,2)
        plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
        plt.title("Heatmap")
        plt.axis('off')

        plt.subplot(1,3,3)
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title("Overlay")
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    return overlay
