import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import Xception, preprocess_input

model = Xception(weights='imagenet', include_top=False) 
img_path = r"C:\Users\dushy\Downloads\archive\Final Dataset\Fake\fake_95_aug_4.jpg" 
img = image.load_img(img_path, target_size=(299, 299))  
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)
 
layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name]
feature_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
feature_maps = feature_model.predict(img_array)

def plot_feature_maps(feature_maps, num_maps=6):
    fig, axes = plt.subplots(1, num_maps, figsize=(20, 10))
    for i in range(num_maps):
        axes[i].imshow(feature_maps[0, :, :, i], cmap='viridis')
        axes[i].axis('off')
    plt.show()

plot_feature_maps(feature_maps[0])
