import os

os.environ['KERAS_BACKEND'] = 'jax'

import keras
from keras import layers
from keras import ops

import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset

eurosat_ds = load_dataset('yaguilherme/eurosat-full')
eurosat_ds.set_format(type='tensorflow', columns=['image', 'label'])

num_classes = len(eurosat_ds['train'].features['label'].names)
input_shape = (64, 64, 3)

x_train = eurosat_ds['train']['image']
y_train = eurosat_ds['train']['label']

x_test = eurosat_ds['validation']['image']
y_test = eurosat_ds['validation']['label']

print(f'x_train shape: {x_train.shape} - y_train shape: {y_train.shape}')
print(f'x_test shape: {x_test.shape} - y_test shape: {y_test.shape}')

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 10 # for real use increase to at least 100
image_size = 64 # square 64 x 64
patch_size = 8 # patch size to extract from images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim
] # size of the transformer layers
transformer_layers = 8
mlp_head_units = [
    2048,
    1024,
] # size of the dense layers of the final classifier

data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name='data_augmentation'
)

data_augmentation.layers[0].adapt(np.array(x_train))

def mlp(x, hidden_units, dropout_rate):
  for units in hidden_units:
    x = layers.Dense(units, activation=keras.activations.gelu)(x)
    x = layers.Dropout(dropout_rate)(x)
  return x

class Patches(layers.Layer):
  def __init__(self, patch_size):
    super().__init__()
    self.patch_size = patch_size

  def call(self, images):
    input_shape = ops.shape(images)
    batch_size = input_shape[0]
    height = input_shape[1]
    width = input_shape[2]
    channels = input_shape[3]
    num_patches_h = height // self.patch_size
    num_patches_w = width // self.patch_size
    patches = keras.ops.image.extract_patches(images, self.patch_size)
    patches = ops.reshape(
        patches,
        (
            batch_size, 
            num_patches_h * num_patches_w,
            self.patch_size * self.patch_size * channels
        ),
    )
    return patches

  def get_config(self):
    config = super().get_config()
    config.update({"patch_size": self.patch_size})
    return config
  
class PatchEncoder(layers.Layer):
  def __init__(self, num_patches, projection_dim):
    super().__init__()
    self.num_patches = num_patches
    self.projection = layers.Dense(units=projection_dim)
    self.position_embedding = layers.Embedding(
        input_dim=num_patches,
        output_dim=projection_dim
    )

  def call(self, patch):
    positions = ops.expand_dims(
        ops.arange(start=0, stop=self.num_patches, step=1),
        axis=0
    )
    projected_patches = self.projection(patch)
    encoded = projected_patches + self.position_embedding(positions)
    return encoded

  def get_config(self):
    config = super().get_config()
    config.update({"num_patches": self.num_patches})
    return config
  
def create_vit_classifier():
  inputs = keras.Input(shape=input_shape)
  augmented = data_augmentation(inputs)
  patches = Patches(patch_size)(augmented)
  encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

  for _ in range(transformer_layers):
    # Layer normalization 1
    x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

    # Create a multi-head attention layer
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=projection_dim, dropout=0.1
    )(x1, x1)

    # Skip connection 1
    x2 = layers.Add()([attention_output, encoded_patches])
    
    # Layer normalization 2
    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

    # MLP
    x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

    # Skip connection 2
    x3 = layers.Add()([x3, x2])

  # Create a [batch_size, projection_dim] tensor
  representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
  representation = layers.Flatten()(representation)
  representation = layers.Dropout(0.5)(representation)

  # Add MLP
  features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)

  # Classify output
  logits = layers.Dense(num_classes)(features)

  # Create keras model
  model = keras.Model(inputs=inputs, outputs=logits)
  keras.utils.plot_model(model, show_shapes=True)
  return model

def run_experiment(model):
  optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)

  model.compile(
      optimizer=optimizer,
      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[
          keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
          keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
      ]
  )

  checkpoint_filepath = 'tmp/checkpoint.weights.h5'
  checkpoint_callback = keras.callbacks.ModelCheckpoint(
      checkpoint_filepath,
      monitor='val_accuracy',
      save_best_only=True,
      save_weights_only=True
  )
  history = model.fit(
      x=x_train,
      y=y_train,
      batch_size=batch_size,
      epochs=num_epochs,
      validation_split=0.1,
      callbacks=[checkpoint_callback]
  )

  model.load_weights(checkpoint_filepath)
  _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
  print(f"Test accuracy: {round(accuracy * 100, 2)}%")
  print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

  return history

vit_classifier = create_vit_classifier()
history = run_experiment(vit_classifier)

def plot_history(item):
  plt.plot(history.history[item], label=item)
  plt.plot(history.history['val_' + item], label="val_"+item)
  plt.xlabel('epochs')
  plt.ylabel(item)
  plt.title('Train and Validation {} over epochs'.format(item), fontsize=14)
  plt.legend()
  plt.grid()
  plt.show()

plot_history('loss')
plot_history('top-5-accuracy')