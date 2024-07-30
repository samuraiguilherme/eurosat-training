import os
from datasets import load_dataset, ClassLabel, Sequence
from transformers import (
    DefaultDataCollator,
    TFAutoModelForImageClassification,

    AutoImageProcessor,
    create_optimizer
)
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers
from transformers.keras_callbacks import KerasMetricCallback
import evaluate

model_id = "google/vit-base-patch16-224-in21k"
output_dir = './img-eurosat-model-trained'

simp_ds = load_dataset('yaguilherme/eurosat-ds-test2')

print('raw simp_ds: ',simp_ds)

label_names = os.listdir('eurosat-simp')

# Cast to ClassLabel
simp_ds = simp_ds.cast_column("label", ClassLabel(names=label_names))

train_test_ds = simp_ds['train'].train_test_split(test_size=0.2)
print('train_test_ds: ',train_test_ds)

labels = train_test_ds['train'].features['label'].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

image_processor = AutoImageProcessor.from_pretrained(model_id)

data_augmentation = tf.keras.Sequential(
    [
        layers.Resizing(image_processor.size['height'], image_processor.size['width']),
        layers.Rescaling(1./255, offset=-1),
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2)
    ],
    name="data_augmentation"
)
val_data_augmentation = tf.keras.Sequential(
    [
        layers.CenterCrop(image_processor.size['height'], image_processor.size['width']),
        layers.Rescaling(scale=1.0 /127.5, offset=-1)
    ],
    name='val_data_augmentation'
)

def convert_to_tf_tensor(image: Image):
    np_image = np.array(image)
    tf_image = tf.convert_to_tensor(np_image)
    return tf.expand_dims(tf_image, 0)

def preprocess_train(example_batch):
    images = [
        data_augmentation(convert_to_tf_tensor(image.convert('RGB'))) for image in example_batch['image']
    ]
    example_batch['pixel_values'] = [tf.transpose(tf.squeeze(image)) for image in images]
    return example_batch

def preprocess_val(example_batch):
    images = [
        val_data_augmentation(convert_to_tf_tensor(image.convert('RGB'))) for image in example_batch['image']
    ]
    example_batch['pixel_values'] = [tf.transpose(tf.squeeze(image)) for image in images]
    return example_batch

train_test_ds['train'].set_transform(preprocess_train)
simp_ds['validation'].set_transform(preprocess_val)

data_collator = DefaultDataCollator(return_tensors='tf')

accuracy = evaluate.load('accuracy')

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

batch_size=16
num_epochs=5
num_train_steps=len(train_test_ds['train']) * num_epochs
learning_rate=3e-5
weight_decay_rate=0.01

optimizer, lr_schedule = create_optimizer(
    init_lr=learning_rate,
    num_train_steps=num_train_steps,
    weight_decay_rate=weight_decay_rate,
    num_warmup_steps=0
)

model = TFAutoModelForImageClassification.from_pretrained(
    model_id,
    id2label=id2label,
    label2id=label2id
)

tf_train_ds = train_test_ds['train'].to_tf_dataset(
    columns='pixel_values',
    label_cols='label',
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator
)
tf_val_ds = simp_ds['validation'].to_tf_dataset(
    columns='pixel_values',
    label_cols='label',
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator
)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = [
    tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
    tf.keras.metrics.SparseTopKCategoricalAccuracy(3, name='top-3-accuracy')
]
model.compile(
    optimizer=optimizer,
    loss=loss,
)
callbacks = [
    KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_val_ds)
]
model.fit(
    tf_train_ds,
    validation_data=tf_val_ds,
    callbacks=callbacks,
    epochs=num_epochs
)

curpath = os.path.abspath(os.curdir)
fulldirPath = os.path.join(curpath, 'eurosat-model-trained')

model.save_pretrained(fulldirPath)
image_processor.save_pretrained(fulldirPath)