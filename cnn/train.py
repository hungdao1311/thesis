import os
import tensorflow as tf

train_dir = './train'
val_dir = './test'

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop



train_data_gen = ImageDataGenerator()
val_data_gen = ImageDataGenerator()

train_generator = train_data_gen.flow_from_directory(train_dir, class_mode='categorical', batch_size=20,
                                                     target_size=(32, 32))
valid_generator = val_data_gen.flow_from_directory(val_dir, batch_size=20, class_mode='categorical',
                                                   target_size=(32, 32))

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(2, activation='sigmoid')
    ])
model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])
history = model.fit_generator(train_generator, validation_data=valid_generator, steps_per_epoch=391, epochs=20,
                              verbose=1)
# !mkdir -p saved_model
# model.save(os.path.join('saved_model', 'my_model'))
model.save('D:\Workspace\Thesis\graduate-thesis\cnn\saved_model\my_model')