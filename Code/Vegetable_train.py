import seaborn as sns
import matplotlib.pyplot as plt
from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import numpy as np
import tensorflow as tf

import pandas as pd

# Define data generators for training and testing
train_data_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_data_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_data_gen.flow_from_directory(
    r"D:\Vegnet\Dataset\New_Vegnet_Vege\Bell_Pepper",
    target_size=(299, 299),
    batch_size=10,
    class_mode='categorical', 
    shuffle=True
) 

test_generator = test_data_gen.flow_from_directory(
    r"D:\Vegnet\Dataset\Train\Bell_Pepper",
    target_size=(299, 299),
    batch_size=10,
    class_mode='categorical',
    shuffle=True
)

# Define the number of classes
num_classes = train_generator.num_classes
num_classes = test_generator.num_classes
class_names = list(test_generator.class_indices.keys())


# Load the Xception model
base_model = Xception(weights='imagenet', include_top=False)

# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a fully connected layer
x = Dense(1024, activation='relu')(x)

# Add a logistic layer
predictions = Dense(num_classes, activation='softmax')(x)

# Define the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Print the model summary
model.summary()

# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=5, validation_data=test_generator)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator)
print('Test loss:', test_loss)

print('Test accuracy:', test_acc)

# Make predictions on the test set
y_true = test_generator.classes
y_pred = np.argmax(model.predict(test_generator), axis=-1)

print('Confusion Matrix:')
cm = confusion_matrix(y_true, y_pred)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
print(cm_df)
# Save the trained model
model.save('Bell_Pepper_2.h5')
