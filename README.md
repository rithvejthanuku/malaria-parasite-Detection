# malaria-parasite-Detection
the project shows how the malaria parasite can be detected using deep learning models , the models used were VGG16, Resnet50, CNN


from google.colab import drive
drive.mount('/content/drive')



import zipfile
with zipfile.ZipFile("/content/drive/MyDrive/Malaria-Detection-master.zip","r") as files:
  files.extractall("malariadataset")


import zipfile
with zipfile.ZipFile("/content/malariadataset/Malaria-Detection-master/Dataset.zip","r") as files:
  files.extractall("dataset")

import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt

# Generate random synthetic image data (you would replace this with loading images from your dataset)
# This code generates 10 random 64x64 grayscale images.
num_images = 10
image_size = (224, 224)

# Create an empty list to store the images
images = []

for _ in range(num_images):
    # Generate random pixel values for the image
    random_image = np.random.rand(*image_size)

    # Append the random image to the list
    images.append(random_image)

# Display random images
plt.figure(figsize=(12, 6))
for i in range(num_images):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images[i], cmap='Oranges')
    plt.title(f"Image {i + 1}")
    plt.axis('off')

plt.show()


from PIL import Image
import os

# Directory containing your dataset images
input_dir = "/content/dataset/Dataset/Train"
output_dir = "/content/dataset/Dataset/Train"  # The resized images will be saved here

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the new dimensions (width, height)
new_width = 224
new_height = 224

# Loop through the images in the input directory and resize each one
for filename in os.listdir(input_dir):
    if filename.endswith(('.jpg', '.png', '.jpeg')):  # Consider only image files
        # Open the image
        image_path = os.path.join(input_dir, filename)
        image = Image.open(image_path)

        # Resize the image
        resized_image = image.resize((new_width, new_height))

        # Save the resized image to the output directory
        output_path = os.path.join(output_dir, filename)
        resized_image.save(output_path)

print("Resizing complete.")


from PIL import Image
import os

# Directory containing your dataset images
input_dir = "/content/dataset/Dataset/Test"
output_dir = "/content/dataset/Dataset/Test"  # The resized images will be saved here

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the new dimensions (width, height)
new_width = 224
new_height = 224

# Loop through the images in the input directory and resize each one
for filename in os.listdir(input_dir):
    if filename.endswith(('.jpg', '.png', '.jpeg')):  # Consider only image files
        # Open the image
        image_path = os.path.join(input_dir, filename)
        image = Image.open(image_path)

        # Resize the image
        resized_image = image.resize((new_width, new_height))

        # Save the resized image to the output directory
        output_path = os.path.join(output_dir, filename)
        resized_image.save(output_path)

print("Resizing complete.")

import cv2
import os

# Directory containing your dataset images
input_dir = "/content/dataset/Dataset/Train"
output_dir = "/content/dataset/Dataset/Train"  # The normalized images will be saved here

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through the images in the input directory and normalize each one
for filename in os.listdir(input_dir):
    if filename.endswith(('.jpg', '.png', '.jpeg')):  # Consider only image files
        # Open the image
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale

        # Normalize the image (min-max scaling)
        min_val = image.min()
        max_val = image.max()
        normalized_image = (image - min_val) / (max_val - min_val)

        # Optionally, you can scale to a different range, e.g., [-1, 1]
        normalized_image = 2 * normalized_image - 1

        # Save the normalized image to the output directory
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, (normalized_image * 255).astype(int))  # Scale back to 0-255 range for saving

print("Normalization complete.")


import cv2
import os

# Directory containing your dataset images
input_dir = "/content/dataset/Dataset/Test"
output_dir = "/content/dataset/Dataset/Test"  # The normalized images will be saved here

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through the images in the input directory and normalize each one
for filename in os.listdir(input_dir):
    if filename.endswith(('.jpg', '.png', '.jpeg')):  # Consider only image files
        # Open the image
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale

        # Normalize the image (min-max scaling)
        min_val = image.min()
        max_val = image.max()
        normalized_image = (image - min_val) / (max_val - min_val)

        # Optionally, you can scale to a different range, e.g., [-1, 1]
        normalized_image = 2 * normalized_image - 1

        # Save the normalized image to the output directory
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, (normalized_image * 255).astype(int))  # Scale back to 0-255 range for saving

print("Normalization complete.")





import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Generate or load your dataset as a NumPy array
# For demonstration, we'll create a random dataset
data = np.random.rand(100, 3)  # 100 samples with 3 features

# Create a Min-Max scaler
scaler = MinMaxScaler()

# Fit the scaler to your data and transform it
normalized_data = scaler.fit_transform(data)

# Iterate through and print the normalized data
for sample in normalized_data:
    print(sample)


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import os

# Directory containing your original dataset
input_dir = "/content/dataset/Dataset/Train"
output_dir = "/content/dataset/Dataset/Train"  # Augmented images will be saved here

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create an ImageDataGenerator with augmentation options
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Iterate through the images in the input directory, augment them, and save to the output directory
for filename in os.listdir(input_dir):
    if filename.endswith(('.jpg', '.png', '.jpeg')):  # Consider only image files
        # Load the image
        image_path = os.path.join(input_dir, filename)
        img = image.load_img(image_path)
        x = image.img_to_array(img)
        x = x.reshape((1,) + x.shape)

        # Generate and save augmented images
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=output_dir, save_prefix='aug', save_format='jpeg'):
            i += 1
            if i >= 3:  # Adjust the number of augmented images generated per original image
                break

print("Data augmentation complete.")




IMAGE_SIZE = [224, 224]

train_path = '/content/dataset/Dataset/Train'
valid_path = '/content/dataset/Dataset/Test'

train_path

train_datagen = ImageDataGenerator(
                 rescale=1./255,
                 shear_range=0.2,
                 zoom_range=0.2,
                 horizontal_flip=True)
train="/content/dataset/Dataset/Train"
training_set=train_datagen.flow_from_directory(train,target_size=(224,224),
                                               batch_size=32,shuffle=True,class_mode="categorical")

test="/content/dataset/Dataset/Test"
test_datagen = ImageDataGenerator(
                 rescale=1./255)
test_set=train_datagen.flow_from_directory(test,target_size=(224,224),
                                               batch_size=16,shuffle=False,class_mode="categorical")

import tensorflow as tf
from tensorflow.keras.models import Sequential


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model=tf.keras.Sequential()
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation ="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(64,activation="relu"))
model.add(Dense(2,activation="softmax"))
model.summary()

model.summary()

history=model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])



history=model.fit(x=training_set,validation_data=test_set,epochs=35)



history=model.fit(x=test_set,validation_data=training_set,epochs=35)

import matplotlib.pyplot as plt
import tensorflow as tf

# Assuming 'history' contains the history object from model.fit()

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

training_set.class_indices

from tensorflow.keras.preprocessing import image
import numpy as np

test_image = image.load_img('/content/dataset/Dataset/Test/Parasite/C39P4thinF_original_IMG_20150622_105554_cell_14.png',
                            target_size=(224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result=model.predict(test_image)
training_set.class_indices
print(result)

if result[0][0] == 1:
  print('Infected Cell')
else:
  print('Uninfected cell')

train


import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the image using Keras preprocessing
test_image = image.load_img('/content/dataset/Dataset/Test/Parasite/C39P4thinF_original_IMG_20150622_105554_cell_14.png',
                            target_size=(224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# Get the prediction result from the model
result = model.predict(test_image)

# Define the class indices from the training set
class_indices = {'Parasite': 0, 'Uninfected': 1 } # Update with your class indices

# Display the image
plt.imshow(test_image[0].astype(np.uint8))  # Show the image
plt.axis('off')  # Hide axis labels
plt.show()

# Display the prediction result
predicted_class = np.argmax(result)
for class_name, index in class_indices.items():
    if index == predicted_class:
        print(f"Predicted class: {class_name}, Probability: {result[0][predicted_class]}")


from sklearn.metrics import classification_report

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

test_ds = tf.keras.utils.image_dataset_from_directory('/content/dataset/Dataset/Test',
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(224, 224),
  batch_size=32)


test_ds = tf.keras.utils.image_dataset_from_directory('/content/dataset/Dataset/Train',
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(224, 224),
  batch_size=32)

true_labels = []
predicted_labels = []

for images, labels in test_ds:
    predicted_batch = model.predict(images)
    predicted_labels.extend(np.argmax(predicted_batch, axis=1))
    true_labels.extend(labels.numpy())

from sklearn import metrics


# Calculate accuracy
accuracy = metrics.accuracy_score(true_labels, predicted_labels)
print("Accuracy:", accuracy)

# Generate a classification report
# Define your class names
class_names = ['Parasite', 'Uninfected']
class_report = classification_report(true_labels, predicted_labels, target_names=class_names)
print("Classification Report:\n", class_report)

# Assuming true_labels and predicted_labels are defined
print("True Labels:", true_labels)
print("Predicted Labels:", predicted_labels)


from sklearn.metrics import classification_report

# Define your class names
class_names = ['Parasite', 'Uninfected']



# Generate a classification report
class_report = classification_report(true_labels, predicted_labels, target_names=class_names)
print("Classification Report:\n", class_report)


# Compute and display the confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:\n", conf_matrix)



import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import os

# Directory containing your original dataset
input_dir = "/content/dataset/Dataset/Train"
output_dir = "/content/dataset/Dataset/Train"  # Augmented images will be saved here

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create an ImageDataGenerator with augmentation options
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Iterate through the images in the input directory, augment them, and save to the output directory
for filename in os.listdir(input_dir):
    if filename.endswith(('.jpg', '.png', '.jpeg')):  # Consider only image files
        # Load the image
        image_path = os.path.join(input_dir, filename)
        img = image.load_img(image_path)
        x = image.img_to_array(img)
        x = x.reshape((1,) + x.shape)

        # Generate and save augmented images
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=output_dir, save_prefix='aug', save_format='jpeg'):
            i += 1
            if i >= 3:  # Adjust the number of augmented images generated per original image
                break

print("Data augmentation complete.")



import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications import ResNet101

num_classes = 2
resnet_model = Sequential()

pretrained_model= ResNet101(include_top=False,
                            input_shape=(224,224,3),
                            pooling='avg',
                            weights='imagenet')  # You can choose to load imagenet weights or not, depending on your requirements

for layer in pretrained_model.layers:
    layer.trainable = False

resnet_model.add(pretrained_model)
resnet_model.add(Flatten())
resnet_model.add(Dense(128, activation='relu'))
resnet_model.add(Dense(num_classes))




# Print model summary
resnet_model.summary()

resnet_model.compile(optimizer=Adam(learning_rate=0.001), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

train_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)


training_set = train_datagen.flow_from_directory(
    '/content/dataset/Dataset/Train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'  # or 'binary' depending on your problem
)

test_set = test_datagen.flow_from_directory(
    '/content/dataset/Dataset/Test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'  # or 'binary' depending on your problem
)

base_model = ResNet50(weights='imagenet', include_top=False)

NUM_CLASSES = 2  # Replace 10 with the actual number of classes in your dataset


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

resnet_model = Model(inputs=base_model.input, outputs=predictions)

resnet_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 35
history = resnet_model.fit(
    training_set,
    epochs=epochs,
    validation_data=test_set
)

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()



import matplotlib.pyplot as plt
import tensorflow as tf

# Assuming 'history' contains the history object from model.fit()

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


from tensorflow.keras.preprocessing import image

img1 = image.load_img('/content/dataset/Dataset/Test/Parasite/C39P4thinF_original_IMG_20150622_105554_cell_14.png',target_size=(224,224))
img1

test_image=image.img_to_array(img1)
test_image=np.expand_dims(test_image, axis = 0)
result =np.argmax(resnet_model.predict(test_image), axis=1)
result

predictions1 = class_names[result[0]]
predictions1



test_ds = tf.keras.utils.image_dataset_from_directory('/content/dataset/Dataset/Test',
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(224, 224),
  batch_size=32)


true_labels = []
predicted_labels = []

for images, labels in test_ds:
    predicted_batch = resnet_model.predict(images)
    predicted_labels.extend(np.argmax(predicted_batch, axis=1))
    true_labels.extend(labels.numpy())

# Calculate accuracy
accuracy = metrics.accuracy_score(true_labels, predicted_labels)
print("Accuracy:", accuracy)

# Generate a classification report
class_report = classification_report(true_labels, predicted_labels, target_names=class_names)
print("Classification Report:\n", class_report)

print("True Labels:", true_labels)
print("Predicted Labels:", predicted_labels)


from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input

# List of paths to your test images
image_paths = ['/content/dataset/Dataset/Test/Parasite/C39P4thinF_original_IMG_20150622_105554_cell_14.png']

# List to store preprocessed images
X_test = []

for img_path in image_paths:
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)  # Preprocess the image
    X_test.append(img_array)

# Convert list to numpy array
X_test = np.array(X_test)

# Now you can predict labels using your model
y_pred = resnet_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)


# Generate a classification report
class_report = classification_report(true_labels, predicted_labels, target_names=class_names)
print("Classification Report:\n", class_report)

model.save('path/to/your/model')ackages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/usr/local/lib/python3.10/dist-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))

[ ]
model.save('path/to/your/model')
