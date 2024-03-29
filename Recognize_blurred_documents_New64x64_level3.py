# Imports
from tensorflow.keras import optimizers, losses
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# # # Loading train data
width =  64
height = 64

print("Importing Data...")

image_xtrain_path = "X_Train data location (with '/')(64x64!)"
xtrain_images = glob.glob(image_xtrain_path + "*.jpg")
xtrain_images.sort()

x_train = []
for img in xtrain_images:
   image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
   image = image.astype(np.float32) / 255
   x_train.append(image)

image_ytrain_path = "Y_Train data location (with '/')(64x64!)"
ytrain_images = glob.glob(image_ytrain_path + "*.jpg")
ytrain_images.sort()

y_train = []
for img in ytrain_images:
   image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
   image = image.astype(np.float32) / 255
   y_train.append(image)

# # # Loading test data
image_xtest_path = "X_Test data location (with '/')(64x64!)"
xtest_images = glob.glob(image_xtest_path + "*.jpg")
xtest_images.sort()

x_test = []
for img in xtest_images:
   image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
   image = image.astype(np.float32) / 255
   x_test.append(image)
 
image_ytest_path = "Y_Test data location (with '/')(64x64!)"
ytest_images = glob.glob(image_ytest_path + "*.jpg")
ytest_images.sort()

y_test = []
for img in ytest_images:
   image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
   image = image.astype(np.float32) / 255
   y_test.append(image)   

print("Importing Data was Successful.")
del(image_xtrain_path, xtrain_images, img, image_ytrain_path, ytrain_images, image_xtest_path, xtest_images, image_ytest_path, ytest_images, i, ii, image)
print("Deleted extra data.")
#::::::::::::::::::::::::::::::::::::::::
# Prepare_data
print("Preparing Data...")
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
x_train = np.reshape(x_train, (len(x_train), height, width, 1))
y_train = np.reshape(y_train, (len(y_train), height, width, 1))
x_test = np.reshape(x_test, (len(x_test), height, width, 1))
print("Prepared Successful.")
# #::::::::::::::::::::::::::::::::::::::::
# # Create Layers and Model
print("Crating Model...")
autoencoder = load_model('Enter Location!/modelRBDNew64x64.h5')
autoencoder.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss=losses.binary_crossentropy, metrics=['acc'])
print("Model Created and Compiled.")
#:::::::::::::::::::::::::::::::::::::::::::
# Train the model
print("Start Training...")
autoencoder.fit(x_train, y_train,
                epochs=30,
		initial_epoch=20,
                batch_size=32,
                shuffle=True,
                validation_split=0.1)
#Save trained model
print("Saving Trained Model...")
autoencoder.save('Enter Location!/modelRBDNew64x64.h5')
print("Saved Successful.")
#::::::::::::::::::::::::::::::::::::::::::
# Predict and Visualization
print("Predicting Tests")
decoded_imgs = autoencoder.predict(x_test)
print("Predicted Successful.")

# Save predicted images
print("Saving Predicted Images...")
for i in range(len(decoded_imgs)):
    decoded_imgs[i] = decoded_imgs[i].astype(np.float32) * 255
    cv2.imwrite('Enter Location!/{}.jpg'.format(i), decoded_imgs[i])
print("Images Saved Successful.")
#::::::::::::::::::::::::::::::::::
n = 7
plt.figure(figsize=(21, 7))
for i in range(n):
    # display original
    ax = plt.subplot(3, n, i+1)
    plt.imshow(y_test[i].reshape(height, width))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # display construction
    ax = plt.subplot(3, n, i+n+1)
    plt.imshow(x_test[i].reshape(height, width))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # display reconstruction
    ax = plt.subplot(3, n, i+(2*n)+1)
    plt.imshow(decoded_imgs[i].reshape(height, width))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
cv2.waitKey(0) 
cv2.destroyAllWindows()
#::::::::::::::::::::::::::::::::::::::::::::
# Plot model
print("Ploting Model...")
plot_model(autoencoder, to_file='Enter Location!/modelRBDNew64x64.png', show_shapes=True, expand_nested=True, dpi=300)
print("Plotted Successful.")
print("All Process Passed Successful!")
    
