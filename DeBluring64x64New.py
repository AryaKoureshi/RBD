# Imports
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers, losses
import numpy as np
import matplotlib.pyplot as plt
import cv2
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
height, width = 64, 64
print("Loading Model...")
MyModel = load_model('Enter Location!/modelRBDNew64x64.h5')
MyModel.compile(optimizer=optimizers.Adam(), loss=losses.binary_crossentropy)
print("Model Loaded.")
print("loading image...")
image = cv2.imread('Enter Location! with File name', cv2.IMREAD_GRAYSCALE)
print("Image Loaded.")
heightimg = int(np.shape(image)[0])
widthimg = int(np.shape(image)[1])
height64 = int(heightimg/64)
width64 = int(widthimg/64)
image = image.astype(np.float32) / 255
print("Height : ", heightimg, "\n", "Width : ", widthimg)
print("Image after processing has {}x{} size!".format(width64 * 64, height64 * 64))
MyImage = np.zeros((height64 * 64, width64 * 64))

print("Start-Process...")
a = 0
for k in range(width64):
    for i in range(height64):
        px = image[64*i:64*(i+1),64*k:64*(k+1)]
        px = np.reshape(px, (1, height, width, 1))
        PredictedImage = MyModel.predict(px)
        PredictedImage = np.reshape(PredictedImage, (64, 64))
        MyImage[64*i:64*(i+1), 64*k:64*(k+1)] = PredictedImage
        a += 1
        print("{}/{}".format(a, width64 * height64))
print("Process completed.")

print("Saving image...")
MyImage = MyImage.astype(np.float32) * 255
cv2.imwrite('Enter Location!/predicted.jpg', MyImage)
print("Image saved.")

plt.figure(figsize=(2, 2))
# display original
ax = plt.subplot(2, 2, 1)
plt.imshow(image.reshape(heightimg, widthimg))
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
# display predicted by model
ax = plt.subplot(2, 2, 2)
plt.imshow(MyImage.reshape(height64 * 64, width64 * 64))
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Powered by Arya Koureshi")
###Powered by Arya Koureshi
