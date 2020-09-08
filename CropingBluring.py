import cv2
import glob
import numpy as np
images_path = "D:/Files/ProgrammingLessons/python/RecognizingBlurredDocuments/FirstData/"
images = glob.glob(images_path + "*.jpg")
images.sort()
c = 1
for img in images:
   image = cv2.imread(img)
   height = int(np.shape(image)[0])
   width = int(np.shape(image)[1])
   height64 = int(height/64)
   width64 = int(width/64)
   print("shape {}: /n".format(img), np.shape(image))
   print("height : ", height, "\n", "width : ", width)
   blurImg = cv2.blur(image ,(25,25))
   for k in range(width64):
       for i in range(height64):
           px = image[64*i:64*(i+1),64*k:64*(k+1)]
           px_blur = blurImg[64*i:64*(i+1),64*k:64*(k+1)]
           cv2.imwrite('D:/Files/ProgrammingLessons/python/RecognizingBlurredDocuments/Ytrain/data{}shape{}x{}.jpg'.format(c,i,k),px)            
           cv2.imwrite('D:/Files/ProgrammingLessons/python/RecognizingBlurredDocuments/Xtrain/data{}shape{}x{}.jpg'.format(c,i,k),px_blur)            
   c += 1
cv2.waitKey(0)
cv2.destroyAllWindows()