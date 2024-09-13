import cv2

path = "Grad_3_class/A4C/221.png"
img = cv2.imread(path)
rgb_img = cv2.resize(img, (224, 224))
cv2.imwrite(path, rgb_img)
