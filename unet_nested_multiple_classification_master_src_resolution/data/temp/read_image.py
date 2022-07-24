from PIL import Image
import cv2

img_path = 'temp_bar_gray.png'
img = Image.open(img_path)
print(img)

img_cv = cv2.imread(img_path)
print(img_cv)