import cv2
image = cv2.imread("Iron-Man.png")
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_img = cv2.medianBlur(gray_img, 1)
sketch = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
cv2.imshow("Image", image)
cv2.imshow("corners", sketch)
cv2.waitKey(0)