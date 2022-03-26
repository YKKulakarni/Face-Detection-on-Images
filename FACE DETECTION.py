import cv2

# load the line
cascade = cv2.CascadeClassifier("C:\\Users\\Home\\Desktop\\FD\\haarcascade_frontalface_default.xml")

# read image
img = cv2.imread("D:\\YK\\Dandeli Trip\\GOPR5106.JPG")

# resize image
img2 = cv2.resize(img,(500,360))

# convert to gray scale
gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

# Find faces in gray scale
Faces = cascade.detectMultiScale(gray,1.1,4)

for (x1,y1,w,h) in Faces:
    cv2.rectangle(img2,(x1,y1),(x1+w,y1+h),(0,255,0),1)

# show image
cv2.imshow("Face Detection",img2)
cv2.waitKey(0)
