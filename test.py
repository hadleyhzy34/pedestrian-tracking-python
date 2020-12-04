import cv2 
import imutils 
import numpy as np
from model import pedestrian_tracking

font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

#initialize matches that contain pedestrian id, position based on the first frame
# detector 
hog = cv2.HOGDescriptor() 
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) 
   
# Reading the Image 
image = cv2.imread('pedestrian1.jpg') 
# print(image.shape)
   
# Resizing the Image 
image = imutils.resize(image, 
                       width=min(400, image.shape[1])) 
   
# Detecting all the regions in the  
# Image that has a pedestrians inside it 
(regions, _) = hog.detectMultiScale(image,  
                                    winStride=(4, 4), 
                                    padding=(4, 4), 
                                    scale=1.05) 

# center points of 2d image
rectangles = []
   
# Drawing the regions in the Image 
for (x, y, w, h) in regions: 
    # print(x,y,w,h)
    rectangles.append([x,y,w,h])
    cv2.rectangle(image, (x, y),  
                  (x + w, y + h),  
                  (0, 0, 255), 2)

rectangles = np.array(rectangles)

center_2d = np.zeros((rectangles.shape[0],2))
center_2d[:,0] = rectangles[:,0] + rectangles[:,2]//2
center_2d[:,1] = rectangles[:,1] + rectangles[:,3]
center_2d = center_2d.astype(int)
# print(center_2d,center_2d.shape)

# initialize matches
matches = np.zeros((rectangles.shape[0],3),dtype=int)
for i,point in enumerate(center_2d):
    matches[i][0]=i
    matches[i][1]=point[0]
    matches[i][2]=point[1]

# print(matches)

#show pedestrian id 
for match in matches:
    cv2.putText(image,str(match[0]),(match[1],match[2]),font,fontScale,fontColor,lineType)

# # Showing the output Image 
cv2.imshow("Image1", image) 
cv2.waitKey(0) 
cv2.destroyAllWindows() 


#pedestrian tracking for the second image based on hungarian algorithm and previous frame 
# Reading the Image 
image = cv2.imread('pedestrian3.jpg') 
image = imutils.resize(image, 
                       width=min(400, image.shape[1])) 
(regions, _) = hog.detectMultiScale(image,  
                                    winStride=(4, 4), 
                                    padding=(4, 4), 
                                    scale=1.05) 
rectangles = []
   
# Drawing the regions in the Image 
for (x, y, w, h) in regions: 
    rectangles.append([x,y,w,h])
    cv2.rectangle(image, (x, y),  
                  (x + w, y + h),  
                  (0, 0, 255), 2)

rectangles = np.array(rectangles)

#tracking
projection_matrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1]])
matches = pedestrian_tracking(rectangles,projection_matrix,matches)

for match in matches:
    cv2.putText(image,str(match[0]),(match[1],match[2]),font,fontScale,fontColor,lineType)

# # Showing the output Image 
cv2.imshow("Image2", image) 
cv2.waitKey(0) 
cv2.destroyAllWindows() 

