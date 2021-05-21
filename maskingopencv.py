import cv2
import numpy as np


### Boundaries :

bounds  = [
    ([0,0,60],[20,20,150]), # red
    ([15,70,105],[22,130,180]), # yellow
    ([10,27,89],[18,48,143]), #orange
    ([2,12,38],[28,35,50]), #brown
    ([16,55,20],[50,105,60]), #green
    ([30,20,0],[80,50,20])  #blue
]

#################,

'''
mm = cv2.imread("KELTOS.png", cv2.WINDOW_AUTOSIZE)

r = 700.0 / mm.shape[1]
dim = (700, int(mm.shape[0] * r))

resized = cv2.resize(mm, dim, interpolation= cv2.INTER_AREA)

# loop over the boundaries
for (lower, upper) in bounds:
	# create NumPy arrays from the boundaries*
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")
	# find the colors within the specified boundaries and apply
	# the mask
	mask = cv2.inRange(resized, lower, upper)
	output = cv2.bitwise_and(resized, resized, mask = mask)
	# show the images
	cv2.imshow("images", np.hstack([resized, output]))
	cv2.waitKey(0)
'''

wc = cv2.VideoCapture('pivideo_2.mp4')

while(1):

    _, imgFrame =  wc.read()

    hsv = cv2.cvtColor(imgFrame, cv2.COLOR_BGR2HSV)

    # Set range for blue color and
    # define mask
    
    #bgr

    #blue:
    blue_lower = np.array([94, 80, 200], np.uint8)
    blue_upper = np.array([120, 255, 255], np.uint8)
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    #red:
    red_lower = np.array([70,80,200], np.uint8)
    #[10,120,213]
    red_upper = np.array([140,170,255], np.uint8)
    #[40,140,235]
    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    #green:
    green_lower = np.array([35,60,35], np.uint8)
    green_upper = np.array([80,105,80], np.uint8)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    #brown:
    brown_lower = np.array([2,12,38], np.uint8)
    brown_upper = np.array([28,35,50], np.uint8)
    brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)
    #yellow:
    yellow_lower = np.array([0,60,180], np.uint8)
    yellow_upper = np.array([40,150,215], np.uint8)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    #orange:    
    orange_lower = np.array([2,12,38], np.uint8)
    orange_upper = np.array([28,35,50], np.uint8)
    orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)



    kernal = np.ones((5, 5), "uint8")

    # For blue color
    blue_mask = cv2.dilate(blue_mask, kernal)
    res_blue = cv2.bitwise_and(imgFrame, imgFrame, mask = blue_mask)
    # For red color
    red_mask = cv2.dilate(red_mask, kernal)
    res_red = cv2.bitwise_and(imgFrame, imgFrame, mask = red_mask)
    # For green color
    green_mask = cv2.dilate(green_mask, kernal)
    res_green = cv2.bitwise_and(imgFrame, imgFrame, mask = green_mask)
    # For brown color
    brown_mask = cv2.dilate(brown_mask, kernal)
    res_brown = cv2.bitwise_and(imgFrame, imgFrame, mask = brown_mask)
    # For yellow color
    yellow_mask = cv2.dilate(yellow_mask, kernal)
    res_yellow = cv2.bitwise_and(imgFrame, imgFrame, mask = yellow_mask)
    # For orange color
    orange_mask = cv2.dilate(orange_mask, kernal)
    res_orange = cv2.bitwise_and(imgFrame, imgFrame, mask = orange_mask)

    
    
    # Creating contour to track blue color
    contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)    
            imgFrame = cv2.rectangle(imgFrame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(imgFrame, "Blue Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0))

    contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)    
            imgFrame = cv2.rectangle(imgFrame, (x, y), (x + w, y + h), (0,0,255), 2)
            cv2.putText(imgFrame, "Red Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255))

    contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)    
            imgFrame = cv2.rectangle(imgFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(imgFrame, "Green Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
    
    contours, hierarchy = cv2.findContours(brown_mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)    
            imgFrame = cv2.rectangle(imgFrame, (x, y), (x + w, y + h), (42,42,165), 2)
            cv2.putText(imgFrame, "Brown Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (42,42,165))

    contours, hierarchy = cv2.findContours(yellow_mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)    
            imgFrame = cv2.rectangle(imgFrame, (x, y), (x + w, y + h), (0,255,255), 2)
            cv2.putText(imgFrame, "Yellow Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255))

    contours, hierarchy = cv2.findContours(orange_mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)    
            imgFrame = cv2.rectangle(imgFrame, (x, y), (x + w, y + h), (0,165,255), 2)
            cv2.putText(imgFrame, "Orange Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,165,255))


    # Program Termination
    cv2.imshow("Multiple Color Detection in Real-TIme", imgFrame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        wc.release()
        break
    