# import libraries
from __future__ import division
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np
import math
from random import randint

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False


# define a function that takes input from the mouse to generate a masked area
def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False

        # draw a rectangle around the region of interest
        cv2.rectangle(ref, refPt[0], refPt[1], (0, 255, 0), 2)


# construct the argument parser and parse the argumentsMT_Vid.mp4
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to video", default="/home/karim/Desktop/BEE/FYP_Bee_Swarming/MT_Vid.mp4")
ap.add_argument("-a", "--min-area", type=int, default=125, help="minimum area size")
args = vars(ap.parse_args())

vs = cv2.VideoCapture(args["video"])

# initialize the reference frame in the video stream
global firstFrame
firstFrame = None

frame = vs.read()
frame = frame if args.get("video", None) is None else frame[1]


# resize the frame, convert it to grayscale, and blur it
# the size of the blurring kernel is a parameter that affects the threshold results
blur_kernel = 25  # in pixels
frame = imutils.resize(frame, width=500)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)


# if the first frame is None, initialize it
if firstFrame is None:
    firstFrame = gray
    #cv2.imshow("Reference", frame) #Uncomment if you need to redefine a reference image.
    #cv2.imwrite("Reference.jpg", frame)
    ref = cv2.imread("Reference.jpg")
    clone = ref.copy()
    cv2.namedWindow("Reference")
    #set a callback to actually input the mask dimensions
    cv2.setMouseCallback("Reference", click_and_crop)

    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow("Reference", ref)
        key = cv2.waitKey(1) & 0xFF

        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            ref = clone.copy()

        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            # print(refPt)
            global reference_points
            reference_points = refPt
            firstFrame = firstFrame[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
            break

    # if there are two reference points, then crop the region of interest
    # from the image and display it
    if len(refPt) == 2:
        roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        cv2.imshow("ROI", roi)
        cv2.waitKey(0)
    #print(reference_points)
    # close all open windows
    cv2.destroyAllWindows()

    # Display the reference image upon which the motion detection is applied
    cv2.imshow("Reference Image", firstFrame)
    adaptiveFrame = firstFrame
    originalFrame = firstFrame.copy()

#initialize global components
bee = {}
global_counter = 0
colors = {}
status = {}
flow = 0
frame_num = 0
counting = 0
to_pop = {}

z = 0
# loop over the frames saved in the test folder
for w in range(1, 80):

    #set the number of bee detected initially to zero
    number_in_image = 0

    #initialize a local dictionary that will contain the coordinates of the bees detected on the specific frame
    bee_coord = {}

    path = "/home/karim/Desktop/BEE/FYP_Bee_Swarming/Test_frames_5/test_frame_{0}.jpg".format(w)
    #print(path)
    frame = cv2.imread(path)

    if frame is None:
        break

    # resize the frame, convert it to grayscale, and blur it
    # the size of the blurring kernel is a parameter that affects the results
    blur_kernel = 25  # in pixels
    frame = imutils.resize(frame, width=500)
    frame = frame[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)



    # compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv2.absdiff(originalFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]  # initially 40 #then 25

    frameDelta2 = cv2.absdiff(adaptiveFrame, gray)
    thresh2 = cv2.threshold(frameDelta2, 25, 255, cv2.THRESH_BINARY)[1]

    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh_og = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh2, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    #update adaptiveFrame
    height = adaptiveFrame.shape[0]
    width = adaptiveFrame.shape[1]

    learning_rate = 0.05

    if z % 5 == 0:
        print("Updating the reference")
        for i in range(height-1):
            for j in range(width-1):
                #Change the value of the pixel based on a the simple adapative background
                adaptiveFrame[i,j] = adaptiveFrame[i,j]*(1-learning_rate) + gray[i,j]*learning_rate


    z = z + 1
    # set a counter for indexing coordinates
    count = 0

    # loop over the contours
    for c in cnts:
        #set a default color
        col = (0, 255, 0)
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < args["min_area"]:
            continue

        #if the contour is too large, ignore it
        if cv2.contourArea(c) > 500: #was 500
            continue

        #else generate a rectangle around it
        (x, y, w, h) = cv2.boundingRect(c)

        #inspect the intensity of the pixels in question
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        extracted = frame[y:y + h, x:x + w]
        # print("extracted", extracted.shape)

        # We should know analyze the extracted images one by one
        height = extracted.shape[0]
        width = extracted.shape[1]

        #compute the average pixel value in grayscale
        average = 0
        for i in range(width - 1):
            for j in range(height - 1):
                average = average + grayscale[j, i]

        average = average / (width * height)

        #compute the average blue pixel value
        blue = extracted[:,:,1]
        blue_average = 0
        for i in range(width - 1):
            for j in range(height - 1):
                blue_average = blue_average + blue[j, i]

        #filter high intensity pixels which would represent changes in lighting conditions
        if (average < 150):  # reject high pixel values as bees and high blue values

            # cv2.imwrite("/home/karim/Desktop/BEE/CODE/Detection_image/motion_{}.jpg".format(count), extracted)
            # append the coordinates to the array bee_coord
            center_x = x + w / 2
            center_y = y + h / 2

            #add the number of bees within the image
            number_in_image += 1


            #append the x,y coordinates to the list of coordinates of the current frame
            bee_coord["{}".format(count)] = [center_x,center_y]

            #uncomment for debugging
            #print("the coordinates of the particle are ", x + w / 2, " and ", y + h / 2)

            #we should now operate on that detected pixel
            #define a global index
            index = -999
            threshold = 30 #in pixels
            #look if the global bee dictionary is empty
            if frame_num ==0:
                #generate new instances directly
                bee["{}".format(global_counter)] = [[center_x, center_y]]
                global_counter += 1

            else:
                #look into previous coordinates to look for corresponding object
                for key, coord in bee.items():

                    #add a case in which the logs include more than a datalog we can then predict the future location of the bee



                    #this is valid only for the case in which only a single datalog is available for the object
                    #look at the last component of coord
                    #extract the x and y coordinates
                    x1 = coord[-1][0]
                    y1 = coord[-1][1]
                    #generate a temporary list
                    #compute a radius vector
                    radius = math.sqrt((x1-center_x)**2+(y1-center_y)**2)
                    #check if the component is near the previous coordinates they correspond


                    if radius < threshold:
                        threshold = radius
                        index = key

                if index == -999:
                    #there is no previous component near the current bee
                    #add a new bee component to the list
                    bee["{}".format(global_counter)] = [[center_x,center_y]]
                    global_counter += 1
                    col = (0, 255, 0)

                if index != -999:
                    #we have found a previous match at the location "index"
                    #append the new coordinates to that key
                    coordinate_list = bee["{}".format(index)]
                    coordinate_list.append([center_x, center_y])
                    col = colors["{}".format(index)]
                    status["{}".format(index)] = "present"
                    #notify if the bee match corresponds to a missing be

                    if index in to_pop:
                        #print(index, "belongs to the to_pop, bee was lost and is now found")
                        #remove the ID from the to_pop
                        to_pop.pop("{}".format(index))




            cv2.rectangle(frame, (x, y), (x + w, y + h), col, 2)
            # increase the local counter by 1
            count = count + 1








    # if the `q` key is pressed, break from the loop
    if key == ord("q"):
        break



    # generate a copy of the dictionary to include the colors of each object
    for k in bee:
        h = int(k)
        B = 40 + (h * 50)
        G = 20 + (h * 75)
        R = 60 + (h * 80)
        colors["{}".format(k)] = (B % 255, G % 255, R % 255)

    #print("the status of bees",status)
    #print("the number of bees in the frame",number_in_image)
    #print(bee)
    #inspect the status of each bee component in order to update the flow parameter

    pop = []
    shape = frame.shape
    ylow = 0
    yhigh = shape[0]
    boundary_thresh = 0.15*(yhigh-ylow) #within 12 percent of the edges
    #print("boundary tresh", boundary_thresh)
    max_misses = 7
    #print("ylow is", ylow)
    #print("yhigh is", yhigh)
    #print(counting)
    counting = counting + 1
    for k in status:
            if (status["{}".format(k)] == "absent"):
                    print("lost one bee")
                #might want to add a condition that the last coordinates be next to the edges
                 #can't predict the flow if it only appears in a single frame
                    #look at the last pair of coordinates
                    x2 = bee["{}".format(k)][-1][0]
                    y2 = bee["{}".format(k)][-1][1]
                    #print("y2 is", y2)
                    #look at the y coordinates, if they are next to the borders continue
                    #else this is a missing bee we should be flexible and wait for the next frame
                    if abs(y2 - ylow) < boundary_thresh or abs(y2 - yhigh) < boundary_thresh:
                        #print("within the applied boundaries")
                        if (len(bee["{}".format(k)]) >= 2): #It has two points at least and is within the boundary
                            #print("it has atleast two coordinates")
                            x1 = bee["{}".format(k)][-2][0]
                            y1 = bee["{}".format(k)][-2][1]
                            #look at the difference between the y components to assess if in or out
                            vector = y2 - y1
                            if vector > 0: #which means that the bee got into the hive
                                flow -= 1
                                #print("the net flow is {}".format(flow))
                            else: #which means that the bee got out of the hive
                                flow += 1
                                #print("the net flow is {}".format(flow))
                            #generate a list of the keys to pop
                            pop.append(k)
                        else: #Only have a single value but within boundary. Assume it missing might find it later on
                            #print("Absent and has a single coordinate, allow for a miss !")
                            # pop.append(k) #might want to keep the lost bee for other frames
                            print(k, "has only one coordinate and is added to the potential misses")
                            if k in to_pop:  # if the bee is already present
                                # update the counter
                                to_pop["{}".format(k)] += 1
                                #print(k, " has history, updating it to ", to_pop["{}".format(k)])
                            else:  # add it to the to_pop dictionary
                                to_pop["{}".format(k)] = 1
                                #print("Adding", k, "to the to_pop list for the first time")

                    else: #If it is not within the boundaries it must be missing
                        print(k, "not within the boundaries must be added to the missing")
                        #print("Absent but not within the boundaries, allow for a miss !")
                        #pop.append(k) #might want to keep the lost bee for other frames
                        if k in to_pop: #if the bee is already present
                            #update the counter
                            to_pop["{}".format(k)] += 1
                            #print(k, " has history, updating it to ", to_pop["{}".format(k)])
                        else: #add it to the to_pop dictionary
                            to_pop["{}".format(k)] = 1
                            #print("Adding", k, "to the to_pop list for the first time")


    for p, cnt in to_pop.items():
        if cnt >= max_misses:
            print(p, "has been missing for", cnt," long time, removing the object's ID from the logfile")
            pop.append(p)
            to_pop.pop("{}".format(p)) #move it to pop

    print("bee", bee)
    print("pop", pop)
    print("status",status)
    #remove this object from the list
    for q in pop:
        #if the ID is present in pop it has to be removed from the logs such as colors, bee and status
            bee.pop("{}".format(q))
            status.pop("{}".format(q))
            colors.pop("{}".format(q))


    #reset the status of the bees to absent
    for k in bee:
        status["{}".format(k)] = "absent"

    # show the frame and record if the user presses a key
    cv2.imshow("Entrance Feed", frame)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Thresh_OG", thresh_og)
    cv2.imshow("Frame Delta", frameDelta)
    cv2.imshow("ref",firstFrame)
    cv2.waitKey(2000)
    key = cv2.waitKey(1) & 0xFF

    print("the net flow is {}".format(flow))
    frame_num += 1


#cleanup the camera and close any open windows
vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()
