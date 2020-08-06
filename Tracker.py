import cv2

# Image of cars on road
# img_file = 'image2.jpg'

# Video clip of cars on road
# video = cv2.VideoCapture('video1.mp4')     # the video location goes here
video_With_Pedestrian = cv2.VideoCapture('videoPedestrain.mp4')       # the video location goes here

# The pre trained car classifier
classifier_file_for_cars = "cars_detector.xml"
classifier_file_for_Pedestrian = "Pedestrians_detector.xml"

# // Video Clip //---------------------------------------------------------------

# Create the car classifier and a pedestrian classifier
car_tracker = cv2.CascadeClassifier(classifier_file_for_cars)
pedestrian_tracker = cv2.CascadeClassifier(classifier_file_for_Pedestrian)


while True:
    # Read the current frame
    (read_successful, frame) = video_With_Pedestrian .read()

    # Safe coding
    if read_successful:
        # must convert to grayscale
        grayScaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # Detect the cars AND pedestrians
    cars = car_tracker.detectMultiScale(grayScaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayScaled_frame)

    # print(cars)   # this displays the number of cars detected from the input given and returns an array of coordinates data

    # Drawing rectangles around the cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x+3, y+3), (x + w, y + h), (0, 0, 255), 2)

    # Drawing rectangles around the pedestrians
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # Display the frame
    cv2.imshow('Car and Pedestrian Tracking',frame)

    # waiting for a key
    key = cv2.waitKey(1)

    # Stop if Q key is pressed
    if key == 81 or key == 113:
        break

# release the video capture object ( just a clean up )
video_With_Pedestrian.release()

# -------------------------------------------------------------------------------------------




#  // Images Detection // --------------------------------------------------
# # We are pulling the "image.jpg" into the program using opencv
# img = cv2.imread(img_file)
#
# # Converting the image into B/W image (grayscale image)
# black_and_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#
# # Create the car classifier
# car_tracker = cv2.CascadeClassifier(classifier_file)
#
# # Detect the cars
# cars = car_tracker.detectMultiScale(black_and_white)
#
# # print(cars)     # this displays the number of cars detected from the input given and returns an array of coordinates data
#
# # Drawing rectangles around the cars
# for(x, y, w, h) in cars:
#     cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
#
#
# # Displaying the image
# cv2.imshow('Car and Pedestrian Tracking', img)
#
# # Wait till a key is pressed to close
# cv2.waitKey()
# --------------------------------------------------------------------------------------------------

# Code to check if the above code ran successfully
print('Code Completed')