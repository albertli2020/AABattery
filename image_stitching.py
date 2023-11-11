
import numpy as np
import cv2
# grab the paths to the input images and initialize our images list
print("[INFO] loading images...")
#imagePaths = ['p1.jpg', 'p1_2.jpg', 'p2.jpg'] #, 'p2_3.jpg', 'p3.jpg']
#imagePaths = ['p1_c.jpg', 'p2_c.jpg', 'p3_c.jpg', 'p4_c.jpg']#, 'p5_c.jpg' ] #, 'p2_3.jpg', 'p3.jpg']
#imagePaths = ['9_p1.jpg', '9_p2.jpg', '9_p3.jpg', '9_p4.jpg', '9_p5.jpg', '9_p6.jpg', '9_p7.jpg', '9_p8.jpg', '9_p9.jpg', '9_p10.jpg', '9_p11.jpg']
#imagePaths = ['9_p1.jpg', '9_p2.jpg', '9_p3.jpg', '9_p4.jpg','9_p5.jpg',  '9_p6.jpg', '9_p7.jpg', '9_p8.jpg']#, '9_p11.jpg']
#imagePaths = ['stitch_input_interm.jpg',  'p111_1.jpg', '9_p10.jpg', '9_p11.jpg']
#imagePaths = ['p111_1.jpg', '9_p11.jpg']
imagePaths = ['Panorama-full-counter-clockwise_1699662716.204457.jpg',
              'Panorama-full-counter-clockwise_1699662719.187625.jpg',
              'Panorama-full-counter-clockwise_1699662722.161237.jpg',
              'Panorama-full-counter-clockwise_1699662725.004979.jpg',
              'Panorama-full-counter-clockwise_1699662727.2628832.jpg']

output_path = "stitch_output_pfcw.jpg"
images = []
# loop over the image paths, load each one, and add them to our
# images to stich list
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    images.append(image)
# initialize OpenCV's image sticher object and then perform the image
# stitching
print("[INFO] stitching images...")
stitcher = cv2.Stitcher_create(mode = cv2.Stitcher_PANORAMA) #cv2.Stitcher_SCANS
stitcher.setPanoConfidenceThresh(0.0)
#cv2.Stitcher_create()
(status, stitched) = stitcher.stitch(images)

# if the status is '0', then OpenCV successfully performed image
# stitching
if status == 0:
    # check to see if we supposed to crop out the largest rectangular
    # region from the stitched image
    if False:
        # create a 10 pixel border surrounding the stitched image
        print("[INFO] cropping...")
        stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10,
            cv2.BORDER_CONSTANT, (0, 0, 0))
        # convert the stitched image to grayscale and threshold it
        # such that all pixels greater than zero are set to 255
        # (foreground) while all others remain 0 (background)
        gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

        # find all external contours in the threshold image then find
        # the *largest* contour which will be the contour/outline of
        # the stitched image
        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        #cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # allocate memory for the mask which will contain the
        # rectangular bounding box of the stitched image region
        mask = np.zeros(thresh.shape, dtype="uint8")
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

        # create two copies of the mask: one to serve as our actual
        # minimum rectangular region and another to serve as a counter
        # for how many pixels need to be removed to form the minimum
        # rectangular region
        minRect = mask.copy()
        sub = mask.copy()
        # keep looping until there are no non-zero pixels left in the
        # subtracted image
        while cv2.countNonZero(sub) > 0:
            # erode the minimum rectangular mask and then subtract
            # the thresholded image from the minimum rectangular mask
            # so we can count if there are any non-zero pixels left
            minRect = cv2.erode(minRect, None)
            sub = cv2.subtract(minRect, thresh)

        # find contours in the minimum rectangular mask and then
        # extract the bounding box (x, y)-coordinates
        cnts, _ = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(c)
        # use the bounding box coordinates to extract the our final
        # stitched image
        stitched = stitched[y:y+h, x:x+w]
    else:
        pass #stitched = stitched[10:, 8000:14000]

    cv2.imwrite(output_path, stitched)
    '''
    # convert the resized image to grayscale, blur it slightly,
    # and threshold it
    gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (91, 91), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    # find contours in the thresholded image and initialize the
    # shape detector
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP, #cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(stitched, cnts, -1, 255, 3)

    # write the output stitched image to disk
    cv2.imwrite("stitch_output_with_contours.jpg", stitched)
    '''
    # display the output stitched image to our screen
    cv2.imshow("Stitched", stitched)
    cv2.waitKey(0)
else:
    # otherwise the stitching failed, likely due to not enough keypoints)
    # being detected
    print("[INFO] image stitching failed ({})".format(status))