import sys
import time
import cv2
import numpy as np

from djitellopy import Tello

default_command_delay_time = 0.7
picture_first_frame_delay_time = 4.0

def colorAnalyzeImage(img, show_image=True):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #red Color 
    lower_red = np.array([0,100,100]) #[0,100,100]
    upper_red = np.array([10,255,255]) #[7,255,255]
    #green color 
    lower_green = np.array([40,50,50])
    upper_green = np.array([93,255,255])
    #blue color 
    lower_blue = np.array([95,50,50])  # 95, 110, 110 #[90,60,0]
    upper_blue = np.array([122,255,255]) #[121,255,255]
    #red2 color 
    lower_red2 = np.array([160,0,0]) #100, 100
    upper_red2 = np.array([180,255,255])

    green = cv2.inRange(hsv, lower_green, upper_green)
    blue = cv2.inRange(hsv, lower_blue, upper_blue)
    red = cv2.inRange(hsv, lower_red, upper_red)
    red2 = cv2.inRange(hsv, lower_red2, upper_red2)

    red = red + red2
    red_color_str = 'Red'

    cnts_red, hierarchy = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(cnts_red) > 0:
        #cv2.drawContours(img, cnts_red, -1, 255, 3)
        # find the biggest countour (c) by the area
        c = max(cnts_red, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        if show_image:
            # draw the biggest contour (c) in red
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        if w>25 and h>25: 
            print(red_color_str, " detected with width, height =", w, h)

    cnts_blue, hierarchy = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(cnts_blue) > 0:
        #cv2.drawContours(img, cnts_blue, -1, 255, 3)
        # find the biggest countour (c) by the area
        c = max(cnts_blue, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        if show_image:
            # draw the biggest contour (c) in blue
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        if w>25 and h>25:     
            print("Blue detected with width, height =", w, h)
        
    cnts_green, hierarchy = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(cnts_green) > 0:
        # find the biggest countour (c) by the area
        c = max(cnts_green, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        if show_image:
            # draw the biggest contour (c) in green
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        if w>25 and h>25:     
            print("Green detected with width, height =", w, h)

    if show_image:
        cv2.imshow("color analysis result",img)
        cv2.waitKey()

def colorAnalyzeImage2(img, show_image=True):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #red Color 
    lower_red = np.array([0,100,100]) #[0,100,100]
    upper_red = np.array([10,255,255]) #[7,255,255]
    #green color 
    lower_green = np.array([33,50,50]) #40
    upper_green = np.array([93,255,255])
    #blue color 
    lower_blue = np.array([95,50,50])  # 95, 110, 110 #[90,60,0]
    upper_blue = np.array([122,255,255]) #[121,255,255]
    #purple color 
    lower_purple = np.array([250//2,7*255//100,70*255//100])  #278
    upper_purple = np.array([310//2,20*255//100,100*255//100]) #[121,255,255]

    #light-green color 
    #lower_lg = np.array([157//2,6*255//100,81*255//100])  
    #upper_lg = np.array([205//2,20*255//100,99*255//100]) 
    lower_lg = np.array([180//2,54*255//100,23*255//100])  
    upper_lg = np.array([200//2,86*255//100,43*255//100])

    lower_ly = np.array([175//2,70*255//100,40*255//100])  
    upper_ly = np.array([195//2,100*255//100,50*255//100]) 

    #red2 color 
    lower_red2 = np.array([160,0,0]) #100, 100
    upper_red2 = np.array([180,255,255])


    green = cv2.inRange(hsv, lower_green, upper_green)
    blue = cv2.inRange(hsv, lower_blue, upper_blue)
    red = cv2.inRange(hsv, lower_red, upper_red)
    red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    pur = cv2.inRange(hsv, lower_purple, upper_purple)
    lg = cv2.inRange(hsv, lower_lg, upper_lg)
    ly = cv2.inRange(hsv, lower_ly, upper_ly)

    red = red + red2
    red_color_str = 'Red'

    objs = [red, blue, green, lg, pur, ly]
    obj_color_strs = ['Red', 'Blue', 'Green', 'Light-Green', 'Light-Purple', 'Light-Yellow']
    obj_cnt_colors = [(0, 0, 255), (255, 0, 0), (100, 0, 0), (0, 100, 0), (198, 171, 213), (1, 96,114)]
    
    oi = 0
    for o in objs:
        cnts, hierarchy = cv2.findContours(o, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        count = len(cnts)
        if count > 0:
            # find the biggest countour (c) by the area
            cgs = sorted(cnts, key=cv2.contourArea, reverse=True)
            if count > 6:
                count = 6
            for i in range(count):
                c = cgs[i]
                #x,y,w,h = cv2.boundingRect(c)
                a = cv2.contourArea(c)
                if a >= 80: #500:#5000:         
                    # draw the biggest contour (c) in green
                    #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                    #cv2.drawContours(img, [c], -1, obj_cnt_colors[oi], 5)
                    e = cv2.fitEllipse(c)
                    (_, (h, w), _) = e 
                    ea = (3.1415926/4.0)*h*w
                    r = ea/a
                    if r>0.92 and r<1.16: #1.08:
                        if show_image:
                            cv2.ellipse(img, e, obj_cnt_colors[oi], 5)
                        print(obj_color_strs[oi], i+1," detected with area =", a, r)
                    else:
                        print(obj_color_strs[oi], i+1," disqualified with area =", a, r)
        oi += 1            
    cv2.imwrite("after_image_analysis.jpg", img)

    if show_image:
        cv2.imshow("color analysis result",img)
        cv2.waitKey()        

def qualificationStageTask1(tello:Tello) -> int:
    """
    Take-off autonomously, hover for 15 sec, and land.
    Points: 5 if successful; 0 if failed
    """
    print("Starting QST_1...")
    hover_step_time = 15.0/2.0
    if hover_step_time < default_command_delay_time:
        hover_step_time = default_command_delay_time
    tello.takeoff()
    time.sleep(hover_step_time)
    
    h = tello.get_height()
    time.sleep(hover_step_time)
    print("Hovered at height: ", h, "cm.")
    tello.land()
    time.sleep(default_command_delay_time)

    print("... ended QST_1.")
    return 5

def qualificationStageTask2(tello:Tello) -> int:
    """
    Take-off autonomously, fly square pattern with 100 cm sides, and land.
    Points: 5 if successful; 0 if failed
    """
    print("Starting QST_2...")
    side_length = 100

    tello.takeoff()
    time.sleep(default_command_delay_time)

    helf_len = side_length//2
    print("Step1: flying to left for", helf_len, " cm.")
    tello.go_xyz_speed(0, helf_len, 0, 30)
    time.sleep(default_command_delay_time)
    print("Step2: flying forward for", side_length, " cm.")
    tello.go_xyz_speed(side_length, 0, 0, 30)
    time.sleep(default_command_delay_time)
    print("Step3: flying to right for", side_length, " cm.")
    tello.go_xyz_speed(0, 0-side_length, 0, 30)
    time.sleep(default_command_delay_time)
    print("Step4: flying backwards for", side_length, " cm.")
    tello.go_xyz_speed(0-side_length, 0, 0, 30)
    time.sleep(default_command_delay_time)
    print("Step5: flying left for", helf_len, " cm.")
    tello.go_xyz_speed(0, helf_len, 0, 30)
    time.sleep(default_command_delay_time)

    tello.land()
    time.sleep(default_command_delay_time)
    print("... ended QST_2.")
    return 5

def qualificationStageTask3(tello:Tello) -> int:
    """
    Take-off autonomously, fly diamond pattern with 100 cm sides, and land.
    Points: 10 if successful; 0 if failed
    """
    print("Starting QST_3...")
    side_length = 80

    tello.takeoff()
    time.sleep(default_command_delay_time)

    print("Step1: turn counter clokwise for 30 degrees")
    tello.rotate_counter_clockwise(30)
    time.sleep(default_command_delay_time)
    print("Step2: flying forward for", side_length, " cm.")
    tello.move_forward(side_length)
    time.sleep(default_command_delay_time)
    print("Step3: turn clokwise for 60 degrees")
    tello.rotate_clockwise(60)
    time.sleep(default_command_delay_time)
    print("Step4: flying forward for", side_length, " cm.")
    tello.move_forward(side_length)
    time.sleep(default_command_delay_time)
    
    print("Step5: turn clokwise for 120 degrees")
    tello.rotate_clockwise(120)
    time.sleep(default_command_delay_time)
    print("Step6: flying forward for", side_length, " cm.")
    tello.move_forward(side_length)
    time.sleep(default_command_delay_time)
    print("Step7: Turn clokwise for 60 degrees")
    tello.rotate_clockwise(60)
    time.sleep(default_command_delay_time)
    print("Step8: flying forward for", side_length, " cm.")
    tello.move_forward(side_length)
    time.sleep(default_command_delay_time)

    print("Step5: turn clokwise for 150 degrees")
    tello.rotate_clockwise(150)
    time.sleep(default_command_delay_time)

    tello.land()
    time.sleep(default_command_delay_time)

    print("... ended QST_3.")
    return 10


def qualificationStageTask4(tello:Tello) -> int:
    """
    Take-off autonomously, fly circle pattern, and land.
    Points: 10 if successful; 0 if failed
    """
    print("Starting QST_4...")
    r = 55
    if r < 50:
        r = 50
    d = 2*r

    tello.takeoff()
    time.sleep(default_command_delay_time)

    tello.curve_xyz_speed(r, r, 0, d, 0, 0, 50)
    time.sleep(default_command_delay_time/2.0)
    h = tello.get_height()
    print("Circling at height: ", h, "cm.")
    time.sleep(default_command_delay_time/2.0)
    r = 0 - r
    d = 0 - d
    tello.curve_xyz_speed(r, r, 0, d, 0, 0, 50)
    time.sleep(default_command_delay_time)
    
    tello.land()
    time.sleep(default_command_delay_time)

    print("... ended QST_4.")
    return 10

def qualificationStageTask5(tello:Tello) -> int:
    """
    Take-off autonomously, take a team selfie, and land.
    Points: 20 if successful; 0 if failed
    """
    print("Starting QST_5...")
    tello.streamon()
    start_time = time.time()
    time.sleep(default_command_delay_time)
    frame_read = tello.get_frame_read()

    tello.takeoff()
    time.sleep(default_command_delay_time)

    
    #time.sleep(picture_first_frame_delay_time/2)
    delta_time = picture_first_frame_delay_time - (time.time() - start_time)
    if delta_time > 0:
         time.sleep(delta_time)
    img = frame_read.frame
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite("team_selfie.png", rgb_img)

    tello.streamoff()
    time.sleep(default_command_delay_time)

    tello.land()
    time.sleep(default_command_delay_time)
    
    print("... ended QST_5.")
    return 20

def qualificationStageTask6(tello:Tello) -> int:
    """
    Take-off autonomously, read an ArUCo tag (show on console), and land.
    Points: 20 if successful; 0 if failed
    """
    print("Starting QST_6...")
    tello.streamon()
    start_time = time.time()
    time.sleep(default_command_delay_time)
    frame_read = tello.get_frame_read()

    tello.takeoff()
    time.sleep(default_command_delay_time)

    #time.sleep(picture_first_frame_delay_time/2)
    delta_time = picture_first_frame_delay_time - (time.time() - start_time)
    if delta_time > 0:
         time.sleep(delta_time)
    gray = cv2.cvtColor(frame_read.frame, cv2.COLOR_BGR2GRAY)
    tello.streamoff()
    time.sleep(default_command_delay_time)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters =  cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
    if corners is not None and ids is not None:
        #for i, corner in enumerate(corners):
        #    print('ID[', i, ']:', ids[i][0])
        print("ArUCo Tag detected as ID =", ids[0][0])
    
    tello.land()
    time.sleep(default_command_delay_time)
    
    print("... ended QST_6.")
    return 20    

def qualificationStageTask7(tello:Tello) -> int:
    """
    Take-off autonomously, pop a balloon in front, and land.
    The balloon can be either mounted onto a wall or mounted on a stick.
    Points: 30 if successful; 0 if failed
    """
    print("Starting QST_7...")

    tello.streamon()
    time.sleep(default_command_delay_time)
    start_time = time.time()
    frame_read = tello.get_frame_read()

    d1 = 50 # assuming the ballon is 100 cm in front of the drone, straight
    d2 = 50
    tello.takeoff()
    time.sleep(default_command_delay_time)
   
    #time.sleep(picture_first_frame_delay_time)

    print("STEP1: flying forward for", d1, " cm, to take a picture after popping the balloon.")
    tello.move_forward(d1)
    time.sleep(default_command_delay_time)

    delta_time = picture_first_frame_delay_time - (time.time() - start_time)
    if delta_time > 0:
         time.sleep(delta_time)
    img = frame_read.frame
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite("balloon_popped_before.png", rgb_img)

    print("STEP2: flying forward for", d2, " cm, to pop the balloon.")
    tello.move_forward(d2)
    time.sleep(default_command_delay_time)

    print("STEP3: flying backwards for", d2, " cm, to take a picture after popping the balloon.")
    tello.move_back(d2)
    time.sleep(default_command_delay_time)

    img = frame_read.frame
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite("balloon_popped_after.png", rgb_img)
    tello.streamoff()
    time.sleep(default_command_delay_time)

    print("STEP4: flying backwards for", d1, " cm, to go back to starting point.")
    tello.move_back(d1)
    time.sleep(default_command_delay_time)
    
    tello.land()
    time.sleep(default_command_delay_time)
    print("... ended QST_7.")
    return 30


def qualificationStageTask8(tello:Tello) -> int:
    """
    Take-off autonomously, recognize balloon color, and land.
    For judging purposes, please demonstrate with blue, red, and green balloons.
    Points: 50 if successful; 0 if failed
    """
    print("Starting QST_8...")

    tello.streamon()
    start_time = time.time()
    time.sleep(default_command_delay_time)
    frame_read = tello.get_frame_read()
    time.sleep(default_command_delay_time)

    tello.takeoff()
    time.sleep(default_command_delay_time)
    
    tello.move_forward(50)
    time.sleep(default_command_delay_time)
    h = tello.get_height()
    time.sleep(default_command_delay_time)
    print("Hovering at height: ", h, "cm.")

    delta_time = time.time() - start_time - picture_first_frame_delay_time
    if delta_time > 0:
         time.sleep(delta_time)
    #time.sleep(picture_first_frame_delay_time)

    img = frame_read.frame
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv2.imwrite("ballon_seen.png", rgb_img)
   
    tello.streamoff()
    time.sleep(default_command_delay_time)
    
    colorAnalyzeImage(rgb_img, show_image=False)
   
    tello.land()
    time.sleep(default_command_delay_time)
    print("... ended QST_8.")
    return 50

def offlineTask101() -> int:
    img = cv2.imread("ballon_seen.png")
    #img = cv2.imread("green_b.png")
    colorAnalyzeImage(img)
    return 101

def offlineTask103() -> int:
    img = cv2.imread("aruco_tag.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters =  cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    corners, ids, _ = detector.detectMarkers(gray)
    if corners is not None and ids is not None:
        #for i, corner in enumerate(corners):
        #    print('ID[', i, ']:', ids[i][0])
        print("ArUCo Tag detected as ID =", ids[0][0])
    return 101


def offlineTask102(tello:Tello) -> int:
    tello.streamon()
    time.sleep(default_command_delay_time)
    frame_read = tello.get_frame_read()
    time.sleep(picture_first_frame_delay_time)

    img = frame_read.frame
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv2.imwrite("ballon_seen_stationary.png", rgb_img)
   
    tello.streamoff()
    time.sleep(default_command_delay_time)
    
    colorAnalyzeImage(rgb_img, show_image=True)
    colorAnalyzeImage(img)
    return 102

def offlineTask104() -> int:
    #img = cv2.imread("stitch_output_wo_contours.jpg")
    #img = cv2.imread("stitch_input_1.jpg")
    img = cv2.imread("IMG_8255.jpg")
    colorAnalyzeImage2(img)
    return 104

def missionStageTask(tello:Tello) -> int:
    """
    Take-off autonomously, take four photos
    Points: 20 if successful; 0 if failed
    """
    print("Starting MST...")
    tello.streamon()
    time.sleep(default_command_delay_time)
    #tello.set_video_fps(Tello.FPS_15)
    # time.sleep(default_command_delay_time)
    #tello.set_video_resolution(Tello.RESOLUTION_720P)
    #time.sleep(default_command_delay_time)

    start_time = time.time()
    time.sleep(default_command_delay_time)
    frame_read = tello.get_frame_read()

    tello.takeoff()
    
    time.sleep(default_command_delay_time)
   
    tello.move_up(75) #need to do this without waiting 
    delta_time = picture_first_frame_delay_time - (time.time() - start_time)
    if delta_time > 0:
         time.sleep(delta_time)
    images = []
    bgr_img = frame_read.frame.copy()
    img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    ir = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    images.append(ir)
    cv2.imwrite("stitch_input_1.jpg", img)
    tello.move_forward(30) #need to do this without waiting
    time.sleep(default_command_delay_time)
    bgr_img = frame_read.frame.copy()
    img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    ir = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    images.append(ir)
    cv2.imwrite("stitch_input_2.jpg", img)

    tello.move_forward(30) #need to do this without waiting
    time.sleep(default_command_delay_time)
    bgr_img = frame_read.frame.copy()
    img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    ir = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    images.append(ir)
    cv2.imwrite("stitch_input_3.jpg", img)
    '''
    tello.rotate_clockwise(180) #need to do this w
    time.sleep(default_command_delay_time)
    bgr_img = frame_read.frame.copy()
    img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    ir = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    images.append(ir)
    cv2.imwrite("stitch_input_4.jpg", img)

    tello.move_forward(30) #need to do this without waiting 
    time.sleep(default_command_delay_time)
    bgr_img = frame_read.frame.copy()
    img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    ir = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    images.append(ir)
    cv2.imwrite("stitch_input_5.jpg", img)
    
    tello.move_forward(30) #need to do this without waiting
    time.sleep(default_command_delay_time)
    bgr_img = frame_read.frame.copy()
    img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    ir = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    images.append(ir)
    cv2.imwrite("stitch_input_6.jpg", img)
    '''
    tello.streamoff()
    time.sleep(default_command_delay_time)
    tello.land() #need to find out how to do the motion is a separate thread
    time.sleep(default_command_delay_time)

    # initialize OpenCV's image sticher object and
    # stitching
    print("[INFO] stitching images...")
    stitcher = cv2.Stitcher_create()

    (status, stitched_br) = stitcher.stitch(images)
    if status == 0:
        stitched = cv2.rotate(stitched_br, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite("stitch_output_wo_contours.jpg", stitched)
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
        # display the output stitched image to our screen
        cv2.imshow("Stitched", stitched)
        cv2.waitKey(0)
    else:
        # otherwise the stitching failed, likely due to not enough keypoints)
        # being detected
        print("[INFO] image stitching failed ({})".format(status))


    
    print("... ended MST.")
    return 20000                  

def main(argv):
    tid = int(argv[0])
    print("QST TID:", tid)
    if tid < 100:
        tello = Tello(retry_count=1)
        # When connecting the drone to a home/office AP, we would need to find
        # the drone's DHCP assigned IP address and provide it as the host argument.
        # tello = Tello(host='192.168.50.170', retry_count=0)

        tello.connect()
        time.sleep(default_command_delay_time)

        b = tello.get_battery()
        time.sleep(default_command_delay_time)
        print("Battery level: ", b)

        if b < 6:
            print("Battery too low!!! Abort!!!")
            return

        if tid == 1:
            qualificationStageTask1(tello)
        elif tid == 2:
            qualificationStageTask2(tello)
        elif tid == 3:
            qualificationStageTask3(tello)
        elif tid == 4:
            qualificationStageTask4(tello)
        elif tid == 5:
            qualificationStageTask5(tello)
        elif tid == 6:
            qualificationStageTask6(tello)
        elif tid == 7:
            qualificationStageTask7(tello)
        elif tid == 8:
            qualificationStageTask8(tello)
        elif tid == 9:
            offlineTask102(tello)
        elif tid == 99:
            missionStageTask(tello)
    else:
        if tid == 101:
            offlineTask101()
        elif tid == 103:
            offlineTask103()
        elif tid == 104:
            offlineTask104()    

if __name__ == "__main__":
   main(sys.argv[1:])