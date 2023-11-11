import sys
import os
import threading
import time
import cv2
import numpy as np

from djitellopy import Tello

default_command_delay_time = 0.7
picture_first_frame_delay_time = 4.0
motion_stage = 0
flights = [{"name":'Panorama-full-counter-clockwise', "steps":9},
    {"name":'Panorama-full-clockwise',  "steps":5},
    {"name":'Panorama-half-counter-clockwise', "steps":4},
    {"name":'Panorama-half-clockwise', "steps":4},
    {"name": 'Panorama-move-forward', "steps":6}]
flight_number = 1

def panorama_full_clockwise(tello:Tello, sync_lock:threading.Lock):
    global motion_stage
    for i in range(8):
        tello.rotate_clockwise(40)
        with sync_lock:
            motion_stage += 1
        time.sleep(1)

    tello.rotate_clockwise(40)
    with sync_lock:
        motion_stage += 1
    time.sleep(1)

def panorama_half_clockwise(tello:Tello, sync_lock:threading.Lock):
    global motion_stage
    tello.rotate_counter_clockwise(90)
    for i in range(3):
        with sync_lock:
            motion_stage += 1
        time.sleep(1)
        tello.rotate_clockwise(60)

    with sync_lock:
        motion_stage += 1
    time.sleep(1)
    tello.rotate_counter_clockwise(90)
    time.sleep(1)


def panorama_full_counter_clockwise(tello:Tello, sync_lock:threading.Lock):
    global motion_stage
    for i in range(4):
        tello.rotate_counter_clockwise(80)
        with sync_lock:
            motion_stage += 1
        time.sleep(1)

    tello.rotate_counter_clockwise(40)
    with sync_lock:
        motion_stage += 1
    time.sleep(1)
    

def panorama_half_counter_clockwise(tello:Tello, sync_lock:threading.Lock):
    global motion_stage
    tello.rotate_clockwise(90)

    for i in range(3):
        with sync_lock:
            motion_stage += 1
        time.sleep(1)
        tello.rotate_counter_clockwise(60)

    with sync_lock:
        motion_stage += 1
    time.sleep(1)
    tello.rotate_clockwise(90)
    time.sleep(1)

def panorama_move_forward(tello:Tello, sync_lock:threading.Lock):
    global motion_stage
    for i in range(5):
        with sync_lock:
            motion_stage += 1
        time.sleep(1)
        tello.move_forward(30)
    
    with sync_lock:
        motion_stage += 1
        
    time.sleep(1)        


def colorAnalyzeImage(image, show_image=True):
    _, w, _ = image.shape
    half = w//2
    img = image### [half:, :] 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #red Color 
    lower_red = np.array([0,90*255//100,60*255//100]) #[0,100,100]
    upper_red = np.array([10,100*255//100,80*255//100]) #[7,255,255]

    #blue color 
    lower_blue = np.array([201//2,89*255//100,65*255//100]) # 211, 99, 75
    upper_blue = np.array([221//2,109*255//100,85*255//100]) 
    
    # Light Green color
    lower_lgreen = np.array([61//2,58*255//100,49*255//100]) # 71, 68, 59
    upper_lgreen = np.array([81//2,78*255//100,69*255//100])
    
    #python blue color 
    lower_pb = np.array([198//2,55*255//100,57*255//100]) # 208, 65, 67
    upper_pb = np.array([218//2,75*255//100,77*255//100]) 

    #bright-yellow color 
    lower_by = np.array([41//2,66*255//100,60*255//100])  #49, 79, 70; 50, 60, 82
    upper_by = np.array([58//2,100*255//100,88*255//100])  #1706, 90, 66; 55, 97, 66

    #red2 color 
    lower_red2 = np.array([170,90*255//100,60*255//100]) #[0,100,100]
    upper_red2 = np.array([180,100*255//100,80*255//100]) #[7,255,255]

    #german mustard color 
    lower_gm = np.array([18//2,92*255//100,52*255//100]) 
    upper_gm = np.array([36//2,106*255//100,80*255//100])  
    #30, 99, 74
    #19, 100, 55
    
    # Greenbrier color
    lower_gb = np.array([122//2,40*255//100,50*255//100]) # 132, 50, 60
    upper_gb = np.array([142//2,60*255//100,70*255//100]) 

    # Purple Opulence color
    lower_po = np.array([244//2,40*255//100,52*255//100]) # 254, 50, 62
    upper_po = np.array([264//2,60*255//100,72*255//100]) 

    #green = cv2.inRange(hsv, lower_green, upper_green)
    blue = cv2.inRange(hsv, lower_blue, upper_blue)
    pb = cv2.inRange(hsv, lower_pb, upper_pb)
    red = cv2.inRange(hsv, lower_red, upper_red)
    red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    lgreen = cv2.inRange(hsv, lower_lgreen, upper_lgreen)
    by = cv2.inRange(hsv, lower_by, upper_by)
    gm = cv2.inRange(hsv, lower_gm, upper_gm)
    gb = cv2.inRange(hsv, lower_gb, upper_gb)
    po = cv2.inRange(hsv, lower_po, upper_po)
    red = red + red2

    objs = [red, blue, pb, by, gm, gb, po, lgreen]
    obj_color_strs = ['Red', 'Blue', 'Python Blue', 'Yellow', 'German-Mustard', 'Greenbrier', 'Purple Opulence', 'Light Green']
    obj_cnt_colors = [(0, 0, 255), (192, 95, 2), (172, 120, 60), (6, 195, 216), (3, 123, 214), (93, 154, 77), (159, 80, 98), (48, 150, 131)]
    
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
                if a >= 100:#400: #1000:#5000:         
                    # draw the biggest contour (c) in green
                    #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                    #cv2.drawContours(img, [c], -1, obj_cnt_colors[oi], 5)
                    e = cv2.fitEllipse(c)
                    ((x, y), (h, w), _) = e 
                    ea = (3.1415926/4.0)*h*w
                    r = ea/a
                    if r>0.92 and r<1.5:#15: #1.08:
                        if show_image:
                            cv2.ellipse(img, e, obj_cnt_colors[oi], 5)
                            cv2.putText(img, obj_color_strs[oi], (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, obj_cnt_colors[oi], 2, cv2.LINE_AA)
                        print(obj_color_strs[oi], i+1," detected with area =", a, r)
                    else:
                        pass #print(obj_color_strs[oi], i+1," disqualified with area =", a, r)    
        oi += 1            
    cv2.imwrite("after_image_analysis.jpg", img)

    if show_image:
        cv2.imshow("color analysis result",img)
        cv2.waitKey()        

def motionControl(tello:Tello, sync_lock:threading.Lock) -> int:
    global motion_stage, flight_number
    # take a command from command queue and send to drone
    tello.takeoff()
    time.sleep(default_command_delay_time)
    
    tello.move_up(20)
    time.sleep(default_command_delay_time)
    h = tello.get_height()
    time.sleep(default_command_delay_time)
    print("Hovering at height: ", h, "cm.")

    if flight_number == 0:
        panorama_full_counter_clockwise(tello, sync_lock)
    elif flight_number == 1:
        panorama_full_clockwise(tello, sync_lock)
    elif flight_number == 2:
        panorama_half_counter_clockwise(tello, sync_lock)
    elif flight_number == 3:
        panorama_half_clockwise(tello, sync_lock)
    elif flight_number == 4:
        panorama_move_forward(tello, sync_lock)
    
    #time.sleep(5)
    tello.land()
    time.sleep(default_command_delay_time)
    return 0

def perceiveObjects(tello:Tello, sync_lock:threading.Lock) -> int:
    global motion_stage
    f = flights[flight_number]
    print("Flight no = ", flight_number, f)
    #tello.streamoff()
    #time.sleep(default_command_delay_time)
    #tello.streamon()
    #time.sleep(default_command_delay_time)
    frame_read = tello.get_frame_read()
    time.sleep(picture_first_frame_delay_time)
    local_ms = 0
    last_lms = 0
    wait_count = 0
    images = []
    start_time = int(time.time())
    output_folder = os.path.join(f["name"], f'{start_time}')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for s in range(f["steps"]):
        while True:
            with sync_lock:
                local_ms = motion_stage
            if local_ms>last_lms:
                last_lms = local_ms
                break
            else:
                wait_count += 1
                time.sleep(0.1) 
        print("Waited for ", wait_count, "*100ms")
        img = frame_read.frame
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ir = cv2.rotate(rgb_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        images.append(ir)
        # cv2.imwrite("stitch_input_2.jpg", img)
        of_path = os.path.join(output_folder, f'{time.time()}.jpg')
        cv2.imwrite(of_path, rgb_img)
    #cv2.imwrite("ballon_seen_stationary.png", rgb_img)
    tello.streamoff()
    time.sleep(default_command_delay_time)
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
        colorAnalyzeImage(stitched, show_image=False)
        # display the output stitched image to our screen
    else:
        # otherwise the stitching failed, likely due to not enough keypoints)
        # being detected
        print("[INFO] image stitching failed ({})".format(status))


    
    print("... ended MST.")

def missionTaskBeginner(tello:Tello) -> int:
    global motion_stage, flights, flight_number
    sync_lock = threading.Lock()
    motion_stage = 0
    t1 = threading.Thread(target = perceiveObjects, args=(tello, sync_lock,))
    t2 = threading.Thread(target = motionControl, args=(tello, sync_lock,))

    t1.start()
    t2.start()
    t1.join()
    t2.join()
    
    return 99

def offlineTask102(tello:Tello) -> int:
    # tello.set_video_direction(Tello.CAMERA_DOWNWARD)
    time.sleep(default_command_delay_time)
    tello.streamon()
    time.sleep(default_command_delay_time)
    frame_read = tello.get_frame_read()
    time.sleep(picture_first_frame_delay_time/2)

    img = frame_read.frame
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv2.imwrite("balloon_seen_stationary2.png", rgb_img)
   
    tello.streamoff()
    time.sleep(default_command_delay_time)
    
    colorAnalyzeImage(rgb_img, show_image=True)
    #colorAnalyzeImage(img)
    return 102


def offlineTask103() -> int:
    rgb_img = cv2.imread("balloon_seen_stationary2.png")
    colorAnalyzeImage(rgb_img, show_image=True)
    #colorAnalyzeImage(img)
    return 103



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

        #tello.emergency()
        #b = 1
        if b < 15:
            print("Battery too low!!! Abort!!!")
            return
        
        #tello.streamon()
        #time.sleep(0.5)
        if tid == 9:
            offlineTask102(tello)
        elif tid == 99:
            missionTaskBeginner(tello)
        #tello.streamoff()    
    else:
        if tid == 101:
            offlineTask101()
        elif tid == 103:
            offlineTask103()
        elif tid == 104:
            offlineTask104()    

if __name__ == "__main__":
   main(sys.argv[1:])