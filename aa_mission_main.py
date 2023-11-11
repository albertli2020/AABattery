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
flights = [{"name":'Panorama-full-counter-clockwise', "steps":5},
    {"name":'Panorama-full-clockwise',  "steps":5},
    {"name":'Panorama-half-counter-clockwise', "steps":4},
    {"name":'Panorama-half-clockwise', "steps":4}]
flight_number = 0

def panorama_full_clockwise(tello:Tello, sync_lock:threading.Lock):
    global motion_stage
    for i in range(4):
        tello.rotate_clockwise(80)
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

def colorAnalyzeImage(image, show_image=True):
    _, w, _ = image.shape
    half = w//2
    img = image[half:, :] 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #red Color 
    lower_red = np.array([0,100,100]) #[0,100,100]
    upper_red = np.array([10,255,255]) #[7,255,255]
    #evening green color 
    lower_eg = np.array([50//2,42*255//100,46*255//100]) #57, 51, 68; 53, 44, 55; 49, 55, 54
    upper_eg = np.array([60//2,60*255//100,72*255//100]) # 57, 50, 55; 57, 47, 57; 59, 56, 47;  55, 48, 56; not 47, 54, 51

    #arcala Green 112, 47, 42
    lower_green = np.array([98//2,32*255//100,38*255//100]) #120, 38, 47; 110, 36, 51
    upper_green = np.array([122//2,52*255//100,52*255//100]) #100, 34, 50

    #blue color 
    lower_blue = np.array([95,50,50])  # 95, 110, 110 #[90,60,0]
    upper_blue = np.array([122,255,255]) #[121,255,255]
    #purple color 
    lower_purple = np.array([250//2,7*255//100,70*255//100])  #278
    upper_purple = np.array([310//2,20*255//100,100*255//100]) #[121,255,255]

    #light-green color 
    lower_lg = np.array([157//2,6*255//100,81*255//100])  
    upper_lg = np.array([205//2,20*255//100,99*255//100]) 

    #dull-green color 
    lower_dg = np.array([76//2,32*255//100,58*255//100])  #80, 38, 67; 86, 39, 67;  80, 34, 60
    upper_dg = np.array([88//2,50*255//100,70*255//100])  #81, 38, 67; 77, 49, 60; 76, 46, 62; 79, 47, 63

    #bright-yellow color 
    lower_by = np.array([41//2,66*255//100,60*255//100])  #49, 79, 70; 50, 60, 82
    upper_by = np.array([58//2,100*255//100,88*255//100])  #56, 90, 66; 55, 97, 62

    #red2 color 
    lower_red2 = np.array([160,0,0]) #100, 100
    upper_red2 = np.array([180,255,255])


    green = cv2.inRange(hsv, lower_green, upper_green)
    blue = cv2.inRange(hsv, lower_blue, upper_blue)
    red = cv2.inRange(hsv, lower_red, upper_red)
    red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    pur = cv2.inRange(hsv, lower_purple, upper_purple)
    lg = cv2.inRange(hsv, lower_lg, upper_lg)
    dg = cv2.inRange(hsv, lower_dg, upper_dg)
    by = cv2.inRange(hsv, lower_by, upper_by)
    eg = cv2.inRange(hsv, lower_eg, upper_eg)
    red = red + red2

    objs = [red, blue, green, lg, pur, dg, by, eg]
    obj_color_strs = ['Red', 'Blue', 'Green', 'Light-Green', 'Light-Purple', 'Green Tea Mochi', 'Yellow', 'Evening-Green']
    obj_cnt_colors = [(0, 0, 255), (255, 0, 0), (64, 108, 57), (0, 100, 0), (198, 171, 213), (142, 171, 105), (208, 183, 65), (121, 120, 53)]
    
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
                if a >= 400: #1000:#5000:         
                    # draw the biggest contour (c) in green
                    #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                    #cv2.drawContours(img, [c], -1, obj_cnt_colors[oi], 5)
                    e = cv2.fitEllipse(c)
                    (_, (h, w), _) = e 
                    ea = (3.1415926/4.0)*h*w
                    r = ea/a
                    if r>0.92 and r<1.5:#15: #1.08:
                        if show_image:
                            cv2.ellipse(img, e, obj_cnt_colors[oi], 5)
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
    tello.streamon()
    time.sleep(default_command_delay_time)
    frame_read = tello.get_frame_read()
    time.sleep(picture_first_frame_delay_time)
    local_ms = 0
    last_lms = 0
    wait_count = 0
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
        of_path = os.path.join(output_folder, f'{time.time()}.jpg')
        cv2.imwrite(of_path, rgb_img)
    #cv2.imwrite("ballon_seen_stationary.png", rgb_img)
    tello.streamoff(os.path.join)
    time.sleep(default_command_delay_time)
    colorAnalyzeImage(rgb_img, show_image=False)

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
        elif tid == 9:
            offlineTask102(tello)
        elif tid == 99:
            missionTaskBeginner(tello)
    else:
        if tid == 101:
            offlineTask101()
        elif tid == 103:
            offlineTask103()
        elif tid == 104:
            offlineTask104()    

if __name__ == "__main__":
   main(sys.argv[1:])