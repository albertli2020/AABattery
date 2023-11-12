import sys
import os
import threading
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

from djitellopy import Tello
from coloredObjectExtractor import ColoredObjectExtractor

default_command_delay_time = 0.1 #7
picture_first_frame_delay_time = 4.0
motion_stage = 0

flights = [{"name":'Panorama-full-counter-clockwise', "steps":9},
    {"name":'Panorama-full-clockwise',  "steps":5},
    {"name":'Panorama-half-counter-clockwise', "steps":4},
    {"name":'Panorama-half-clockwise', "steps":4},
    {"name": 'Panorama-move-forward', "steps":6}]
flight_number = 3

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

numObjectsDetectedForColor = numObjectsDetectedForColor = {
    "red": 0,
    "blue": 0,
    "python_blue": 0,
    "yellow": 0,
    "german_mustard": 0,
    "greenbrier": 0,
    "purple_opulence": 0,
    "ligh_green": 0 }
colorKeysAreasToDetect = [("red", 150),
                           ("blue", 150),
                           ("python_blue", 150),
                           ("yellow", 200),
                           ("german_mustard", 150),
                           ("greenbrier", 150),
                           ("purple_opulence", 150),
                           ("ligh_green", 80)]

def colorAnalyzeImage(image, show_image=True, saveAnalyzedImage=True):
    global numObjectsDetectedForColor, colorKeysAreasToDetect
    _, w, _ = image.shape
    half = w//2
    img = image### [half:, :] 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #hsv[:, :, 0] = hsv[:, :, 0] * 1.05
    #hsv[:, :, 1] = hsv[:, :, 1] * .95
    #hsv[:, :, 2] = hsv[:, :, 2] * .95
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) 

    for (colorKey, minArea) in colorKeysAreasToDetect:
        coe = ColoredObjectExtractor(colorKey)
        objs = coe.extract(hsv, minArea, img)
        n = numObjectsDetectedForColor[colorKey]
        nn = len(objs)
        if n < nn:
            numObjectsDetectedForColor[colorKey] = nn

    if saveAnalyzedImage:
        cv2.imwrite("after_image_analysis.jpg", img)

    if show_image:
        cv2.imshow("color analysis result",img)
        cv2.waitKey()
    
    return img

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














def recordAndShowFrames(tello:Tello):
    global numObjectsDetectedForColor
    frame_read = tello.get_frame_read()
    #h, w, _ = frame_read.frame.shape
    #v = cv2.VideoWriter('video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 5, (w, h))
    #v = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*'XVID'), 5, (w, h))
    #create two subplots
    img = frame_read.frame
    rgb_after = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #fig, ax = plt.subplots()
    #f, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]})
    plt.figure(figsize=(12, 9))
    #a0.axis("off")
    #a0.set(title="Waiting for streaming to start...")
    #colors = list(numObjectsDetectedForColor.keys())
    #vs = list(numObjectsDetectedForColor.values())
    #create two image plots
    plt.imshow(rgb_after)
    #a1.bar(vs, colors, color ='maroon', width = 0.4)
    #a1.set(xlabel="Max No. of colored objects", ylabel="Colors", title ="Objects detected per color")
    plt.ion()
    for fn in range(24):
        start_time = time.time()
        img = frame_read.frame
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #rgb_before = rgb_img.copy()
        #v.write(rgb_img)
        img = colorAnalyzeImage(rgb_img, show_image=False, saveAnalyzedImage=False)
        rgb_after = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_after)
    
        #b = tello.get_battery()
        #plt.title(f'Frame Number: {fn+1}, battery level: {b}')
        plt.title(f'Frame Number: {fn+1}')
        #print("Battery level: ", b)
        delta_time = (1/5.0) - (time.time()-start_time)
        if delta_time > 0:
            plt.pause(delta_time)

    tello.streamoff()
    time.sleep(default_command_delay_time)

    plt.ioff() # due to infinite loop, this gets never called.
    plt.show()

    plt.figure(figsize=(12, 9))
    colors = list(numObjectsDetectedForColor.keys())
    vs = list(numObjectsDetectedForColor.values())
    plt.barh(colors, vs, color ='maroon', height = 0.4)
    plt.xlabel("Max No. of colored objects")
    plt.ylabel("Colors")
    plt.title("Objects detected per color")
    plt.show()
        
    #v.release()

def stationaryTask9(tello:Tello) -> int:
    
    
    
    
    # tello.set_video_direction(Tello.CAMERA_DOWNWARD)
    recordAndShowFrames(tello)
    return 9

def missionTaskBeginner(tello:Tello) -> int:
    global motion_stage, flights, flight_number
    sync_lock = threading.Lock()
    motion_stage = 0
    #t1 = threading.Thread(target = perceiveObjects, args=(tello, sync_lock,))
    t2 = threading.Thread(target = motionControl, args=(tello, sync_lock,))

    tello.streamoff()
    time.sleep(default_command_delay_time)
        
    tello.streamon()
    time.sleep(default_command_delay_time)
    #t1.start()
    t2.start()
    #t1.join()
    stationaryTask9(tello)
    t2.join()
    
    return 99


def offlineTask103() -> int:
    rgb_img = cv2.imread("balloon_seen_stationary2.png")
    colorAnalyzeImage(rgb_img, show_image=True)
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
            stationaryTask9(tello)
        elif tid == 99:
            missionTaskBeginner(tello)
        #tello.streamoff()    
    else:
        if tid == 101:
            pass #offlineTask101()
        elif tid == 103:
            offlineTask103()
        elif tid == 104:
            pass #offlineTask104()    

if __name__ == "__main__":
   main(sys.argv[1:])