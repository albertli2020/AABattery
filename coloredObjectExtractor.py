import numpy as np
import cv2
class ColoredObjectExtractor:
    MAX_NUM_OBJECTS_PER_COLOR = 6
    COLORS = {
        "red": {"name": 'Red',
          "hsvLow1": [  0//2,  90*255//100,  60*255//100], # 351, 84, 65
          "hsvHigh1":[ 20//2, 100*255//100,  80*255//100], # 348, 77, 83
          "hsvLow2": [340//2,  72*255//100,  60*255//100], # 349, 74, 83
          "hsvHigh2":[360//2, 100*255//100,  86*255//100],
          "contourBgrColor": (0, 0, 255)
        },
        "blue" : {"name": 'Blue',
          "hsvLow1": [194//2,  89*255//100,  34*255//100], #204, 99, 44
          "hsvHigh1":[214//2, 109*255//100,  78*255//100], #201, 99, 75
          "contourBgrColor": (113, 68, 1)
        },
        "light_blue": {"name": 'Light Blue',
          "hsvLow1": [198//2,  48*255//100,  52*255//100], #204, 55, 89
          "hsvHigh1":[218//2,  75*255//100,  92*255//100],
          "contourBgrColor": (172, 120, 60)
        },
        "yellow": {"name": 'Yellow',
          "hsvLow1": [ 41//2,  66*255//100,  60*255//100],  #60, 100, 80
          "hsvHigh1":[ 66//2, 102*255//100,  100*255//100],  #64,  77, 99
          "contourBgrColor": (6, 195, 216)
        },
        "orange": {"name": 'Orange',
          "hsvLow1": [ 18//2,  92*255//100,  52*255//100], #23, 100, 80 
          "hsvHigh1":[ 36//2, 106*255//100,  86*255//100],  
          "contourBgrColor": (3, 123, 214)
        },
        "green": {"name": 'Green',
          "hsvLow1": [120//2,  40*255//100,  40*255//100], #153, 92, 67
          "hsvHigh1":[160//2,  95*255//100,  80*255//100],
          "contourBgrColor": (93, 154, 77)
        },
        "purple": {"name": 'Purple',
          "hsvLow1": [244//2,  40*255//100,  52*255//100], #266, 48, 71
          "hsvHigh1":[280//2,  60*255//100,  80*255//100], #277, 42, 73; 278, 42, 77
          "contourBgrColor": (159, 80, 98)
        },
        "light_green": {"name": 'Light Green',
          "hsvLow1": [ 70//2,  30*255//100,  36*255//100], #110, 33, 80
          "hsvHigh1":[ 120//2,  80*255//100,  95*255//100], # to exclude 53, 42, 50
          "contourBgrColor": (48, 150, 131)
        },
        "pink": {"name": 'Pink',
          "hsvLow1": [ 295//2,  18*255//100,  76*255//100], #308, 27, 98; 313, 34, 93
          "hsvHigh1":[ 330//2,  48*255//100,  108*255//100], #313, 33, 96; 321, 40, 86;  300, 19, 100
          "contourBgrColor": (229, 166, 247) }
    }
    
    def __init__(self, color_key_str, init_min_areas=[0, 0]):
        self.colorKey = color_key_str
        self.numObjectTypes = len(init_min_areas)
        self.objectCounters = [0] * self.numObjectTypes
        self.objectDetails = [[]] * self.numObjectTypes
        #print("ColorKey is: ",  self.colorKey)
        self.colorName = ColoredObjectExtractor.COLORS[color_key_str]['name']
        self.contourBgrColor =  ColoredObjectExtractor.COLORS[color_key_str]["contourBgrColor"]
        self.hsvLow1 = np.array(ColoredObjectExtractor.COLORS[color_key_str]["hsvLow1"])
        self.hsvHigh1 = np.array(ColoredObjectExtractor.COLORS[color_key_str]["hsvHigh1"])
        if 'hsvLow2' in ColoredObjectExtractor.COLORS[color_key_str]:
            self.hsvLow2 = np.array(ColoredObjectExtractor.COLORS[color_key_str]["hsvLow2"])
            self.hsvHigh2 = np.array(ColoredObjectExtractor.COLORS[color_key_str]["hsvHigh2"])
        else:
            self.hsvLow2 = None
            self.hsvHigh2 = None

    def extract(self, hsv, min_area, bgr_img=None, autoTune = False):
        returnedObjects = []
        inRange = cv2.inRange(hsv, self.hsvLow1, self.hsvHigh1)
        if self.hsvLow2 is None:
            pass
        else:
            inRange += cv2.inRange(hsv, self.hsvLow2, self.hsvHigh2)
        cnts, hierarchy = cv2.findContours(inRange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        count = len(cnts)
        if count > 0:
            # find the biggest countour (c) by the area
            cgs = sorted(cnts, key=cv2.contourArea, reverse=True)
            if count > ColoredObjectExtractor.MAX_NUM_OBJECTS_PER_COLOR:
                count =  ColoredObjectExtractor.MAX_NUM_OBJECTS_PER_COLOR
            num_detected_Objects = 0
            maxArea = 1000000.0
            for i in range(count):
                c = cgs[i]
                #x,y,w,h = cv2.boundingRect(c)
                a = cv2.contourArea(c)
                if i == 0:
                    maxArea = a
                    ar = 1.0
                else:
                    ar = a / maxArea    
                if a >= min_area and ar > .2 :#400: #1000:#5000:         
                    # draw the biggest contour (c) in green
                    e = cv2.fitEllipse(c)
                    ((x, y), (h, w), _) = e
                    '''
                    epsilon = 0.0125*cv2.arcLength(c,True)
                    approx = cv2.approxPolyDP(c,epsilon,True)
                    if False:
                        pts = []
                        for kp in kps:
                            pts.append(kp.pt)
                        e = cv2.fitEllipse(pts)
                        ox = x - w/2
                        oy = y - h/2
                        ((x, y), (h, w), angle) = e
                        x += ox
                        y += oy
                        e = ((x, y), (h, w), angle)
                    elif False:
                        if len(kps):
                            x = kps[0].pt[0]
                            y = kps[0].pt[1]
                            h = kps[0].size
                            w = h
                            print(x, y, h, w)
                    else:
                        e = cv2.fitEllipse(approx)
                        ((x, y), (h, w), _) = e
                    '''
                    if autoTune:
                      self.autotuneParams(hsv, e)
                    ea = (3.1415926/4.0)*h*w
                    r = ea/a
                    if r>0.92 and r<1.8: #1.8:#15: #1.08:
                        num_detected_Objects += 1
                        if bgr_img is None:
                            pass
                        else:
                            cv2.ellipse(bgr_img, e, self.contourBgrColor, 5)
                            s = "{:.1f}, {:.3f}".format(a,r)
                            #s = self.colorName
                            cv2.putText(bgr_img, s, (int(x) - 50, int(y)+40),
                                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, self.contourBgrColor, 2, cv2.LINE_AA)
                        if not autoTune:
                            print(self.colorName, num_detected_Objects," detected with area =", a, r)
                        returnedObjects.append(((x,y),(h,w), a, r))

        return returnedObjects

    def autotuneParams(self, image, e):
        mask = np.zeros(image.shape[:2], dtype="uint8")
        ((x, y), (la, sa), _) = e
        cv2.circle(mask, (int(x), int(y)), int((sa+la)/4.0), 255, -1)
        masked = cv2.bitwise_and(image, image, mask=mask)

        # Exclude 0 values for each channel independently
        non_zero_values_per_channel = [masked[..., i][masked[..., i] != 0] for i in range(masked.shape[2])]

        # Check if there are non-zero values for each channel
        if all(non_zero_values.size > 0 for non_zero_values in non_zero_values_per_channel):
            # Find minimum, maximum, and mean values for each channel
            #min_values = [np.min(non_zero_values) for non_zero_values in non_zero_values_per_channel]
            #max_values = [np.max(non_zero_values) for non_zero_values in non_zero_values_per_channel]
            #mean_values = [np.mean(non_zero_values) for non_zero_values in non_zero_values_per_channel]
            percentile_5_values =np.array([np.percentile(non_zero_values, 5) for non_zero_values in non_zero_values_per_channel])
            percentile_95_values =np.array([np.percentile(non_zero_values, 95) for non_zero_values in non_zero_values_per_channel])
            # Display the results
            #print('Minimum Values (H, S, V):', min_values)
            #print('Maximum Values (H, S, V):', max_values)
            #print('5% Values (H, S, V):', percentile_5_values)
            #print('95% Values (H, S, V):', percentile_95_values)
            #print(self.hsvLow1, self.hsvHigh1)
            
            if self.hsvLow2 is not None:
                if percentile_5_values[0] >90:
                  self.hsvLow2 = (percentile_5_values * .95).astype(int)
                else:
                  self.hsvLow1 = (percentile_5_values * .95).astype(int)  
                self.hsvHigh2 = (percentile_95_values * 1.05).astype(int)
            else:
                self.hsvLow1 = (percentile_5_values * .95).astype(int)
                self.hsvHigh1 = (percentile_95_values * 1.05).astype(int)
            #print(self.hsvLow1, self.hsvHigh1)    


        

