import numpy as np
import cv2
class ColoredObjectExtractor:
    MAX_NUM_OBJECTS_PER_COLOR = 6
    COLORS = {
        "red": {"name": 'Red',
          "hsvLow1": [  0//2,  90*255//100,  60*255//100],
          "hsvHigh1":[ 20//2, 100*255//100,  80*255//100],
          "hsvLow2": [340//2,  90*255//100,  60*255//100],
          "hsvHigh2":[360//2, 100*255//100,  80*255//100],
          "contourBgrColor": (0, 0, 255)
        },
        "blue" : {"name": 'Blue',
          "hsvLow1": [194//2,  89*255//100,  34*255//100], #204, 99, 44
          "hsvHigh1":[214//2, 109*255//100,  78*255//100], #201, 99, 75
          "contourBgrColor": (113, 68, 1)
        },
        "light_blue": {"name": 'Light Blue',
          "hsvLow1": [198//2,  48*255//100,  57*255//100],
          "hsvHigh1":[218//2,  75*255//100,  87*255//100],
          "contourBgrColor": (172, 120, 60)
        },
        "yellow": {"name": 'Yellow',
          "hsvLow1": [ 41//2,  66*255//100,  60*255//100],
          "hsvHigh1":[ 58//2, 100*255//100,  88*255//100],
          "contourBgrColor": (6, 195, 216)
        },
        "orange": {"name": 'Orange',
          "hsvLow1": [ 18//2,  92*255//100,  52*255//100], 
          "hsvHigh1":[ 36//2, 106*255//100,  80*255//100],  
          "contourBgrColor": (3, 123, 214)
        },
        "greenbrier": {"name": 'Greenbrier',
          "hsvLow1": [120//2,  35*255//100,  45*255//100],
          "hsvHigh1":[144//2,  65*255//100,  75*255//100],
          "contourBgrColor": (93, 154, 77)
        },
        "purple_opulence": {"name": 'Purple Opulence',
          "hsvLow1": [244//2,  40*255//100,  52*255//100],
          "hsvHigh1":[264//2,  60*255//100,  72*255//100],
          "contourBgrColor": (159, 80, 98)
        },
        "light_green": {"name": 'Light Green',
          "hsvLow1": [ 55//2,  40*255//100,  40*255//100], #86, 47, 76
          "hsvHigh1":[ 90//2,  80*255//100,  90*255//100],
          "contourBgrColor": (48, 150, 131)
        } }
    
    def __init__(self, color_key_str):
        self.colorKey = color_key_str
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

    def extract(self, hsv, min_area, bgr_img=None):
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
            for i in range(count):
                c = cgs[i]
                #x,y,w,h = cv2.boundingRect(c)
                a = cv2.contourArea(c)
                if a >= min_area:#400: #1000:#5000:         
                    # draw the biggest contour (c) in green
                    #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                    #cv2.drawContours(img, [c], -1, obj_cnt_colors[oi], 5)
                    e = cv2.fitEllipse(c)
                    ((x, y), (h, w), _) = e 
                    ea = (3.1415926/4.0)*h*w
                    r = ea/a
                    if r>0.92 and r<1.5:#15: #1.08:
                        num_detected_Objects += 1
                        if bgr_img is None:
                            pass
                        else:
                            cv2.ellipse(bgr_img, e, self.contourBgrColor, 5)
                            cv2.putText(bgr_img, self.colorName, (int(x), int(y)),
                                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, self.contourBgrColor, 2, cv2.LINE_AA)
                        print(self.colorName, num_detected_Objects," detected with area =", a, r)
                        returnedObjects.append(((x,y),(h,w), a, r))
        return returnedObjects
          
