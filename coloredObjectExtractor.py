import numpy as np
import cv2
import bisect

class ColoredObjectExtractor:
    MAX_NUM_OBJECTS_PER_COLOR = 6
    COLORS = {
        "red": {"name": 'Red',
          "hsvLow1": [  0//2,  90*255//100,  40*255//100], # 351, 84, 65
          "hsvHigh1":[ 14//2, 100*255//100,  80*255//100], # 348, 77, 83
          "hsvLow2": [340//2,  72*255//100,  40*255//100], # 349, 74, 83
          "hsvHigh2":[360//2, 100*255//100,  80*255//100], # 359, 89, 44
          "contourBgrColor": (0, 0, 255)
        },
        "darker_blue" : {"name": 'Darker Blue',
          "hsvLow1": [192//2,  54*255//100,  12*255//100], #204, 99, 44
          "hsvHigh1":[228//2, 104*255//100,  78*255//100], #201, 99, 75
          "contourBgrColor": (113, 68, 1)
        },
        "lighter_blue": {"name": 'Lighter Blue',
          "hsvLow1": [160//2,  8*255//100,  30*255//100], #204, 55, 89
          "hsvHigh1":[202//2,  30*255//100,  50*255//100], #192, 15, 40
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
        "darker_green": {"name": 'Darker Green',
          "hsvLow1": [70//2,  40*255//100,  10*255//100], #88, 61, 24, 100, 54, 16
          "hsvHigh1":[150//2,  80*255//100,  38*255//100], #101, 53, 21  #162, 47, 43
          "contourBgrColor": (39, 46, 31)
        },
        "purple": {"name": 'Purple',
          "hsvLow1": [260//2,  24*255//100,  35*255//100], #266, 48, 71
          "hsvHigh1":[312//2,  50*255//100,  76*255//100], #277, 42, 73; 278, 42, 77
          "contourBgrColor": (159, 80, 98)
        },
        "lighter_green": {"name": 'Lighter Green',
          "hsvLow1": [ 68//2,  30*255//100,  50*255//100], #110, 33, 80
          "hsvHigh1":[ 120//2,  80*255//100,  88*255//100], # to exclude 53, 42, 50
          "contourBgrColor": (48, 150, 131)
        },
        "tennis_ball": {"name": 'Tennis Ball',
          "hsvLow1": [ 50//2,  30*255//100,  55*255//100],  #52, 43, 66
          "hsvHigh1":[ 70//2,  50*255//100,  67*255//100],  #61, 26, 66
          "hsvLow2": [ 56//2,  46*255//100,  20*255//100],  #61, 61, 68  == 80
          "hsvHigh2":[ 78//2,  80*255//100,  100*255//100],  #59, 57, 72
          "contourBgrColor": (43, 166, 164)
        },
        "pink": {"name": 'Pink',
          "hsvLow1": [ 310//2,  26*255//100,  62*255//100], #308, 27, 98; 313, 34, 93
          "hsvHigh1":[ 342//2,  52*255//100,  90*255//100], #313, 33, 96; 321, 40, 86;  300, 19, 100
          "contourBgrColor": (229, 166, 247) }
    }
  
    
    def __init__(self, color_key_str, min_area, max_area):
        self.colorKey = color_key_str
        self.minArea = min_area
        self.maxArea = max_area
        # we might want to make the below an input parameter?
        if self.maxArea < 1000:
            self.maxConvexity = 1.6 #32#4#10 #1.8:#15: #1.08:
            self.minConvexity = 0.92
        else:
            self.maxConvexity = 1.3 #1.2 #1.85 #5 #8 #26 #1.8:#15: #1.08:
            self.minConvexity = 0.92
        self.sortedFilteredContours = []
        self.areaOfLargestContour = 0.0
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

    def filterContoursByArea(self, contours):
        count = len(contours)
        self.sortedFilteredContours = []
        if count > 0:
            # sort the countour (c) list by the area
            cgs = sorted(contours, key=cv2.contourArea, reverse=True)
            i = 0
            while i < count:
                a = cv2.contourArea(cgs[i])
                if a <= self.maxArea:
                    break
                else:
                    print("[COE]: Disqualified a potential", self.colorName, "object as it is considered too big:", a)
                    i += 1
            filteredCount = 0
            while i < count:
                c = cgs[i]
                a = cv2.contourArea(c)
                if a >= self.minArea:
                    self.sortedFilteredContours.append((c, a))
                    i += 1
                    filteredCount += 1
                    if filteredCount == ColoredObjectExtractor.MAX_NUM_OBJECTS_PER_COLOR:
                        break
                else:
                    break
                
    def qualifyObjectsFromAreaSortedContours(self, runIndex=0):
        """
        Check inertia and convexity of the ellipse fitted from the sorted and filtered contours.
        Note the assumption that there is a contour list sorted and filtered per area.
        """
        numQualifiedObjects = 0
        qualifiedObjects = []
        areaOfLargestQualifiedObject = None
        for (c, a) in self.sortedFilteredContours:
            e = cv2.fitEllipse(c)
            ((x, y), (h, w), _) = e
            if h < 1:
                inertiaRatio = 100000.0
            else:    
                inertiaRatio = w/h
            if inertiaRatio < 4.0:  
                ea = (3.1415926/4.0)*h*w
                convexityRatio = ea/a
                print(inertiaRatio, convexityRatio) 
                if convexityRatio>self.minConvexity and convexityRatio<self.maxConvexity: 
                    if areaOfLargestQualifiedObject is None:
                        areaOfLargestQualifiedObject = a
                        r = 1.0
                    else:
                        r = a / areaOfLargestQualifiedObject
                    if r < 0.2:        
                        print("Disqualified a potential", self.colorName, "object due to same type area ratio:", a, r, runIndex)
                    else:
                        qualifiedObjects.append((self.contourBgrColor, e, a, convexityRatio))
                        numQualifiedObjects += 1
                        print("[Great!!]Qualified a", self.colorName, "object successfully", a, r, convexityRatio, runIndex)
                else:
                    print("Disqualified a potential", self.colorName, "object due to covexity:", a, convexityRatio, runIndex)
            else:
                print("Disqualified a potential", self.colorName, "object at",  x, y, "due to iternia:", inertiaRatio, runIndex)
        return numQualifiedObjects, qualifiedObjects

    def extract(self, hsv, numIterations = 1):
        qualifiedObjectList = []
        for i in range(numIterations, 0, -1):
            inRange = cv2.inRange(hsv, self.hsvLow1, self.hsvHigh1)
            if self.hsvLow2 is not None:
                inRange += cv2.inRange(hsv, self.hsvLow2, self.hsvHigh2)
            contours, hierarchy = cv2.findContours(inRange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            self.filterContoursByArea(contours)
            count, qualifiedObjectList = self.qualifyObjectsFromAreaSortedContours(runIndex=i)
            count = len(qualifiedObjectList)
            if count == 0:
                break
            else:
                if i > 1:
                    # Use the largest object qualified to retune parameters
                    (_, e, _, _) = qualifiedObjectList[0]
                    self.autotuneParams(hsv, e)
        return qualifiedObjectList

    def autotuneParams(self, image, e):
        mask = np.zeros(image.shape[:2], dtype="uint8")
        ((x, y), (sa, la), _) = e
        cv2.circle(mask, (int(x), int(y)), int(sa/2.0), 255, -1)
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
                if percentile_5_values[0] > 90:
                    self.hsvLow2 = (percentile_5_values * .95).astype(int)
                else:
                    self.hsvLow1 = (percentile_5_values * .95).astype(int)
                if percentile_95_values[0] < 90:
                    self.hsvHigh1 = (percentile_95_values * 1.05).astype(int)
                else:  
                    self.hsvHigh2 = (percentile_95_values * 1.05).astype(int)
            else:
                self.hsvLow1 = (percentile_5_values * .95).astype(int)
                self.hsvHigh1 = (percentile_95_values * 1.05).astype(int)
            #print(self.hsvLow1, self.hsvHigh1)    
        

