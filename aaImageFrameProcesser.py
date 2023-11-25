import os
import glob
import threading
import time
import cv2
import numpy as np
import math
from queue import Queue
import csv
from coloredObjectExtractor import ColoredObjectExtractor

balloon_min_area = 1100
balloon_max_area = 15000

colorKeyedObjectsDetectionConfigAndData = {
    'red': {'count': 0, 'min_area':balloon_min_area, 'max_area':balloon_max_area},
    'darker_blue': {'count': 0, 'min_area':balloon_min_area, 'max_area':balloon_max_area},
    'lighter_blue': {'count': 0, 'min_area':balloon_min_area*3//4, 'max_area':balloon_max_area},
    'yellow': {'count': 0, 'min_area':balloon_min_area, 'max_area':balloon_max_area},
    'orange': {'count': 0, 'min_area':balloon_min_area, 'max_area':balloon_max_area},
    'darker_green': {'count': 0, 'min_area':balloon_min_area, 'max_area':balloon_max_area},
    'purple': {'count': 0, 'min_area':balloon_min_area*3//4, 'max_area':balloon_max_area},
    'lighter_green': {'count': 0, 'min_area':balloon_min_area*3//4, 'max_area':balloon_max_area},
    'tennis_ball': {'count': 0, 'min_area':50, 'max_area':610},
    'pink': {'count': 0, 'min_area':balloon_min_area, 'max_area':balloon_max_area}  }

def colorAnalyzeImage(image, saveAnalyzedImageToFolder=None, useMax=True, etb = False):
    global colorKeyedObjectsDetectionConfigAndData
    # [colorKey, minArea, x, y, h, w, a, r]
    fields = ['colorKey', 'minArea', 'centerX', 'centerY', 'height', 'width', 'area', 'ratio']
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #hsv[:, :, 0] = hsv[:, :, 0] * 1.05
    #hsv[:, :, 1] = hsv[:, :, 1] * .95
    #hsv[:, :, 2] = hsv[:, :, 2] * .95
    #image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) 

    value = []
    for colorKey in colorKeyedObjectsDetectionConfigAndData.keys():
        if colorKey == 'tennis_ball' and etb:
            pass
        else:
            cad = colorKeyedObjectsDetectionConfigAndData[colorKey]
            minArea = cad['min_area']
            maxArea = cad['max_area']
            coe = ColoredObjectExtractor(colorKey, min_area=minArea, max_area=maxArea)
            #coe.extract(hsv, image, True)
            objs = coe.extract(hsv, numIterations=1)
            n = cad['count']
            nThisFrame = len(objs)
            if useMax:
                if n < nThisFrame:
                    cad['count'] = nThisFrame
            else:
                cad['count'] += nThisFrame        
            for i in range(nThisFrame):
                (cc, e, a, r) = objs[i]
                ((x,y),(h,w),_) = e
                value.append({'colorKey':colorKey,
                            'minArea':minArea,
                            'minArea':maxArea,
                            'centerX':x,
                            'centerY':y,
                            'height':h,
                            'width':w,
                            'area':a,
                            'ratio':r}) #,
                            #'contourColor':cc})
                if True:
                    cv2.ellipse(image, e, cc, 5)
                    s = "{:.1f}, {:.3f}".format(a,r)
                    cv2.putText(image, s, (int(x) - 50, int(y)+40),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, cc, 2, cv2.LINE_AA)

    if saveAnalyzedImageToFolder is not None:
        of_path = os.path.join(saveAnalyzedImageToFolder, f'aa_{time.time()}.jpg')
        cv2.imwrite(of_path, image)
        csv_path = os.path.join(saveAnalyzedImageToFolder, f'aa_{time.time()}.csv')
        with open(csv_path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames = fields)
            writer.writeheader()
            writer.writerows(value)

    return image



def stitchFisheyeImage(dst_image, src_image, degrees_clock_wise, angel_span, inner_radius):
    srcSizeX = src_image.shape[1]
    srcSizeY = src_image.shape[0]
    dstSizeX = dst_image.shape[1]
    dstSizeY = dst_image.shape[0]
    #print("[Fisheye Stitcht] Source canvas size:", srcSizeY, srcSizeX)
    #print("[Fisheye Stitcht] Destination canvas size:", dstSizeY, dstSizeX)

    lc = np.zeros_like(dst_image, dtype=np.uint8)
    xPos = (dstSizeX//2 - srcSizeX//2)
    yPos = (dstSizeY//2 - srcSizeY - inner_radius)
    if xPos < 0:
        xStart = srcSizeX + xPos
        xPos = 0
    else:
        xStart = 0
    if yPos < 0:
        yStart = srcSizeY + yPos
        yPos = 0
    else:
        yStart = 0
    #print("xPos=", xPos,"yPos=", yPos, "xStart=", xStart,"yStart=", yStart)
    lc[yPos+yStart:yPos+srcSizeY, xPos+xStart:xPos+srcSizeX] = src_image[yStart:, xStart:]
    center = (dstSizeX//2, dstSizeY//2 + inner_radius)  # this is in the form of (x, y), rather than (y, x)
    mask = np.zeros_like(lc, dtype = np.uint8)
    #cv2.ellipse(mask, center, (dstSizeX//2, dstSizeY//2), 0, 258, 282, (255, 255, 255), thickness=cv2.FILLED)
    cv2.ellipse(mask, center, (dstSizeX//2, dstSizeY//2), 0, 270-angel_span/2, 270+angel_span/2, (255, 255, 255), thickness=cv2.FILLED)
    lc = cv2.bitwise_and(lc, mask)
    rotation_matrix = cv2.getRotationMatrix2D(center, degrees_clock_wise, 1)
    #rotation_matrix[0, 2] -= 0 //(1-0.75) * center[0] # x-axis
    #rotation_matrix[1, 2] -= 0 //center[1]
    lc = cv2.warpAffine(lc, rotation_matrix, (dstSizeX, dstSizeY))
    dst_image += lc

class AAImageFrameProcesser():
    FT_INDEPENDENT_IMAGE_ACQUIRER = 0
    FT_OVERLAPPING_IMAGE_ACQUIRER = 1
    FT_FISHEYE_IMAGE_ACQUIRER = 2
    FT_HORIZONTAL_SCAN_IMAGE_ACQUIRER = 3
    FT_OTHER_PANORAMA_IMAGE_ACQUIRER = 4

    def __init__(self, image_buffer:Queue, flight_info):
        self.totalNumberFramesProcessed = 0
        self.processedImages = []
        self.processedResults = []
        self.t = None
        self.imageBuffer = image_buffer
        self.flightInfo = flight_info

    def getColorsAndCounts(self, configAndData = colorKeyedObjectsDetectionConfigAndData):
        colors = [] #list(colorKeyedObjectsDetectionConfigAndData.keys())
        counts = []
        for colorKey in configAndData.keys():
            cad = configAndData[colorKey]
            colors.append(colorKey)
            counts.append(cad['count'])
        return colors, counts

    def start(self):
        t = threading.Thread(target = self.process, args=())
        self.t = t
        t.start()

    def end(self):
        if self.t is not None:
            self.t.join()

    def process(self):    
        print("[AAImageFrameProcesser:] Starts processing frames ...")
        image_buffer = self.imageBuffer
        flight_info = self.flightInfo
        startTime = int(time.time())
        startFolder = os.path.join('../output', flight_info['name'])
        if not os.path.exists(startFolder):
            os.makedirs(startFolder)
        startFolder = os.path.join(startFolder, f'{startTime}')
        if not os.path.exists(startFolder):
            os.makedirs(startFolder)
        if flight_info['requiresDrone']:
            unprocessedOutputFolder = os.path.join(startFolder,  'unprocessed')
            if not os.path.exists(unprocessedOutputFolder):
                os.makedirs(unprocessedOutputFolder)
        else:
            unprocessedOutputFolder = None

        processedOutputFolder = os.path.join(startFolder, 'processed')
        if not os.path.exists(processedOutputFolder):
            os.makedirs(processedOutputFolder)    

        nRead = 0
        readFrames=[]
        readFrames2 = []
        while True:
            item = image_buffer.get()
            if item is None:
                if flight_info['type'] == AAImageFrameProcesser.FT_FISHEYE_IMAGE_ACQUIRER:
                    angles = 0
                    deltaAngle = 360/nRead
                    lc = np.zeros((lcRadius*2, lcRadius*2, 3), dtype=np.uint8)
                    for frame in readFrames:
                        stitchFisheyeImage(lc, frame, angles, deltaAngle, ySize//4)
                        angles += deltaAngle
                    img = colorAnalyzeImage(lc, saveAnalyzedImageToFolder=processedOutputFolder)
                    self.totalNumberFramesProcessed += 1
                    self.processedImages.append(img)             
                    break
                elif flight_info['type'] == AAImageFrameProcesser.FT_HORIZONTAL_SCAN_IMAGE_ACQUIRER:
                    stitcher = cv2.Stitcher_create(mode = cv2.Stitcher_PANORAMA) #cv2.Stitch
                    #stitcher.setPanoConfidenceThresh(0.0)
                    print(len(readFrames))
                    print(len(readFrames2))
                    (status, stitched) = stitcher.stitch(readFrames[0:4])
                    if status == 0:
                        img1 = colorAnalyzeImage(stitched, saveAnalyzedImageToFolder=processedOutputFolder)
                        self.totalNumberFramesProcessed += 1
                        self.processedImages.append(img1)
                        colors, counts = self.getColorsAndCounts(colorKeyedObjectsDetectionConfigAndData)
                        self.processedResults.append((colors, counts))
                    else:
                        print("Error stitching Front View!")
                    (status, stitched) = stitcher.stitch(readFrames2[0:4])
                    if status == 0:
                        img2 = colorAnalyzeImage(stitched, saveAnalyzedImageToFolder=processedOutputFolder)
                        totalNumberFramesProcessed += 1
                        self.processedImages.append(img2)
                        colors, counts = self.getColorsAndCounts(colorKeyedObjectsDetectionConfigAndData)
                        self.processedResults.append((colors, counts))
                    else:
                        print("Error stitching Back View!")
                    break                  
                elif flight_info['type'] == AAImageFrameProcesser.FT_OTHER_PANORAMA_IMAGE_ACQUIRER:
                    stitcher = cv2.Stitcher_create(mode = cv2.Stitcher_PANORAMA)
                    #stitcher.setPanoConfidenceThresh(0.0)
                    ss = []
                    print(len(readFrames))
                    for i in range(0, 28, 2):
                        srcs = []
                        for j in range(5):
                            frame = readFrames[i+j]
                            srcs.append(frame)
                        print(len(srcs))
                        (status, stitched) = stitcher.stitch(srcs)
                        if status == 0:
                            stitched = cv2.resize(stitched, dsize=(960, 720), interpolation=cv2.INTER_CUBIC)
                            ss.append(stitched)                    
                    ss2 = []
                    for i in range (0, len(ss)-4, 2):
                        srcs = []
                        for j in range(4):
                            frame = ss[i+j]
                            srcs.append(frame)
                        (status, stitched) = stitcher.stitch(srcs)
                        if status == 0:
                            stitched = cv2.resize(stitched, dsize=(960*2, 720), interpolation=cv2.INTER_CUBIC)
                            ss2.append(stitched)
                    if len(ss2) > 0:        
                        (status, stitched) = stitcher.stitch(ss2)
                        if status == 0:
                            stitched = cv2.resize(stitched, dsize=(960*4, 720), interpolation=cv2.INTER_CUBIC)
                            if status == 0:
                                img = colorAnalyzeImage(stitched, saveAnalyzedImageToFolder=processedOutputFolder)
                                self.totalNumberFramesProcessed += 1
                                self.processedImages.append(img)                                 
                else:
                    break
            
            if unprocessedOutputFolder is not None:
                of_path = os.path.join(unprocessedOutputFolder, f'ba_{nRead}.png')
                cv2.imwrite(of_path, item)
            if flight_info['type'] == AAImageFrameProcesser.FT_HORIZONTAL_SCAN_IMAGE_ACQUIRER:
                if nRead < 12:
                    readFrames.append(item)
                elif nRead > 11:
                    readFrames2.append(item)
                nRead += 1
                print("Processed", nRead, " frame(s) so far...")    
            elif flight_info['type'] == AAImageFrameProcesser.FT_OTHER_PANORAMA_IMAGE_ACQUIRER:
                if nRead == 0:
                    xSize = item.shape[1]
                    ySize = item.shape[0]
                    lcRadius = int(math.sqrt(xSize*xSize + ySize*ySize))
                readFrames.append(item)
                nRead += 1
                print("Processed", nRead, " frame(s) so far...")
            elif self.totalNumberFramesProcessed < 100: #12:
                nRead += 1
                topCropY = item.shape[0]//4
                bottomCropFromY = item.shape[0] - item.shape[0]//6
                bottomCropToY = item.shape[0]
                item[:topCropY, :, :] = 0
                item[bottomCropFromY:bottomCropToY, :, :] = 0
                if self.totalNumberFramesProcessed < 4:
                    excludeTB = True
                else:
                    excludeTB = False 
                img = colorAnalyzeImage(item, saveAnalyzedImageToFolder=processedOutputFolder, useMax=False, etb = excludeTB)
                #rgb_after = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.totalNumberFramesProcessed += 1
                self.processedImages.append(img)
                colors, counts = self.getColorsAndCounts(colorKeyedObjectsDetectionConfigAndData)
                self.processedResults.append((colors, counts))
                print("Processed ", self.totalNumberFramesProcessed, " frame(s) so far...")
        # all done
        print('Finished running image frame processer.')