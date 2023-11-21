import sys
import os
import glob
import threading
import time
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
from queue import Queue
import csv

from aaImageFrameAquirer import AAImageFrameAcquirer
from aaImageFrameProcesser import AAImageFrameProcesser

def missionTaskBeginner(flight_number=0, show_charts=0) -> int:
    global processedImages, totalNumberFramesProcessed, processedResults
    ib = Queue()

    a = AAImageFrameAcquirer(ib, flight_number)
    p = AAImageFrameProcesser(ib, a.flightInfo)

    s = time.time()
    print("[MISSION INFO] -- start time:", s)
    p.start()
    a.start()                
    
    a.end()
    ib.put(None)
    p.end()

    totalNumberFramesProcessed = p.totalNumberFramesProcessed
    processedImages = p.processedImages
    processedResults = p.processedResults

    print("Main thread: totalNumberFramesProcessed -- ", totalNumberFramesProcessed)
    e =  time.time()
    print("[MISSION INFO] -- end time:", e)
    print("[MISSION INFO] -- total flight + processing time:", e-s, "seconds.")
    print("[MISSION INFO] -- Frame by frame PROGRESSIVE results recap:")
    frn = 0
    for pr in processedResults:
        print(frn, pr)
        frn += 1
    if show_charts < 1:
        return 0

    if totalNumberFramesProcessed > 0:
        fig = plt.figure(figsize=(12, 8))
        inner = gridspec.GridSpec(2, 1, hspace=0.2, height_ratios=[1,5])
        ax0 = plt.Subplot(fig, inner[0])
        ax1 = plt.Subplot(fig, inner[1])
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax0.set_title("Objects detected per color")
        #ax0.bar(colors, vs, color ='maroon', width = 0.3)
        fig.add_subplot(ax0)
        fig.add_subplot(ax1)

        nrows = int((totalNumberFramesProcessed+1)/2)
        j = 0
        ris = []
        for i in range(nrows):
            img0 = cv2.cvtColor(processedImages[j], cv2.COLOR_BGR2RGB)
            j+=1
            if j<totalNumberFramesProcessed:
                img1 = cv2.cvtColor(processedImages[j], cv2.COLOR_BGR2RGB)
                ri = cv2.hconcat([img0, img1])
                ris.append(ri)
                j+=1
            else:
                img1 = np.zeros_like(img0, dtype=np.uint8)
                img1[:,:,:] = 255
                ri = cv2.hconcat([img0, img1])
                ris.append(ri)

        concatedImage = cv2.vconcat(ris)        
        class Index:
            def __init__(self):
                self.ind = 0
                self.prev(None)

            def next(self, event):
                if self.ind < (totalNumberFramesProcessed-1):
                    self.ind += 1
                    img = cv2.cvtColor(processedImages[self.ind], cv2.COLOR_BGR2RGB)
                    ax1.imshow(img)
                    ax1.set_title(f'Frame #{self.ind}')
                    (colors, vs) = processedResults[self.ind]
                    ax0.clear()
                    ax0.bar(colors, vs, color ='maroon', width = 0.3)
                    ax0.set_yticks(np.arange(min(vs), max(vs)+1, 1))
                    plt.draw()

            def prev(self, event):
                if self.ind > 0:
                    self.ind -= 1
                    img = cv2.cvtColor(processedImages[self.ind], cv2.COLOR_BGR2RGB)
                    ax1.imshow(img)
                    ax1.set_title(f'Frame #{self.ind}')
                    (colors, vs) = processedResults[self.ind]
                    ax0.clear()
                    ax0.bar(colors, vs, color ='maroon', width = 0.3)
                    ax0.set_yticks(np.arange(min(vs), max(vs)+1, 1))
                    plt.draw()
                elif self.ind == 0:
                    self.ind -= 1
                    ax1.imshow(concatedImage)
                    ax1.set_title('All frames')
                    (colors, vs) = p.getColorsAndCounts()
                    ax0.clear()
                    ax0.bar(colors, vs, color ='maroon', width = 0.3)
                    ax0.set_yticks(np.arange(min(vs), max(vs)+1, 1))
                    plt.draw()

        callback = Index()
        axprev = fig.add_axes([0.125, 0.02, 0.1, 0.075])
        axnext = fig.add_axes([0.8, 0.02, 0.1, 0.075])
        bnext = Button(axnext, 'Next')
        bnext.on_clicked(callback.next)
        bprev = Button(axprev, 'Previous')
        bprev.on_clicked(callback.prev)
        plt.show()

    return 99

def main(argv):
    showCharts = 0
    if len(argv) < 1:
        print("Unkown flight #. Abort.")
    else:
        if len(argv) >= 2:
            showCharts = int(argv[1])
        flightNumber = int(argv[0])
        print("Taking flight #: ", flightNumber)       
        missionTaskBeginner(flight_number = flightNumber, show_charts = showCharts)

if __name__ == "__main__":
   main(sys.argv[1:])