import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Detector:

    def __init__(self, x, y):
        self.detX = x
        self.detY = y
        self.avgColour = []
        self.detections = []

    def addAVGColourSum(self, value):
        self.avgColour.append(value)


drawing = False
mouseX, mouseY = -1, -1
size = 10

number_of_lanes = 3
detectors_per_lane = 2

lanes = []
detectors = []

def detectorsDiscretizationFilter(detectors, frameCounter):
    for i in range(0, len(detectors)):
        for j in range(0, frameCounter - 1):
            frames_unite = 10
            frames_count = 0
            for k in range(1, frames_unite):
                if (detectors[i].detections[j] == 1 and ((j + k) < (frameCounter - k - 1)) and (
                        detectors[i].detections[j + k] == 1)):
                    frames_count = frames_count + 1
                for l in range(j, j + frames_count):
                    detectors[i].detections[l] = 1

    for i in range(0, len(detectors)):
        for j in range(0, frameCounter - 2):
            neighboursSum = detectors[i].detections[j - 1] + detectors[i].detections[j + 1]
            if detectors[i].detections[j] == 1 and neighboursSum == 0:
                detectors[i].detections[j] = 0


# In[70]:


def detectorsDiscretization(detectors, frameCounter):
    for detector in detectors:
        detector.detections = [0] * frameCounter

    for i in range(0, len(detectors)):
        for j in range(0, frameCounter - 1):
            test = abs((detectors[i].avgColour[j] - detectors[i].avgColour[j + 1]) / detectors[i].avgColour[j]) * 100
            if test > 1.5:
                detectors[i].detections[j + 1] = 1


# In[71]:


def getAVGcolourSum(gray, detectors):
    cv.namedWindow('detector', cv.WINDOW_NORMAL)
    for detector in detectors:
        detectorZone = gray[int((detector.detY - (size / 2))):int((detector.detY + (size / 2))),
                       int((detector.detX - (size / 2))):int((detector.detX + (size / 2)))]
        avg_color_per_row = np.average(detectorZone, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        detector.addAVGColourSum(avg_color)
        print(avg_color)
        cv.imshow('detector', detectorZone)


def draw_detector(x, y, lane_number):
    colours = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 0, 127], [0, 127, 0], [127, 0, 0]]
    cv.rectangle(frame, (int(x - (size / 2)), int(y - (size / 2))),
                 (int(x + (size / 2)), int(y + (size / 2))), colours[lane_number], 2)

def set_detector(event, x, y, flags, param, count_of_lanes=0):
    global mouseX, mouseY, drawing

    if event == cv.EVENT_LBUTTONDOWN:
        if len(lanes) != 0 and len(lanes) * len(lanes[0]) == number_of_lanes * detectors_per_lane:
            print("Max number of detectors reached")
        else:
            mouseX, mouseY = x, y
            draw_detector(mouseX, mouseY, len(lanes))
            cv.imshow('Frame', frame)
            print(str(mouseX) + " " + str(mouseY))
            if len(detectors) < detectors_per_lane:
                detectors.append(Detector(mouseX, mouseY))
            if len(detectors) == detectors_per_lane:
                lanes.append(detectors.copy())
                detectors.clear()


from datetime import datetime

t1 = datetime.now()
frameCounter = 0
dataInput = input()
cap = cv.VideoCapture(dataInput)

if not cap.isOpened():
    print("Error opening file")

cv.namedWindow('Frame', cv.WINDOW_NORMAL)
cv.setMouseCallback('Frame', set_detector)

cv.resizeWindow('Frame', 1920, 1080)
ret, frame = cap.read()

cv.imshow('Frame', frame)
cv.waitKey(0)
cv.namedWindow('Frame', cv.WINDOW_NORMAL)
cv.resizeWindow('Frame', 800, 600)

while (cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Переводим в градации серого
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Если считался очередной кадр
    if ret == True:
        # Выводим номер кадра в верхнем левом углу
        cv.putText(frame, "Frame " + str(frameCounter), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                   cv.LINE_AA)
        # Отрисовка детекторов
        colour_counter = 0
        for lane in lanes:
            for detector in lane:
                draw_detector(detector.detX, detector.detY, colour_counter)
            colour_counter += 1
            getAVGcolourSum(gray, lane)
        cv.imshow('Frame', frame)
        frameCounter += 1
        if cv.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv.destroyAllWindows()
t2 = datetime.now()


# Заполняем csv файл координатами детекторов
data = dict()
lane_counter = 1
for lane in lanes:
    detectorNumber = 1
    for detector in lane:
        new_dict = {"lane" + str(lane_counter) + " det" + str(detectorNumber) + " X": [detector.detX],
                    "lane" + str(lane_counter) + " det" + str(detectorNumber) + " Y": [detector.detY]}
        data.update(new_dict)
        detectorNumber += 1
    lane_counter += 1
df = pd.DataFrame(data)
df.to_csv(r'Coordinates.csv', sep=';', index=False)
# Заполняем csv файл средними цветами с детекторов
data = dict()
lane_counter = 1
for lane in lanes:
    detectorNumber = 1
    for detector in lane:
        new_dict = {"lane" + str(lane_counter) + " det" + str(detectorNumber): detector.avgColour}
        data.update(new_dict)
        detectorNumber += 1
    lane_counter += 1
df = pd.DataFrame(data)
df.to_csv(r'AvgColours.csv', sep=';', index=False)

for lane in lanes:
    detectorsDiscretization(lane, frameCounter)

# Заполняем csv файл дискретными значениями с детекторов
data = dict()
lane_counter = 1
for lane in lanes:
    detectorNumber = 1
    for detector in lane:
        new_dict = {"lane" + str(lane_counter) + " det" + str(detectorNumber): detector.detections}
        data.update(new_dict)
        detectorNumber += 1
    lane_counter += 1
df = pd.DataFrame(data)
df.to_csv(r'RawDetections.csv', sep=';', index=False)

for lane in lanes:
    detectorsDiscretizationFilter(lane, frameCounter)

# Заполняем csv файл дискретными отфильтрованными значениями с детекторов
data = dict()
lane_counter = 1
for lane in lanes:
    detectorNumber = 1
    for detector in lane:
        new_dict = {"lane" + str(lane_counter) + " det" + str(detectorNumber): detector.detections}
        data.update(new_dict)
        detectorNumber += 1
    lane_counter += 1
df = pd.DataFrame(data)
df.to_csv(r'FilteredDetections.csv', sep=';', index=False)

x = []
for i in range(1, frameCounter + 1):
    x.append(i)

position = 1

fig, axs = plt.subplots(number_of_lanes, detectors_per_lane)
i = 0
for lane in lanes:
    j = 0
    for detector in lane:
        axs[i, j].plot(x, detector.avgColour)
        axs[i, j].set_title("Lane №" + str(i) + "detector №" + str(j))
        j += 1
    i += 1
for ax in axs.flat:
    ax.set(xlabel="Frames", ylabel="Average colour")
plt.show()

fig, axs = plt.subplots(number_of_lanes, detectors_per_lane)
i = 0
for lane in lanes:
    j = 0
    for detector in lane:
        axs[i, j].plot(x, detector.detections)
        axs[i, j].set_title("Lane №" + str(i) + "detector №" + str(j))
        j += 1
    i += 1
for ax in axs.flat:
    ax.set(xlabel="Frames", ylabel="Detections")
plt.show()

# In[79]:


position = 1
for detector in detectors:
    plt.subplot(2, 1, position)
    plt.plot(x, detector.avgColour)
    position = position + 1
plt.show()

position = 1
for detector in detectors:
    plt.subplot(2, 1, position)
    plt.plot(x, detector.detections)
    position = position + 1
plt.show()
a = open('detection_detectors.txt', 'w')
a.close()
f = open('detection_detectors.txt', 'a')
f.write("F 1 2 3 4 5 6 c" + '\n')
allCount = 0
for i in range(0, frameCounter):
    transportCount = int(df['lane1 det1'][i]) + int(df['lane1 det2'][i]) + int(df['lane2 det1'][i]) + int(
        df['lane2 det2'][i]) + int(df['lane3 det1'][i]) + int(df['lane3 det2'][i])
    f.write('\n' + str(i) + " " + str(df['lane1 det1'][i]) + " " + str(data['lane1 det2'][i]) + " " +
            str(data['lane2 det1'][i]) + " " + str(data['lane2 det2'][i]) + " " +
            str(data['lane3 det1'][i]) + " " + str(data['lane3 det2'][i]) + " " + str(transportCount))
    allCount = allCount + transportCount
f.close()

allCount
