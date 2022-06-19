#!/usr/bin/env python
# coding: utf-8

# In[56]:


import pandas as pd
import matplotlib.pyplot as plt

frame_rate = 33
lane_density_per_sec = []
lanes_density_per_sec = []


def density_to_sec(density_lanes):
    for lane in density_lanes:
        frame_counter = 0
        second = 1
        while frame_counter <= len(lane):
            frame_sum = 0
            for i in range(frame_rate * (second - 1), frame_rate * second):
                if i < len(lane):
                    frame_sum += lane[i]
                frame_counter += 1
            lane_density_per_sec.append(frame_sum / frame_rate)
            second += 1
        lanes_density_per_sec.append(lane_density_per_sec.copy())
        lane_density_per_sec.clear()


class Detector:

    def __init__(self):
        self.detX = 0
        self.detY = 0
        self.avgColour = []
        self.detections = []


number_of_lanes = 3  # количество полос
detectors_per_lane = 2  # количество детекторов на полосу

# список полос
lanes = []
# список детекторов на одну полосу
detectors = []

for i in range(number_of_lanes):
    for j in range(detectors_per_lane):
        detectors.append(Detector())
    lanes.append(detectors.copy())
    detectors.clear()

df = pd.read_csv('FilteredDetections.csv')

for i in range(len(df)):
    single_line_values = df.values[i][0].split(';')
    m = 0
    for j in range(len(lanes)):
        for k in range(len(lanes[j])):
            lanes[j][k].detections.append(single_line_values[m])
            m += 1

print(single_line_values[5][0])
# плотность на полосу
density_per_lane = []
# плотность по полосам
density_lanes = []

detections_counter = 0
for i in range(len(lanes)):
    for k in range(len(lanes[i][0].detections)):
        for j in range(len(lanes[i])):
            if lanes[i][j].detections[k] == '1':
                detections_counter += 1.0
        density_per_lane.append(detections_counter / len(lanes[i]))
        detections_counter = 0
    density_lanes.append(density_per_lane.copy())
    density_per_lane.clear()

x = []
for i in range(1, len(lanes[0][0].detections) + 1):
    x.append(i)

density_to_sec(density_lanes)

fig, axs = plt.subplots(number_of_lanes, figsize=(15, 15))
i = 0
for lane in density_lanes:
    j = 0
    axs[i].step(x, lane)
    axs[i].set_title("Lane №" + str(i))
    i += 1
for ax in axs.flat:
    ax.set(xlabel="Frames", ylabel="Detections")
plt.show()
density_to_sec(density_lanes)
lanes_density_per_sec
