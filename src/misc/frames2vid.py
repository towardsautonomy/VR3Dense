import cv2
import numpy as np
import glob
import os

demo_path = 'tmp/vr3dense_demo_scene104'
demo_frames_path = os.path.join(demo_path, '*.png')
out_vid = os.path.join(demo_path, 'demo.mov')
img_array = []
n_frames = 1000

print('reading frames...')
for i, filename in enumerate(sorted(glob.glob(demo_frames_path))):
    if i == n_frames:
        break
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

print('writing video...')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 10
out = cv2.VideoWriter(out_vid, fourcc, fps, size)
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
print('done!')