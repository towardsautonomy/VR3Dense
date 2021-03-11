from moviepy.editor import *

video_file = 'tmp/vr3dense_demo/demo.mov'
output_gif_file = 'tmp/vr3dense_demo/demo.gif'

# timestamps
start_ts = (0, 0.0) # minute, seconds
stop_ts = (0, 37.0) # minute, seconds
resize_factor = 1.0
speed_factor = 1.0
fps_downsample_factor = 3.0

# get clip
clip = (VideoFileClip(video_file)
        .subclip(start_ts,stop_ts)
        .resize(resize_factor))

# write to gif
clip.speedx(speed_factor).write_gif(output_gif_file, fps=int(clip.fps / fps_downsample_factor))