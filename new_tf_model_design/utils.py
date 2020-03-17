import cv2
import numpy as np
from cv2 import VideoWriter, VideoWriter_fourcc
import os
import glob
import pandas as pd
import json
from PIL import Image, ImageDraw
from saveGIF import*

"""
Author: CS6670 Group
Code structure inspired from carpedm20/DCGAN-tensorflow, GV1028/videogan
"""

def clear_prior_generated_content():
    for i in glob.glob('./genvideos/*'):
        os.remove(i)

    for i in glob.glob('./gengifs/*'):
        os.remove(i)

def delete_empty_jsons(path):
    for i in (glob.glob(path)):
        should_delete = False
        with open(i,'r') as file_name:
            data = json.load(file_name)
            should_delete = (data == [])
      
        if should_delete:
            print('Deleting empty json: ' + i )
            os.remove(i)

def group_same_videos(image_folders):
    out = []
    cur = []
    curr_dir = ''
    for img in image_folders:
        if curr_dir is '':
            curr_dir = img.split('_')[0]
            cur.append(img)
        elif curr_dir == img.split('_')[0]:
            cur.append(img)
        else:
            if cur != []:
                out.append(cur)
            cur = [img]
            curr_dir = img.split('_')[0]
    if cur != []:
        out.append(cur)
    return out

def random_start_group_frames(files,nof):
    out = []
    for group in files:
        start = np.random.randint(0,high=len(group)-1-nof)
        out.append(group[start:start+nof])
    return out
    
def read_process_training_points(path, nof=128, delete_empty=False):
    if delete_empty:
        delete_empty_jsons(path)

    valid_jsons = glob.glob(path)
    valid_jsons = group_same_videos(valid_jsons)
    
    filter_short_training_sets = lambda x: len(x) >= nof
    valid_jsons = list(filter(filter_short_training_sets,valid_jsons))
    print('Loading Valid Videos...Total:', len(valid_jsons))
    return random_start_group_frames(valid_jsons, nof)

def read_process_jsons_video(files,nof=128):
    videos = np.zeros((1,nof,34,34,1))

    for f in range(len(files)):
        with open(files[f],'r') as file_name:
            data = json.load(file_name)[0]
        no_str_list=[]
        for i in data:
            if not isinstance(i[1],str):
                no_str_list+= i[1]
        frame = np.diag(np.array(no_str_list)).reshape((34,34,1))
        videos[0,f,:,:,:,] = frame

    return videos.astype('float32')
    
def read_and_load_video_all_files(dir,delete_genVideo=True,delete_empty = False):
    if delete_genVideo:
        clear_prior_generated_content()

    output= read_process_training_points(dir+'/*json',delete_empty=delete_empty)
    return [read_process_jsons_video(i) for i in output]
    
def process_and_write_video(videos,name):
    videos =np.array(videos)
    width = height = 34
    FPS = 24
    fourcc = VideoWriter_fourcc(*'XVID')
    video = VideoWriter('./genvideos/'+name+'.avi', fourcc, float(FPS), (width, height),0)

    # videos = np.reshape(videos,[-1,videos.shape[0],width,height,1])
    for i in range(videos.shape[0]):
        vid = videos[i,:,:,:,:]
        vid = (vid + 1)*127.5
        for j in range(vid.shape[0]):
            frame = vid[j,:,:,:]
            video.write(frame)
        video.release()



def output_pose_video_matrix(matrix,name):
    output_values = np.array(matrix)
    frames = []
    for i in range(output_values.shape[1]):

        relavent_pose = list(output_values[0,i,:,:,0].diagonal())
        frames.append(form_image_frame(relavent_pose))

    list_pillow_images_to_gif(frames,name)
        
