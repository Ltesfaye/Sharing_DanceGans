import tensorflow as tf
import cv2
import skvideo.io
import skimage.transform
import numpy as np
import datetime
from cv2 import VideoWriter, VideoWriter_fourcc
import os
import glob
import shutil

"""
Author: CS6670 Group
Code structure inspired from carpedm20/DCGAN-tensorflow, GV1028/videogan
"""

def clear_genvideos(path):
    for i in glob.glob(path):
        os.remove(i)
  
'''
def save_gen(generated_images, n_ex = 36, epoch = 0, iter = 0):
    for i in range(generated_images.shape[0]):
        cv2.imwrite('/root/code/Video_Generation/gen_images/image_' + str(epoch) + '_' + str(iter) + '_' + str(i) + '.jpg', generated_images[i, :, :, :])
'''
def process_and_write_image(images,name):
    images = np.array(images)
    # print(images.shape)
    images = (images + 1)*127.5
    img = images[0,0,:,:]
    # print(img.shape)
    cv2.imwrite("./genvideos/" + name + ".jpg", img)

def read_and_process_video(files,size,nof):
    print("Processesing Videos")
    print((datetime.datetime.now())) #added for keeping track of time
    #TODO: Pass in the 64,64,3 thing
    videos = np.zeros((size,nof,64,64,3))
    counter = 0
    for file in files:
        print(file)
        vid = skvideo.io.vreader(file)
        curr_frames = []
        i = 0
        for frame in vid:
            frame = skimage.transform.resize(frame,[64,64])
            curr_frames.append(frame)
            i = i + 1
            if i >= nof:
              break
        curr_frames = np.array(curr_frames)
        curr_frames = curr_frames*255.0
        curr_frames = curr_frames/127.5 - 1
        print("Shape of frames: {0}".format(curr_frames.shape))
        #TODO: This should rely on the passed in (32,64,64,3) thing imo
        videos[counter,:,:,:,:] = curr_frames[0:nof]
        counter = counter + 1
    
    return videos.astype('float32')

def process_and_write_video(videos,name):
    videos =np.array(videos)
    width = 64
    height = 64
    FPS = 24
    fourcc = VideoWriter_fourcc(*'MP42')
    video = VideoWriter('./genvideos/'+name+'.avi', fourcc, float(FPS), (width, height))

    videos = np.reshape(videos,[-1,32,64,64,3])
    for i in range(videos.shape[0]):
        vid = videos[i,:,:,:,:]
        vid = (vid + 1)*127.5
        for j in range(vid.shape[0]):
            frame = vid[j,:,:,:]
            video.write(frame)
        video.release()

