"""
Author: CS6670 Group
Code structure inspired from carpedm20/DCGAN-tensorflow, GV1028/videogan

"""

import tensorflow as tf
import numpy as np
import glob
from utils import *
import sys
import datetime

class VideoGAN():
    def __init__(self,sess,video_dim,zdim,batch_size,epochs,checkpoint_file,lambd):
        self.bv1 = batch_norm(name = "genb1")
        self.bv2 = batch_norm(name = "genb2")
        self.bv3 = batch_norm(name = "genb3")
        self.bv4 = batch_norm(name = "genb4")
        self.bs1 = batch_norm(name = "genb5")
        self.bs2 = batch_norm(name = "genb6")
        self.bs3 = batch_norm(name = "genb7")
        self.bs4 = batch_norm(name = "genb8")
        self.bd1 = batch_norm(name = "dis1")
        self.bd2 = batch_norm(name = "dis2")
        self.bd3 = batch_norm(name = "dis3")
        self.video_dim = video_dim
        self.zdim = zdim
        self.batch_size = batch_size
        self.epochs = epochs
        self.checkpoint_file = checkpoint_file
        self.lambd = lambd
        self.sess = sess

    def build_model(self):
        self.z = tf.compat.v1.placeholder(tf.float32, [None,self.zdim])
        self.zsample = tf.compat.v1.placeholder(tf.float32,[None,self.zdim])
        self.real_video = tf.compat.v1.placeholder(tf.float32, [None] +self.video_dim)
        self.fake_video,self.foreground,self.background,self.mask = self.generator(self.z)
        self.genvideo,self.bg= self.visualize_videos()
        prob_real, logits_real = self.discriminator(self.real_video)
        prob_fake, logits_fake = self.discriminator(self.fake_video,reuse = True)
        d_real_cost = tf.reduce_mean(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_real, labels = tf.ones_like(prob_real)))
        d_fake_cost = tf.reduce_mean(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_fake, labels = tf.zeros_like(prob_fake)))
        self.g_cost = tf.reduce_mean(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_fake, labels = tf.ones_like(prob_fake))) + self.lambd*tf.norm(tensor=self.mask,ord=1)
        self.d_cost = d_real_cost + d_fake_cost

    def train(self):
        gen_var = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,scope="generator")
        dis_var = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,scope="discriminator")
        self.g_opt = tf.compat.v1.train.AdamOptimizer(learning_rate = 0.0002, beta1 = 0.5).minimize(self.g_cost, var_list = gen_var)
        self.d_opt = tf.compat.v1.train.AdamOptimizer(learning_rate = 0.0002, beta1 = 0.5).minimize(self.d_cost, var_list = dis_var)
        visualize_count = 1
        saver = tf.compat.v1.train.Saver()
        if self.checkpoint_file == "None":
            self.ckpt_file = None
        if self.checkpoint_file:
            saver_ = tf.compat.v1.train.import_meta_graph('./checkpoints/' + self.checkpoint_file + '.meta')
            saver_.restore(self.sess,tf.train.latest_checkpoint('./checkpoints/'))
            print("Restored model")
        else:
            tf.compat.v1.global_variables_initializer().run()
        data_files = glob.glob("./trainvideos/*")
        #change print_count to determine how many times you print
        print_count = self.batch_size
        for epoch in range(self.epochs):
            print(("Epoch: ", epoch)) 
            for counter in range(len(data_files)//self.batch_size):
                noise_sample = np.random.normal(-1, 1, size = [visualize_count, self.zdim]).astype(np.float32)
                noise = np.random.normal(-1, 1, size = [self.batch_size, self.zdim]).astype(np.float32)
                print(("....Iteration....:", counter))
                print((datetime.datetime.now())) #added for keeping track of time
                batch_files = data_files[counter*self.batch_size:(counter+1)*self.batch_size]
                videos = read_and_process_video(batch_files,self.batch_size,32)
                print("Processed Videos")
                print((datetime.datetime.now())) #added for keeping track of time
                #print videos.shape
                #process_and_write_video(videos,"true_video" + str(counter))
                _, dloss = self.sess.run([self.d_opt, self.d_cost], feed_dict = {self.z : noise, self.real_video: videos})
                _, gloss = self.sess.run([self.g_opt, self.g_cost], feed_dict = {self.z : noise, self.real_video: videos})
    #            _, gloss = self.sess.run([self.g_opt, self.g_cost], feed_dict = {self.z : noise, self.real_video: videos})
                print("Discriminator Loss: ", dloss)
                print("Generator Loss", gloss)
                print(datetime.datetime.now()) #added for keeping track of time
                #this basically writes the video generated when it is the end of the batch, BUT IM PRETTY SURE THIS IS WRONG
                if np.mod(counter + 1, print_count) == 0: #this mod is 100% wrong right? 
                    gen_videos,bg = self.sess.run([self.genvideo,self.bg], feed_dict = {self.zsample : noise_sample}) #generate video given noise
                    process_and_write_video(gen_videos,"video_epoch" + str(epoch) + "_count" + str(counter))
                    process_and_write_image(bg,"bg_epoch" + str(epoch) + "_count" + str(counter))
                    print (".....Writing sample generated videos......")

            #saves video after end of epoch
            saver.save(self.sess,'./checkpoints/VideoGAN_{}_{}_{}.ckpt'.format(self.batch_size,epoch,counter))
            print('Saved {}'.format(counter))


    def generator(self,z,reuse = False):
        with tf.compat.v1.variable_scope("generator") as scope:
            #Background
            z = tf.reshape(z,[-1,1,1,self.zdim])
            deconvb1 = tf.compat.v1.layers.conv2d_transpose(z,512,kernel_size=[4,4],strides =[1,1],name="gen1")
            deconvb1 = tf.nn.relu(self.bs1(deconvb1))
            deconvb2 = tf.compat.v1.layers.conv2d_transpose(deconvb1,256,kernel_size=[2,2],strides =[2,2],padding="VALID",name="gen2")
            deconvb2 = tf.nn.relu(self.bs2(deconvb2))
            deconvb3 = tf.compat.v1.layers.conv2d_transpose(deconvb2,128,kernel_size=[2,2],strides =[2,2],padding="VALID",name="gen3")
            deconvb3 = tf.nn.relu(self.bs3(deconvb3))
            deconvb4 = tf.compat.v1.layers.conv2d_transpose(deconvb3,64,kernel_size=[2,2],strides =[2,2],padding="VALID",name="gen4")
            deconvb4 = tf.nn.relu(self.bs4(deconvb4))
            deconvb5 = tf.compat.v1.layers.conv2d_transpose(deconvb4,3,kernel_size=[2,2],strides =[2,2],padding="VALID",name="gen5")
            background = tf.nn.tanh(deconvb5)
            #Foreground
            #z  = tf.expand_dims(z,1)
            z = tf.reshape(z,[-1,1,1,1,self.zdim])
            deconv1 = tf.compat.v1.layers.conv3d_transpose(z,filters = 512,kernel_size = [2,4,4],strides = [1,1,1], use_bias = False,name="gen6")
            deconv1 = tf.nn.relu(self.bv1(deconv1))
            deconv2 = tf.compat.v1.layers.conv3d_transpose(deconv1,filters= 256,kernel_size=[4,4,4],strides=[2,2,2],padding = "SAME",use_bias = False,name="gen7")
            deconv2 = tf.nn.relu(self.bv2(deconv2))
            deconv3 = tf.compat.v1.layers.conv3d_transpose(deconv2,filters= 128,kernel_size =[4,4,4],strides = [2,2,2], padding ="SAME",use_bias = False,name="gen8")
            deconv3 = tf.nn.relu(self.bv3(deconv3))
            deconv4 = tf.compat.v1.layers.conv3d_transpose(deconv3,filters=64,kernel_size=[4,4,4],strides=[2,2,2],padding ="SAME",use_bias = False,name="gen9")
            deconv4 = tf.nn.relu(self.bv4(deconv4))

            #Mask
            mask = tf.compat.v1.layers.conv3d_transpose(deconv4,filters= 1, kernel_size=[4,4,4], strides =[2,2,2],padding ="SAME",use_bias = False,name="gen10")
            mask = tf.nn.sigmoid(mask)
            #Video
            foreground = tf.compat.v1.layers.conv3d_transpose(deconv4,filters = 3, kernel_size = [4,4,4], strides = [2,2,2], padding ="SAME",use_bias = False,name="gen11")
            foreground = tf.nn.tanh(foreground)
            #Replicate background and mask
            background = tf.expand_dims(background,1)
            backreplicate = tf.tile(background,[-1,32,1,1,1])
            maskreplicate = tf.tile(mask,[-1,1,1,1,3])
            #Incorporate mask
            video = tf.add(tf.multiply(mask,foreground),tf.multiply(1-mask,background))
            print("Video Shape")
            print(video.get_shape())
            return video,foreground,background,mask
    def discriminator(self,vid,reuse = False):
        with tf.compat.v1.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            conv1 = tf.compat.v1.layers.conv3d(vid,64,kernel_size=[4,4,4],strides=[2,2,2],padding="SAME",reuse=reuse,name="dis1")
            conv1 = lrelu(conv1)
            conv2 = tf.compat.v1.layers.conv3d(conv1,128,kernel_size=[4,4,4],strides=[2,2,2],padding="SAME",reuse=reuse,name="dis2")
            conv2 = lrelu(self.bd1(conv2))
            conv3 = tf.compat.v1.layers.conv3d(conv2,256,kernel_size=[4,4,4],strides=[2,2,2],padding="SAME",reuse=reuse,name="dis3")
            conv3 = lrelu(self.bd2(conv3))
            conv4 = tf.compat.v1.layers.conv3d(conv3,512,kernel_size=[4,4,4],strides=[2,2,2],padding="SAME",reuse=reuse,name="dis4")
            conv4 = lrelu(self.bd3(conv4))
            conv5 = tf.compat.v1.layers.conv3d(conv4,1,kernel_size=[2,4,4],strides=[1,1,1],padding="VALID",reuse=reuse,name="dis5")
            conv5 = tf.reshape(conv5, [-1,1])
            conv5sigmoid = tf.nn.sigmoid(conv5)
            return conv5sigmoid,conv5

    def visualize_videos(self):
        with tf.compat.v1.variable_scope("generator") as scope:
            scope.reuse_variables()
            #Background
            z = tf.reshape(self.zsample,[-1,1,1,self.zdim])
            deconvb1 = tf.compat.v1.layers.conv2d_transpose(z,512,kernel_size=[4,4],strides =[1,1],name="gen1")
            deconvb1 = tf.nn.relu(self.bs1(deconvb1))
            deconvb2 = tf.compat.v1.layers.conv2d_transpose(deconvb1,256,kernel_size=[2,2],strides =[2,2],padding="VALID",name="gen2")
            deconvb2 = tf.nn.relu(self.bs2(deconvb2))
            deconvb3 = tf.compat.v1.layers.conv2d_transpose(deconvb2,128,kernel_size=[2,2],strides =[2,2],padding="VALID",name="gen3")
            deconvb3 = tf.nn.relu(self.bs3(deconvb3))
            deconvb4 = tf.compat.v1.layers.conv2d_transpose(deconvb3,64,kernel_size=[2,2],strides =[2,2],padding="VALID",name="gen4")
            deconvb4 = tf.nn.relu(self.bs4(deconvb4))
            deconvb5 = tf.compat.v1.layers.conv2d_transpose(deconvb4,3,kernel_size=[2,2],strides =[2,2],padding="VALID",name="gen5")
            background = tf.nn.tanh(deconvb5)
            #Foreground
            #z  = tf.expand_dims(z,1)
            z = tf.reshape(z,[-1,1,1,1,self.zdim])
            deconv1 = tf.compat.v1.layers.conv3d_transpose(z,filters = 512,kernel_size = [2,4,4],strides = [1,1,1], use_bias = False,name="gen6")
            deconv1 = tf.nn.relu(self.bv1(deconv1))
            deconv2 = tf.compat.v1.layers.conv3d_transpose(deconv1,filters= 256,kernel_size=[4,4,4],strides=[2,2,2],padding = "SAME",use_bias = False,name="gen7")
            deconv2 = tf.nn.relu(self.bv2(deconv2))
            deconv3 = tf.compat.v1.layers.conv3d_transpose(deconv2,filters= 128,kernel_size =[4,4,4],strides = [2,2,2], padding ="SAME",use_bias = False,name="gen8")
            deconv3 = tf.nn.relu(self.bv3(deconv3))
            deconv4 = tf.compat.v1.layers.conv3d_transpose(deconv3,filters=64,kernel_size=[4,4,4],strides=[2,2,2],padding ="SAME",use_bias = False,name="gen9")
            deconv4 = tf.nn.relu(self.bv4(deconv4))

            #Mask
            mask = tf.compat.v1.layers.conv3d_transpose(deconv4,filters= 1, kernel_size=[4,4,4], strides =[2,2,2],padding ="SAME",use_bias = False,name="gen10")
            mask = tf.nn.sigmoid(mask)
            #Video
            foreground = tf.compat.v1.layers.conv3d_transpose(deconv4,filters = 3, kernel_size = [4,4,4], strides = [2,2,2], padding ="SAME",use_bias = False,name="gen11")
            foreground = tf.nn.tanh(foreground)
            #Replicate background and mask
            background = tf.expand_dims(background,1)
            backreplicate = tf.tile(background,[-1,32,1,1,1])
            maskreplicate = tf.tile(mask,[-1,1,1,1,3])
            #Incorporate mask
            video = tf.add(tf.multiply(mask,foreground),tf.multiply(1-mask,background))
            print("Video Shape")
            print(video.get_shape())
            return video,background
