import glob
import numpy as np
import os
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import datetime
from utils import *

epochs = 1
genShape= 100
disShape = [32,64,64,3]
encoder = False #change if you want to use an encoder
use_gpu = True #change if you want to use gpu


def make_generator_model(noiseShape):
    inputs = layers.Input(shape=noiseShape)

    #background
    z = tf.reshape(inputs,[-1,1,1,noiseShape])
    
    backg = layers.Conv2DTranspose(512, (4, 4), strides=(1,1))(z)
    backg= layers.BatchNormalization()(backg)
    backg1 = layers.LeakyReLU()(backg)
    
    backg1 = layers.Conv2DTranspose(256, (2, 2), strides=(2,2),padding='VALID')(backg1)
    backg1= layers.BatchNormalization()(backg1)
    backg2 = layers.LeakyReLU()(backg1)
    
    backg2 = layers.Conv2DTranspose(128, (2, 2), strides=(2,2),padding='VALID')(backg2)
    backg2= layers.BatchNormalization()(backg2)
    backg3 = layers.LeakyReLU()(backg2)
    
    backg3 = layers.Conv2DTranspose(64, (2, 2), strides=(2,2),padding='VALID')(backg3)
    backg3= layers.BatchNormalization()(backg3)
    backg4 = layers.LeakyReLU()(backg3)
    
    backg4 = layers.Conv2DTranspose(3, (2, 2), strides=(2,2),padding='VALID')(backg4)
    
    # this is the final background output
    background = tf.nn.tanh(backg4)
    background = tf.expand_dims(background,1)
    
    #foreground stuff
    z = tf.reshape(inputs,[-1,1,1,1,noiseShape])
    
    foreg = layers.Conv3DTranspose(512, (2,4,4), strides=(1,1,1),use_bias=False)(z)
    foreg = layers.BatchNormalization()(foreg)
    foreg1 = layers.ReLU()(foreg)
        
    foreg1 = layers.Conv3DTranspose(256, (4,4,4), strides=(2,2,2),padding="SAME",use_bias=False)(foreg1)
    foreg1 = layers.BatchNormalization()(foreg1)
    foreg2 = layers.ReLU()(foreg1)

    foreg2 = layers.Conv3DTranspose(128, (4,4,4), strides=(2,2,2),padding="SAME",use_bias=False)(foreg2)
    foreg2 = layers.BatchNormalization()(foreg2)
    foreg3 = layers.ReLU()(foreg2)
    
    foreg3 = layers.Conv3DTranspose(64, (4,4,4), strides=(2,2,2),padding="SAME",use_bias=False)(foreg3)
    foreg3 = layers.BatchNormalization()(foreg3)
    foreg4 = layers.ReLU()(foreg3)
    
    foreg5 = layers.Conv3DTranspose(3, (4,4,4), strides=(2,2,2),padding="SAME",use_bias=False)(foreg4)
    foreground = tf.nn.tanh(foreg5)
    
    #Mask 
    mask = layers.Conv3DTranspose(1, (4,4,4), strides=(2,2,2),padding="SAME",use_bias=False)(foreg4)
    mask = tf.nn.sigmoid(mask)
    # I'm pretty sure this does nothing
    # _ = tf.tile(background,[-1,32,1,1,1])
    # _ = tf.tile(mask,[-1,1,1,1,3])

    # print((foreground.shape), (background.shape), (mask.shape))
    
    video = tf.add(tf.multiply(mask,foreground),tf.multiply(1-mask,background))
    

    
    model = tf.keras.Model(inputs=inputs, outputs=[video,foreground,background,mask])

    return model
    

def make_discriminator_model(inputShape):
    # model.add(layers.Dropout(0.3))
    inputs = layers.Input(shape=inputShape)
    # model = tf.keras.Sequential()
    
    #first layer
    first = layers.Conv3D(64, (4,4,4), strides=(2,2,2), padding='SAME')(inputs)
    first1 = layers.LeakyReLU()(first)
    
    
    #second layer
    second = layers.Conv3D(128, (4, 4, 4), strides=(2,2,2), padding='SAME')(first1)
    second1 = layers.BatchNormalization()(second)
    second2 =  layers.LeakyReLU()(second1)

    #third layer
    third = layers.Conv3D(256, (4, 4, 4), strides=(2,2,2), padding='SAME')(second2)
    third1 = layers.BatchNormalization()(third)
    third2 = layers.LeakyReLU()(third1)
    
    #fourth layer
    fourth = layers.Conv3D(512, (4, 4, 4), strides=(2,2,2), padding='SAME')(third2)
    fourth1 = layers.BatchNormalization()(fourth)
    fourth2 = layers.LeakyReLU()(fourth1)
    
    fifth = layers.Conv3D(1, (2, 4, 4), strides=(1,1,1), padding='VALID')(fourth2)
    fifth1= layers.Reshape((-1,1))(fifth)
    # print(tf.shape(fifth1))
    
    out =tf.nn.sigmoid(fifth1)
    
    model = tf.keras.Model(inputs=inputs, outputs=[out,fifth1])

    return model

discriminator = make_discriminator_model(disShape)
generator = make_generator_model(genShape)

def get_generator_loss(inputData):
    fake_video,_,_,_ = generator(inputData, training=True)
    prob_fake, logits_fake = discriminator(fake_video,training = False)
    g_cost = tf.reduce_mean(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_fake, labels = tf.ones_like(prob_fake))) 
    return g_cost


def get_discriminator_loss(real_video, fake_video):
    # lambd=  0.0
    prob_real, logits_real = discriminator(real_video,training = True)
    prob_fake, logits_fake = discriminator(fake_video,training = False)

    d_real_cost = tf.reduce_mean(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_real, labels = tf.ones_like(prob_real)))
    d_fake_cost = tf.reduce_mean(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_fake, labels = tf.zeros_like(prob_fake)))
    return d_real_cost + d_fake_cost
    

def save_check_point(generator_optimizer,discriminator_optimizer):
    #saves the model every epoch
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
    checkpoint.save(file_prefix=checkpoint_prefix)




def train(checkpoint_file):
    ''' This is used to store the video and background that needs to be saved'''
    gen_videos = None
    bg = None
    
    device = '/cpu:0'
    if use_gpu:
        device = '/gpu:0'

    with tf.device(device): #assuming we have working gpu's this should take adv
        generator_optimizer = tf.keras.optimizers.Adam( 0.0001, beta_1 = 0.5)
        discriminator_optimizer = tf.keras.optimizers.Adam( 0.0001, beta_1 = 0.5)
        
        if checkpoint_file != '':
            print('Fill in code to restore model')

        data_files = glob.glob("./trainvideos/*")
        for epoch in range(epochs):
            print("Epoch:",epoch)
            for counter in range(len(data_files)):
                noise = tf.random.normal([1, genShape],dtype=tf.dtypes.float32)
                # print(np.array(noise))
                print("....Iteration....:", counter)
                print(datetime.datetime.now()) #added for keeping track of time
                batch_files = data_files[counter*1:(counter+1)*1]
                videos = read_and_process_video(batch_files,1,32)
                print("Processed Video Done!!!")
                
                with tf.GradientTape() as tape:
                    fake_video,_,_,_ = generator(noise,training=True)
                    d_loss = get_discriminator_loss(videos,fake_video)
                grads = tape.gradient(d_loss, discriminator.trainable_variables)
                discriminator_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

                with tf.GradientTape() as tape:
                    g_loss = get_generator_loss(noise)
                grads = tape.gradient(g_loss, generator.trainable_variables)
                generator_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

            
                # print(epoch, 'd loss:', float(d_loss), 'g loss:', float(g_loss))
                gen_videos,_,bg,_ = generator(noise,training=False)
            # print(np.array(gen_videos))
            process_and_write_image(np.array(bg),"bg_epoch" + str(epoch) + "_count" + str(counter))
            process_and_write_video(np.array(gen_videos),"video_epoch" + str(epoch) + "_count" + str(counter))
        
            save_check_point(generator_optimizer,discriminator_optimizer)
               



clear_genvideos('genvideos') # comment out if you want to keep you gen videos
train('')
