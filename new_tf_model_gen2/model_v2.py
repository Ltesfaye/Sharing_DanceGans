import glob
import numpy as np
import os
from tensorflow.keras import layers,activations
import tensorflow as tf
import numpy as np
import random
from utils import *
from plot_losses import * 
import datetime

epochs = 1700
number_of_frames=32 #note that this doesn't update your generator

genShape= 34
disShape = [number_of_frames,34,1]
encoder = False #change if you want to use an encoder
tile_save = False #change if you want to output a video with 128 fames REQUIRES ENCODER TO BE TRUE
use_gpu = True #change if you want to use gpu
Display_plot_when_done = True # change to true if you want losses to be graphed

# 
# error_file = open( file_name, 'w+')


def make_generator_model(inputShape):
    inputs = layers.Input(shape=inputShape)

    first = layers.Dense(8*8*64)(inputs)
    first1 = layers.LeakyReLU()(first)

    second  = layers.Dense(16*17*16*2)(first1)
    second1 = layers.LeakyReLU()(second)

    third = layers.Dense(32*34*1) (second1)
    out = activations.tanh(third)

    out = layers.Reshape((32,34,1))(out)

    return tf.keras.Model(inputs=inputs, outputs=[out])

def make_discriminator_model(inputShape):
  
    inputs = layers.Input(shape=inputShape)

    
    #first layer
    first = layers.Conv2D(64, (5,5), strides=(2,2), padding='SAME')(inputs)
    first1 = layers.LeakyReLU()(first)
    first1 = layers.Dropout(0.14)(first1)
    
    
    #second layer
    second = layers.Conv2D(128, (5,5), strides=(2,2), padding='SAME')(first1)
    second1 = layers.BatchNormalization()(second)
    second1 =  layers.LeakyReLU()(second1)
    second1 = layers.Dropout(0.14)(second1)


    #third layer
    third = layers.Flatten()(second1)
    
    third = layers.Dense(100)(third) # might not need this 
    out = layers.Dense(1)(third)
    
    model = tf.keras.Model(inputs=inputs, outputs=[out])

    return model

discriminator = make_discriminator_model(disShape)
print(discriminator.summary())
print('_________________________________________________________')
generator = make_generator_model(genShape)
print(generator.summary())



cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# loss functions we've tested with 
#CategoricalHinge() - fails to converge 
#MeanAbsolutePercentageError() - fails to converge 

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
    

def save_check_point(generator_optimizer,discriminator_optimizer):
    #saves the model every epoch
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
    checkpoint.save(file_prefix=checkpoint_prefix)

def get_input_generator_value(first_frame):
    if encoder:
        return first_frame.reshape((1,34))
    else:
        return tf.random.normal([1,genShape])

def train():
    device = '/cpu:0'
    if use_gpu:
        device = '/gpu:0'
    
    #The lists bellow are used to save 
    gen_losses = []
    disc_losses= []


    training_videos  = read_and_load_video_all_files('./ballet',nof=number_of_frames)
    output_pose_video_matrix_as_gif(training_videos[0],'example_output')

    with tf.device(device): #assuming we have working gpu's this should take adv
        generator_optimizer = tf.keras.optimizers.Adam(0.0001) #, beta_1 = 0.5)
        discriminator_optimizer = tf.keras.optimizers.Adam(0.0001) #, beta_1 = 0.5)
        
        for epoch in range(epochs):
            print("Epoch:",epoch)
            for counter in range(len(training_videos)):
                video = training_videos[counter] 
                gen_input = get_input_generator_value(video[0,0,:,0]) 
                

                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    generated_video = generator(gen_input, training=True)

                    real_output = discriminator(video, training=True)
                    fake_output = discriminator(generated_video, training=True)

                    gen_loss = generator_loss(fake_output)
                    disc_loss = discriminator_loss(real_output, fake_output)
                gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
                gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

                generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
                discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
            
            gen_losses.append(float(gen_loss))
            disc_losses.append(float(disc_loss))
            
    
            
            if epoch%100==0 or epochs-1 == epoch:
                video = random.choice(training_videos)
                gen_input = get_input_generator_value(video[0,0,:,0])
                out_video= generator(gen_input,training=False)
                output_pose_video_matrix_as_gif(out_video,"video_epoch" + str(epoch))

                if tile_save:
                    assert encoder == True 
                    composite_video = np.zeros((1,128,34,1)).astype('float32')
                    composite_video[:,0:32,:,:]= np.array(out_video).astype('float32')
                    composite_video[:,32:64,:,:]= np.array(generator(composite_video[0,31,:,0].reshape((1,34)),training=False)).astype('float32')
                    composite_video[:,64:96,:,:]= np.array(generator(composite_video[0,63,:,0].reshape((1,34)),training=False)).astype('float32')
                    composite_video[:,96:128,:,:]= np.array(generator(composite_video[0,95,:,0].reshape((1,34)),training=False)).astype('float32')

                    output_pose_video_matrix_as_gif(composite_video,"Composite_Video_epoch" + str(epoch))

                print('Gen Loss',float(gen_loss), 'Dis Loss',float(disc_loss))
                save_check_point(generator_optimizer,discriminator_optimizer)

                if epochs-1 == epoch:
                   
                    plot_errors(gen_losses,disc_losses,Display_plot_when_done)


            
train()



