import glob
import numpy as np
import os
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import datetime
from utils import *

epochs = 1001
genShape= 34
disShape = [128,34,34,1]
# encoder = False #change if you want to use an encoder
use_gpu = True #change if you want to use gpu

def make_generator_model(noiseShape):
    inputs = layers.Input(shape=noiseShape)
    first = layers.Dense(34*34*256, use_bias=False)(inputs)
    first = layers.BatchNormalization()(first)
    first1 = layers.LeakyReLU()(first)
    first1 = layers.Reshape((34, 34, 256))(first1)

    second = layers.Conv2DTranspose(64,(5,5), strides=(1,1),padding='SAME',use_bias=False)(first1)
    second = layers.BatchNormalization()(second)
    second1 = layers.LeakyReLU()(second)

    
    third = layers.Conv2DTranspose(128,(5,5), strides=(1,1),padding='SAME',use_bias=False)(second1)
    third = layers.BatchNormalization()(third)
    third1 = layers.LeakyReLU()(third)

    out = layers.Conv2DTranspose(128,(5,5), strides=(1,1),padding='SAME',use_bias=False)(third1)
    out = layers.LeakyReLU()(out)
    out = layers.Reshape((128,34, 34, 1))(out)
    
    return tf.keras.Model(inputs=inputs, outputs=[out])

def make_discriminator_model(inputShape):
    # model.add(layers.Dropout(0.3))
    inputs = layers.Input(shape=inputShape)
    # model = tf.keras.Sequential()
    
    #first layer
    first = layers.Conv3D(64, (4,4,4), strides=(2,1,1), padding='SAME')(inputs)
    first1 = layers.LeakyReLU()(first)
    # print(first1.shape)
    
    
    #second layer
    second = layers.Conv3D(128, (4, 4, 4), strides=(2,2,2), padding='SAME')(first1)
    second1 = layers.BatchNormalization()(second)
    second2 =  layers.LeakyReLU()(second1)

    #third layer
    third = layers.Conv3D(256, (4, 4, 4), strides=(2,2,2), padding='SAME')(second2)
    third1 = layers.BatchNormalization()(third)
    third2 = layers.LeakyReLU()(third1)
    
    #fourth layer
    fourth = layers.Conv3D(512, (4, 4, 4), strides=(2,1,1), padding='SAME')(third2)
    fourth1 = layers.BatchNormalization()(fourth)
    fourth2 = layers.LeakyReLU()(fourth1)
    
    fifth = layers.Conv3D(1, (2, 4, 4), strides=(1,1,1), padding='VALID')(fourth2)
    fifth1= layers.Reshape((-1,1))(fifth)
    
    
    out =tf.nn.sigmoid(fifth1)
    
    
    model = tf.keras.Model(inputs=inputs, outputs=[out])

    return model

discriminator = make_discriminator_model(disShape)
generator = make_generator_model(genShape)


print(discriminator.summary())
print('_________________________________________________________')
print(generator.summary())

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

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


def train():
    device = '/cpu:0'
    if use_gpu:
        device = '/gpu:0'

    training_videos  = read_and_load_video_all_files('./ballet')
    process_and_write_video(training_videos[0],'example_output')
    output_pose_video_matrix(training_videos[0],'example_output')

    with tf.device(device): #assuming we have working gpu's this should take adv
        generator_optimizer = tf.keras.optimizers.Adam( 0.0001, beta_1 = 0.5)
        discriminator_optimizer = tf.keras.optimizers.Adam( 0.0001, beta_1 = 0.5)
        
        for epoch in range(epochs):
            print("Epoch:",epoch)
            for counter in range(len(training_videos)):
                noise = tf.random.normal([1, genShape])
                video = training_videos[counter]
                
                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    generated_video = generator(noise, training=True)

                    real_output = discriminator(video, training=True)
                    fake_output = discriminator(generated_video, training=True)

                    gen_loss = generator_loss(fake_output)
                    disc_loss = discriminator_loss(real_output, fake_output)
                gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
                gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

                generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
                discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
            
            if epoch%10==0:
                noise = tf.random.normal([1, genShape])
                out_video= generator(noise,training=False)
                process_and_write_video(out_video,"video_epoch" + str(epoch))
                output_pose_video_matrix(out_video,"video_epoch" + str(epoch))
                save_check_point(generator_optimizer,discriminator_optimizer)

            

train()


