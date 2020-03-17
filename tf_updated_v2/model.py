import glob
import numpy as np
import os
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import datetime
from utils import *
import csv

genShape= 100
disShape = [32,64,64,3]

imShape = [64, 64, 3]

def make_generator_model(imageShape):
    inputs = layers.Input(shape=imageShape)
    # TODO: Shape conversion seems jank
    # inputs = layers.Input(shape=noiseShape)

    # inputs = tf.reshape(inputs, [1,64,64,3])

    #Encoder
    # TODO: Figure out correct activation layer - using leaky for now 
    print(inputs.shape)
    enc0 = layers.Conv2D(64, (4,4), strides=(2,2),padding="SAME")(inputs)
    enc0 = layers.BatchNormalization()(enc0)
    enc1 = layers.LeakyReLU()(enc0)

    enc1 = layers.Conv2D(128, (4,4), strides=(2,2),padding="SAME")(enc1)
    enc1 = layers.BatchNormalization()(enc1)
    enc2 = layers.LeakyReLU()(enc1)

    enc2 = layers.Conv2D(256, (4,4), strides=(2,2),padding="SAME")(enc2)
    enc2 = layers.BatchNormalization()(enc2)
    enc3 = layers.LeakyReLU()(enc2)

    enc3 = layers.Conv2D(512, (4,4), strides=(2,2),padding="SAME")(enc3)
    enc3 = layers.BatchNormalization()(enc3)
    enc4 = layers.LeakyReLU()(enc3)

    encfinal = enc4
    
    # Background
    # z = tf.reshape(inputs,[-1,1,1,noiseShape])
    # Note padding here is changed
    backg = layers.Conv2DTranspose(512, (4, 4), strides=(1,1),padding="SAME")(encfinal)
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
    # z = tf.reshape(inputs,[-1,1,1,1,noiseShape])
    foreg0 = tf.expand_dims(encfinal,1)

    # TODO: originally (2,4,4) and (1,1,1) -> verify the change is correct
    foreg = layers.Conv3DTranspose(512, (4,4,2), strides=(2,1,1),padding="SAME",use_bias=False)(foreg0)
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
    
    #backg layer
    backg = layers.Conv3D(64, (4, 4,4), strides=(2,2,2), padding='same')(inputs)
    backg1 = layers.LeakyReLU()(backg)
    
    
    #second layer
    second = layers.Conv3D(128, (4, 4, 4), strides=(2,2,2), padding='same')(backg1)
    second1 = layers.BatchNormalization()(second)
    second2 =  layers.LeakyReLU()(second1)

    #third layer
    third = layers.Conv3D(256, (4, 4, 4), strides=(2,2,2), padding='same')(second2)
    third1 = layers.BatchNormalization()(third)
    third2 = layers.LeakyReLU()(third1)
    
    #fourth layer
    fourth = layers.Conv3D(512, (4, 4, 4), strides=(2,2,2), padding='same')(third2)
    fourth1 = layers.BatchNormalization()(fourth)
    fourth2 = layers.LeakyReLU()(fourth1)
    
    fifth = layers.Conv3D(1, (2, 4, 4), strides=(1,1,1), padding='same')(fourth2)
    fifth1= layers.Reshape((-1,1))(fifth)
    # print(tf.shape(fifth1))
    
    out =tf.nn.sigmoid(fifth1)
    
    model = tf.keras.Model(inputs=inputs, outputs=[out,fifth1])

    return model

discriminator = make_discriminator_model(disShape)
# generator = make_generator_model(genShape)
generator = make_generator_model(imShape)
print(generator.summary())


def get_generator_loss(inputData):
    fake_video,_,_,_ = generator(inputData)
    prob_fake, logits_fake = discriminator(fake_video,training = False)
    g_cost = tf.reduce_mean(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_fake, labels = tf.ones_like(prob_fake))) 
    return g_cost


def get_discriminator_loss(real_video, fake_video):
    # lambd=  0.0
    prob_real, logits_real = discriminator(real_video)
    prob_fake, logits_fake = discriminator(fake_video,training = False)

    # print(prob_real.shape, logits_real.shape)
    # g_cost = tf.reduce_mean(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_fake, labels = tf.ones_like(prob_fake))) 

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
    epochs = 600
    generator_optimizer = tf.keras.optimizers.Adam( 0.0002, beta_1 = 0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam( 0.0002, beta_1 = 0.5)
    
    if checkpoint_file != '':
        print('Fill in code to restore model')

    # read in all the videos into one glob
    data_files = glob.glob("./trainvideos/*")

    # lists to store all the data separately
    video_lst = []
    first_frame_lst = []

    for counter in range(len(data_files)):
        # noise = tf.random.normal([1, genShape])
        # print("....Iteration....:", counter)
        # print(datetime.datetime.now()) #added for keeping track of time

        # process videos into lists immediately - makes epochs go faster after processing
        batch_files = data_files[counter*1:(counter+1)*1]
        videos, first_frame = read_and_process_video(batch_files,1,32)
        video_lst.append(videos)
        first_frame_lst.append(first_frame)

        # print(first_frame.shape)
        # first_frame = np.array(first_frame).reshape((1,64,64,3))
        print("Processed Video Number " + str(counter) )


    # format all the frames to the rightshape
    for i in range(len(first_frame_lst)):
        first_frame_lst[i] = np.array(first_frame_lst[i]).reshape((1,64,64,3))
    gLoss = []
    dLoss = []
    for epoch in range(epochs):
        print("Epoch:",epoch)
        for counter in range(len(data_files)):
            # noise = tf.random.normal([1, genShape])
            print("....Iteration....:", counter)
            print(datetime.datetime.now()) #added for keeping track of time

            videos = video_lst[counter]
            first_frame = first_frame_lst[counter]



            with tf.GradientTape() as tape:
                fake_video,_,_,_ = generator(first_frame)
                d_loss = get_discriminator_loss(videos,fake_video)
            grads = tape.gradient(d_loss, discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

            with tf.GradientTape() as tape:
                g_loss = get_generator_loss(first_frame)
            grads = tape.gradient(g_loss, generator.trainable_variables)
            generator_optimizer.apply_gradients(zip(grads, generator.trainable_variables))
        
            print(epoch, 'd loss:', float(d_loss), 'g loss:', float(g_loss))
            gLoss.append(float(g_loss))
            dLoss.append(float(d_loss))
            gen_videos,_,bg,_ = generator(first_frame,training=False)


            print(bg.shape, gen_videos.shape)
            if counter==9:
                process_and_write_image(bg,"bg_epoch" + str(epoch) + "_count" + str(counter))
                process_and_write_video(gen_videos,"video_epoch" + str(epoch) + "_count" + str(counter))
    
            
        
        save_check_point(generator_optimizer,discriminator_optimizer)
    with open('Loss.csv', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(zip(gLoss, dLoss))
               



clear_genvideos('genvideos') # comment out if you want to keep you gen videos
train('')