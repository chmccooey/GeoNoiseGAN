# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 15:39:35 2020

@author: Christopher
"""

from geo_gan import TrainGAN

# constant hyper parameters that will be used in training
data_path = 'D:\\College_Stuff\\Summer_2020_research_job\\GeoNoiseGAN\\gan_input'
num_epochs = 200 # Number of epochs
image_output_rate_per_batch = 100 # generate images. For instance if this value was 100 it would generate and save output image every 100 batches
batch_size = 10 # size of each batch for training
num_test_samples = 16 # The number of samples to generate each time output images are saved to the disk
learning_rate = 0.0002 # The learning rate of both the generator and discriminator ADAM optimizers

# Run the training process
TrainGAN(data_path, num_epochs, image_output_rate_per_batch, batch_size, num_test_samples, learning_rate)
