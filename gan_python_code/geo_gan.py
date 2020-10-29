# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 15:11:33 2020

@author: ccmcc
@description: Original code that failed from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
	Tutorial with MNIST dataset ://github.com/diegoalejogm/gans/blob/master/2.%20DC-GAN%20PyTorch-MNIST.ipynb
"""

#!/usr/bin/env python

# Generative Adversarial Networks (GAN) example in PyTorch. Tested with PyTorch 0.4.1, Python 3.6.7 (Nov 2018)
# See related blog post at https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f#.sch4xgsa9

#test_directory = 'C:\\Users\\ccmcc\\Desktop\\College_Stuff\\CSC_592_DeepLearning\\GH\\ocean_anomalies\\python_code\\tiles'
#delimiter = '\\'

#Chris print the PIL version (version 7+ fails to work with PyTorch)
import PIL

from IPython import display

from geo_logger import Logger

import os
import torch
import torchvision
from torch import nn
from torch.optim import Adam
from torch.autograd import Variable

from torchvision import transforms, datasets
from geo_discriminator import DiscriminativeNet
from geo_generator import GenerativeNet
from geo_generator import noise

DATA_FOLDER = './torch_data/DCGAN/GEODATA'
resize_value = 100

def TrainGAN(data_path:str, num_epochs:int, image_output_rate_per_batch:int, batch_size:int, num_test_samples:int, learning_rate:float):

    # Print information prior to training
    print('---- Beginning training process ----')
    
    print('data_path="{}"'.format(data_path))
    print('num_epochs={}'.format(num_epochs))
    print('image_output_rate_per_batch={}'.format(image_output_rate_per_batch))
    print('batch_size={}'.format(batch_size))
    print('num_test_samples={}'.format(num_test_samples))
    print('learning_rate={}'.format(learning_rate))
    print('PIL Library Version:', PIL.__version__)
    if torch.cuda.is_available():
        print('CUDA Available: Yes')
    else:
        print('CUDA Available: No')
    
    # check if dataset exists
    if (not os.path.isdir(data_path)):
        print("ERROR: The dataset path \"{}\" does not exist. Quiting...")
        return False
    
    # Create the training data
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        #transform=torchvision.transforms.ToTensor()
        transform = transforms.Compose(
            [
                transforms.Resize((resize_value, resize_value)),
                transforms.Grayscale(num_output_channels=1), #Convert RGB to Greyscale
                transforms.ToTensor(),
                transforms.Normalize([.5], [.5])
                #transforms.Normalize((.5, .5, .5), (.5, .5, .5))
            ])
    )
    
    # Check to make sure there is data
    total_training_size = len(train_dataset)
    print('Total images found = {}'.format(total_training_size))
    if (total_training_size <= 0):
        print("ERROR: There are no training images in the dataset. Quiting...")
        return False
    
    data = train_dataset
    
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    num_batches = len(data_loader)
    
    # Initialize weights
    def init_weights(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('BatchNorm') != -1:
            m.weight.data.normal_(0.00, 0.02)
    
    # Create Network instances and init weights
    generator = GenerativeNet()
    generator.apply(init_weights)
    
    discriminator = DiscriminativeNet()
    discriminator.apply(init_weights)
    
    # Enable cuda if available
    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()
        
    # Optimizers
    d_optimizer = Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    g_optimizer = Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    
    # Loss function
    loss = nn.BCELoss()
    
    # Training
    def real_data_target(size):
        '''
        Tensor containing ones, with shape = size
        '''
        data = Variable(torch.ones(size, 1))
        if torch.cuda.is_available(): return data.cuda()
        return data
    
    def fake_data_target(size):
        '''
        Tensor containing zeros, with shape = size
        '''
        data = Variable(torch.zeros(size, 1))
        if torch.cuda.is_available(): return data.cuda()
        return data
    
    def train_discriminator(optimizer, real_data, fake_data):
        # Reset gradients
        optimizer.zero_grad()
        
        # 1. Train on Real Data
        prediction_real = discriminator(real_data)
        # Calculate error and backpropagate
        error_real = loss(prediction_real, real_data_target(real_data.size(0)))
        error_real.backward()
    
        # 2. Train on Fake Data
        prediction_fake = discriminator(fake_data)
        # Calculate error and backpropagate
        error_fake = loss(prediction_fake, fake_data_target(real_data.size(0)))
        error_fake.backward()
        
        # Update weights with gradients
        optimizer.step()
        
        return error_real + error_fake, prediction_real, prediction_fake
        return (0, 0, 0)
    
    def train_generator(optimizer, fake_data):
        # Reset gradients
        optimizer.zero_grad()
        # Sample noise and generate fake data
        prediction = discriminator(fake_data)
        # Calculate error and backpropagate
        error = loss(prediction, real_data_target(prediction.size(0)))
        error.backward()
        # Update weights with gradients
        optimizer.step()
        # Return error
        return error
    
    # generate samples for testing
    num_test_samples = 16
    test_noise = noise(num_test_samples)
    
    # start training
    logger = Logger(model_name='DCGAN', data_name='GEODATA')
    
    for epoch in range(num_epochs):
        for n_batch, (real_batch,_) in enumerate(data_loader):
            
            # 1. Train Discriminator
            real_data = Variable(real_batch)
            print("At epoch {} for n_batch {}".format(epoch, n_batch))
            #print(type(real_data))
            if torch.cuda.is_available(): real_data = real_data.cuda()
            # Generate fake data
            fake_data = generator(noise(real_data.size(0))).detach()
            # Train D
            d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, 
                                                                    real_data, fake_data)
    
            # 2. Train Generator
            # Generate fake data
            fake_data = generator(noise(real_batch.size(0)))
            # Train G
            g_error = train_generator(g_optimizer, fake_data)
            # Log error
            logger.log(d_error, g_error, epoch, n_batch, num_batches)
            
            # Display Progress
            if (n_batch) % 100 == 0:
                display.clear_output(True)
                # Display Images
                test_images = generator(test_noise).data.cpu()
                logger.log_images(test_images, num_test_samples, epoch, n_batch, num_batches);
                # Display status Logs
                logger.display_status(
                    epoch, num_epochs, n_batch, num_batches,
                    d_error, g_error, d_pred_real, d_pred_fake
                )
            # Model Checkpoints
            logger.save_models(generator, discriminator, epoch)
    
    return True        
# end of train function
    