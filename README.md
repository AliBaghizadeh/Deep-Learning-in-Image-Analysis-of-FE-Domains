# Deep-Learning-in-Image-Analysis-of-FE-Domains

# Deep-Learning-on-Synthetic-Electron-Microscopy-Images

### Introduction
The nano-scale phenomena in science are very common, resulting in extracting significant information about the physical and chemical properties of materials. One common way to image nano-scale phenomena is high-resolution scanning electron microscopy (STEM) which enables one to acquire images on an atomic scale. However, preparing a sample for such a complex experiment and acquiring images at a few million magnifications is costly and requires a good amount of time. I believe that with available few images and knowing the possible physical models governing the phenomena we are looking for, we can use AI techniques to reduce some experimental or post-processing steps (usually manual). To this end, one can create artificial datasets either by coding or using generative AI. In this project, my objective is to showcase the potential of having a synthetic dataset and training an autoencoder on the dataset to reduce noise in images and use the model to generate similar images too.   

### Hardware and Software
The computer I used for this project is equipped to a 13th Gen Intel(R) Core(TM) i9-13900KF processor with 64 GB RAM, and NVIDIA GForce 4080 (Compute Capability 8.9) with 16 GB RAM. I found this configuration quite well for processing image datasets in medium size. <br><br>
To accomplish the project, I have used following packages:<br><br>
1- tensorflow-gpu  version 2.6.0  <br>
2- scikit-learn    version 1.2.1  <br>
3- Python          version 3.9.16 <br>
4- cudatoolkit     version 11.3.1  <br>

### Files and Instructions

1- ***random_lattice_split_data***:  <br>
&nbsp;&nbsp;&nbsp;This file will split your original dataset to train and valid folders. This approach is more suitable for using Image Augmentation generator of  &nbsp;&nbsp;&nbsp;the Keras and in case you need to preserve the structure of the folder of the original images and classes. I recommend having a test dataset 
&nbsp;&nbsp;&nbsp;as well. <br>
2- ***random_lattices_vgg_ImageAug***: <br>
&nbsp;&nbsp;&nbsp;This file applies image augmentation from keras.ImageAugmentation and then use pretrained weights of vgg model. This method is not 
&nbsp;&nbsp;&nbsp;memory friendly for large datasets.<br>
3- ***random_lattices_vgg_tf_data_dataset***:  <br>
&nbsp;&nbsp;&nbsp;This file applies image augmentation from tf.data.dataset on whole images, split data to train and valid datasets and then uses pretrained &nbsp;&nbsp;&nbsp;weights of vgg model for training a model with optional dense layers. This method is memory friendly for large datasets.<br>
4- ***random_lattices_ResNet50***: <br>
&nbsp;&nbsp;&nbsp;This file applies image augmentation from tf.data.dataset on whole images, split data to train and valid datasets and then uses pretrained &nbsp;&nbsp;&nbsp;weights of ResNet50 model for training a model with optional dense layers. This method is memory friendly for large datasets.<br>
