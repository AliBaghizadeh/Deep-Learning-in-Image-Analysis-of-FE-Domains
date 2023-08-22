import os
import numpy as np
from PIL import Image
from glob import glob

# Load images from folders
def load_images_from_folder(folder, noisy = False, noise_std = 0.2, image_size = 512):
    """
    Function: this function read images from train/test folder, resize them, convert them to gray scale and numpy array, add noise
    folder: the path to train or test folders 
    noisy: To add gaussian noise to train and test noisy datasets
    """
    images = []
    for subdir in os.listdir(folder):
        sub_dir = os.path.join(folder, subdir)
        if not os.path.isdir(sub_dir):
            continue
        for filename in os.listdir(sub_dir):
            img = Image.open(os.path.join(sub_dir, filename))
            img = img.resize((image_size, image_size))
            img = img.convert('L')

            if noisy:
                noise = np.random.normal(scale = noise_std, size = img.size)
                img_array = np.array(img)/255.0
                img_array += noise
                img_array = np.clip(img_array, 0.0, 1.0)
                img = Image.fromarray((img_array*255).astype(np.uint8))

            #Normalize pixel values to the range [0, 1] and add channel for model training
            img_array = np.array(img)/255.0
            img_array = np.expand_dims(img_array, axis = -1)

            images.append(img_array)

    return images

def custom_dataset(folder_source, folder_dest, image_size, train_set, noise_std):
    """
    Function: Load and preprocess images from the given source folder, and organize the dataset.
    folder_source: the path to the source folder containing images
    folder_dest: the path to the destination folder for the dataset
    image_size: the desired size for the images
    train_set: the proportion of images to use for training (e.g., 0.8 for 80% train, 20% test)
    """
    #folder_path = os.getcwd()

    #folder_source = os.path.join(folder_path, "noisy images new")
    #folder_dest = os.path.join(folder_path, "noisy images new dataset")

    # Create train and test directories
    os.makedirs(os.path.join(folder_dest, "train_set"), exist_ok = True)
    os.makedirs(os.path.join(folder_dest, "test_set"), exist_ok = True)

    # Split train_set and valid_set

    for class_name in os.listdir(folder_source):
        class_folder_source = os.path.join(folder_source, class_name)
        class_images = os.listdir(class_folder_source)
        np.random.shuffle(class_images)
        split_index = int(len(class_images) * train_set)

        for i, image_name in enumerate(class_images):
            source_path = os.path.join(class_folder_source, image_name)
            dest_folder = "train_set" if i < split_index else "test_set"
            dest_folder_path = os.path.join(folder_dest, dest_folder, class_name)
            os.makedirs(dest_folder_path, exist_ok=True)

            dest_path = os.path.join(dest_folder_path, image_name)

            try:
                os.link(source_path, dest_path)
            except FileExistsError:
                os.replace(source_path, dest_path)
    
    # Load training and testing datasets of original images
    train_folder_clean = os.path.join(folder_dest, "train_set")
    test_folder_clean = os.path.join(folder_dest, "test_set")

    # Create datasets of clean images
    x_train = load_images_from_folder(train_folder_clean)
    x_test = load_images_from_folder(test_folder_clean)

    #Create datasets of noisy images
    x_train_noisy = load_images_from_folder(train_folder_clean, noisy = True, noise_std =noise_std)
    x_test_noisy = load_images_from_folder(test_folder_clean, noisy = True, noise_std =noise_std)
    
    # Add print statements to check the number of loaded images
    print("Number of clean training images:", len(x_train))
    print("Number of clean testing images:", len(x_test))
    print("Number of noisy training images:", len(x_train_noisy))
    print("Number of noisy testing images:", len(x_test_noisy))

    #Convert the image lists to numpy arrays
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    x_train_noisy = np.array(x_train_noisy)
    x_test_noisy = np.array(x_test_noisy)

    #Reshape the arrays to the shape (-1, image_size, image_size, 1) for gray color
    x_train = x_train.reshape(-1, image_size, image_size, 1)
    x_test = x_test.reshape(-1, image_size, image_size, 1)
    x_train_noisy = x_train_noisy.reshape(-1, image_size, image_size, 1)
    x_test_noisy = x_test_noisy.reshape(-1, image_size, image_size, 1)

    #Verify the shapes of the arrays
    print("x_train shape:", x_train.shape)
    print("x_test shape:", x_test.shape)
    print("x_train_noisy shape:", x_train_noisy.shape)
    print("x_test_noisy shape:", x_test_noisy.shape)

    # Save the preprocessed datasets to the destination folder
    np.save(os.path.join(folder_dest, "x_train.npy"), x_train)
    np.save(os.path.join(folder_dest, "x_test.npy"), x_test)
    np.save(os.path.join(folder_dest, "x_train_noisy.npy"), x_train_noisy)
    np.save(os.path.join(folder_dest, "x_test_noisy.npy"), x_test_noisy)