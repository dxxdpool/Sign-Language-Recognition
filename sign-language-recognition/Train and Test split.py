import os
import shutil
from sklearn.model_selection import train_test_split


def split_image_directory(source_dir, train_dir, test_dir, valid_dir, test_size=0.23, valid_size=0.23, random_state=95):
    # Create train, test, and validation directories if they don't exist
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)

    # Get list of all images in the source directory
    images = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

    # Split the images into train, test, and validation sets
    train_images, remaining_images = train_test_split(images, test_size=test_size + valid_size,
                                                      random_state=random_state)
    test_images, valid_images = train_test_split(remaining_images, test_size=valid_size / (test_size + valid_size),
                                                 random_state=random_state)

    # Copy train images to train directory
    for img in train_images:
        shutil.copy(os.path.join(source_dir, img), os.path.join(train_dir, img))

    # Copy test images to test directory
    for img in test_images:
        shutil.copy(os.path.join(source_dir, img), os.path.join(test_dir, img))

    # Copy validation images to validation directory
    for img in valid_images:
        shutil.copy(os.path.join(source_dir, img), os.path.join(valid_dir, img))


# Example usage:
ch = 'i'
source_directory = f"ImageData/{ch}"
train_directory = f"ImageData/train/{ch}"
test_directory = f"ImageData/test/{ch}"
valid_directory = f"ImageData/valid/{ch}"
split_image_directory(source_directory, train_directory, test_directory, valid_directory, test_size=30 / 130,
                      valid_size=30 / 130, random_state=95)
