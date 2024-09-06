"""
Code to create random subsets of the dataset images
================================================
ishmamt
================================================
"""

import os
import shutil
import random
import errno

def select_random_images(source_dir, destination_dir, num_images, seed_value=42):
    # Check if source file exists
    if not os.path.exists(source_dir):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), source_dir)
    
    # Set the seed for reproducibility
    random.seed(seed_value)
    
    # Create destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    
    # List all files in the source directory
    all_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    
    # Filter image files (you can add more extensions if needed)
    image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'))]
    
    # Select a random subset of images
    selected_images = random.sample(image_files, min(num_images, len(image_files)))
    
    # Copy selected images to the destination directory
    for image in selected_images:
        shutil.copy(os.path.join(source_dir, image), destination_dir)
    
    print(f"Copied {len(selected_images)} images to {destination_dir}")


# Example usage
if __name__ == '__main__':
    source_directory = os.path.join("Data", "Images")
    destination_directory = os.path.join("Data", "NewImages")
    number_of_images = 5  # Number of images to select

    select_random_images(source_directory, destination_directory, number_of_images, seed_value=42)
