import os
import datasets
from datasets import Dataset, Features, DatasetDict, ClassLabel
import pandas as pd
from PIL import Image
import math

class GenDataset:
    """
    This class transforms the Eurosat directory into a dataset to be used in HF datasets
    The goal is to transform this folder structure
        https://madm.dfki.de/files/sentinel/EuroSAT.zip
    in this https://huggingface.co/datasets/timm/eurosat-rgb/tree/main
    folder structure (using parquet files, auto converted with push_to_hub):
    """

    def __init__(self, foldername):
        curpath = os.path.abspath(os.curdir)
        self.fulldirPath = os.path.join(curpath, foldername)
        self.features = Features({
            'label': ClassLabel(names=os.listdir(self.fulldirPath)),
            'image': datasets.Image(decode=True)
        })
    
    def __call__(self):
        train_labels = []
        train_images = []
        
        validation_labels = []
        validation_images = []

        # slice each folder into 85% train / 15% validation
        for subfolder in os.listdir(self.fulldirPath):
            subfolder_path = os.path.join(self.fulldirPath, subfolder)
            
            # Check if the item is a directory
            if os.path.isdir(subfolder_path):
                # Get a list of all image files in the subfolder
                image_files = [f for f in os.listdir(subfolder_path) if f.endswith((".jpg", ".png", ".jpeg"))]
                
                # Calculate the number of files to be moved to the "validation" folder (15% of the total)
                num_validation_files = math.ceil(len(image_files) * 0.15)
                
                # Move the first 85% of the files to "train"
                for i in range(0, len(image_files) - num_validation_files):
                    source_path = os.path.join(subfolder_path, image_files[i])
                    print(f"appending {image_files[i]} to train {subfolder}")
                    train_labels.append(subfolder)
                    train_images.append(datasets.Image().encode_example(Image.open(source_path)))

                # Move the last 15% of the files to "validation"
                for i in range(len(image_files) - num_validation_files, len(image_files)):
                    source_path = os.path.join(subfolder_path, image_files[i])
                    print(f"appending {image_files[i]} to validation {subfolder} ")
                    validation_labels.append(subfolder)
                    validation_images.append(datasets.Image().encode_example(Image.open(source_path)))

        train_df = pd.DataFrame({
            'label': train_labels,
            'image': train_images
        })

        val_df = pd.DataFrame({
            'label': validation_labels,
            'image': validation_images
        })

        ds_dict = DatasetDict({
            'train': Dataset.from_pandas(train_df, features=self.features),
            'validation': Dataset.from_pandas(val_df, features=self.features)
        })

        print(ds_dict)
        ds_dict.push_to_hub('yaguilherme/eurosat-full')
        