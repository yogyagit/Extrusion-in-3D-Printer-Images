import torch
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    """
    A custom dataset class for handling image data with labels.
    This class is used for training and validation datasets that include labels.
    """
    def __init__(self, dataframe, transform=None):
        """
        Initializes the dataset with a pandas dataframe and optional transforms.
        :param dataframe: DataFrame containing the image paths and their corresponding labels.
        :param transform: Optional transform to be applied on a sample.
        """
        self.df = dataframe  # DataFrame containing image paths and labels
        self.transform = transform  # Transformations to be applied to images

    def __len__(self):
        """
        Returns the size of the dataset.
        """
        return len(self.df)  # Number of items in the dataset

    def __getitem__(self, index):
        """
        Fetches the image and label at a given index in the dataset.
        :param index: Index of the data item.
        :return: Tuple of image and its corresponding label.
        """
        img_path = self.df.iloc[index]['img_path']  # Get image path
        img_label = self.df.iloc[index]['has_under_extrusion']  # Get image label

        img = Image.open(img_path).convert('RGB')  # Open image file and convert it to RGB

        if self.transform is not None:
            img = self.transform(img)  # Apply transformations if any

        return img, img_label  # Return the processed image and its label


class CustomDatasetTest(Dataset):
    """
    A custom dataset class for handling image data without labels.
    This class is used for test datasets where labels are not available.
    """
    def __init__(self, dataframe, transform=None):
        """
        Initializes the dataset with a pandas dataframe and optional transforms.
        :param dataframe: DataFrame containing the image paths.
        :param transform: Optional transform to be applied on a sample.
        """
        self.df = dataframe  # DataFrame containing image paths
        self.transform = transform  # Transformations to be applied to images

    def __len__(self):
        """
        Returns the size of the dataset.
        """
        return len(self.df)  # Number of items in the dataset

    def __getitem__(self, index):
        """
        Fetches the image at a given index in the dataset.
        :param index: Index of the data item.
        :return: Processed image.
        """
        img_path = self.df.iloc[index]['img_path']  # Get image path

        img = Image.open(img_path).convert('RGB')  # Open image file and convert it to RGB

        if self.transform is not None:
            img = self.transform(img)  # Apply transformations if any

        return img  # Return the processed image
