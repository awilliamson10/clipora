import logging

import datasets
import pandas as pd
from open_clip import tokenize
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class HFDataset(Dataset):
    def __init__(self, data_location, transforms, image_col, text_col):
        logging.debug(f"Loading HF dataset from {data_location}.")
        self.dataset = datasets.load_dataset(data_location, split="train")
        self.image_col = image_col
        self.text_col = text_col
        self.transforms = transforms
        logging.debug("Done loading data.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        images = self.transforms(self.dataset[idx][self.image_col])
        texts = tokenize([self.dataset[idx][self.text_col]])[0]
        return images, texts


class CSVDataset(Dataset):
    def __init__(self, data_location, transforms, image_col, text_col, sep="\t"):
        logging.debug(f"Loading csv data from {data_location}.")
        df = pd.read_csv(data_location, sep=sep)

        self.images = df[image_col].tolist()
        self.captions = df[text_col].tolist()
        self.transforms = transforms
        logging.debug("Done loading data.")

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = tokenize([str(self.captions[idx])])[0]
        return images, texts


def get_dataloader(args, preprocess, split="train"):
    if args.datatype == "hf":
        dataset = HFDataset(
            data_location=args.train_dataset if split == "train" else args.eval_dataset,
            transforms=preprocess,
            image_col=args.image_col,
            text_col=args.text_col,
        )
    elif args.datatype == "csv":
        dataset = CSVDataset(
            data_location=args.train_dataset if split == "train" else args.eval_dataset,
            transforms=preprocess,
            image_col=args.image_col,
            text_col=args.text_col,
            sep=args.csv_separator,
        )
    num_samples = len(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)
    return dataloader
