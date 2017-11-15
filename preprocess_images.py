"""
Script to generate embedding of images from COCO imags using VGG16
"""

import argparse
import json
import torch
from torchvision import transforms
from torch.utils import data
from torch.autograd import Variable
from tqdm import tqdm
from PIL import Image
import os.path as osp
import utils.image


class COCODataset(data.Dataset):
    def __init__(self, input_file, root, transform):
        data = json.load(open(input_file))
        self.data =  data["images"]
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        img = Image.open(osp.join(self.root, item["file_name"]))
        img = img.convert(mode='RGB')

        # convert to Tensor so we can batch it
        id = int(item["id"])
        id = torch.LongTensor([id])

        if self.transform is not None:
            img = self.transform(img)

        return img, id


def main(file, root, split):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
            transforms.Scale(256),
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    coco_dataset_file = file
    root = root

    coco = COCODataset(coco_dataset_file, root=root, transform=transform)
    data_loader = data.DataLoader(coco, batch_size=1, shuffle=False, num_workers=4)

    model = utils.image.get_model()
    if torch.cuda.is_available():
        model.cuda()

    model.eval()

    embeddings = {}

    print("Starting")
    for idx, (img, id) in enumerate(tqdm(data_loader, total=len(data_loader))):
        img_var = Variable(img).cuda()
        embedding = model(img_var)
        embeddings[id[0, 0]] = embedding.data.cpu()

    print("Done computing embeddings")

    torch.save(embeddings, "coco_{0}_vgg_fc7.pth".format(split))


parser = argparse.ArgumentParser("Standalone utility to preprocess COCO images")

parser.add_argument("file", help="Path to COCO annotations file")
parser.add_argument("--root", help="Path to the train/val root directory of images")
parser.add_argument("--split", default="train", choices=("train", "val"))

args = parser.parse_args()

# "/home/varun/datasets/MSCOCO/annotations/instances_{0}2014.json".format(split)
# "/home/varun/datasets/MSCOCO/{0}2014".format(split)
main(args.file, args.root, args.split)
