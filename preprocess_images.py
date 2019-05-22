"""
Script to generate image embeddings from COCO images using a specified architecture
"""

import argparse
import json
import os
import os.path as osp

import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

import utils.image


class COCODataset(data.Dataset):
    def __init__(self, input_file, root, transform, split):
        dataset = json.load(open(input_file))
        self.data = dataset["images"]
        self.root = root
        self.transform = transform
        self.split = split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        img = Image.open(osp.join(self.root,
                                  "{0}2014".format(self.split),
                                  item["file_name"]))
        img = img.convert(mode='RGB')

        # convert to Tensor so we can batch it
        idt = torch.LongTensor([int(item["id"])])

        if self.transform is not None:
            img = self.transform(img)

        return img, idt


def arguments():
    parser = argparse.ArgumentParser(
        "Standalone utility to preprocess COCO images")

    parser.add_argument("file", help="Path to COCO annotations file")
    parser.add_argument("--arch", default="vgg19_bn",
                        choices=("vgg19_bn", "vgg16_bn", "resnet152"))
    parser.add_argument("--root",
                        help="Path to the train/val root directory of images")
    parser.add_argument("--split",
                        default="train", choices=("train", "val"))
    parser.add_argument("--gpu", type=int, default=0)

    return parser.parse_args()


def main(coco_dataset_file, root, split, arch, gpu):
    if torch.cuda.is_available():
        device = torch.device('cuda:{0}'.format(gpu))
    else:
        device = torch.device('cpu')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    data_loader = data.DataLoader(COCODataset(coco_dataset_file, root=root,
                                              transform=transform, split=split),
                                  batch_size=1, shuffle=False, num_workers=4)

    model, layer = utils.image.get_model(arch)
    print("Conv Net Model", arch, layer)

    model = model.eval().to(device)

    embeddings = {}

    print("Starting")
    with torch.no_grad():

        for idx, (img, idt) in enumerate(tqdm(data_loader, total=len(data_loader))):
            img = img.to(device)
            embedding = model(img)
            embeddings[idt.item()] = embedding.cpu()

    print("Done computing image embeddings")

    try:
        os.mkdir('image_embeddings')
    except FileExistsError:
        pass

    print("Saving image embedddings")
    image_embedding_file = osp.join("image_embeddings",
                                    "coco_{0}_{1}_{2}.pth".format(split, arch, layer))
    torch.save(embeddings, image_embedding_file)
    print("Saved to {}".format(image_embedding_file))


if __name__ == "__main__":
    args = arguments()
    main(args.file, args.root, args.split, arch=args.arch, gpu=args.gpu)
