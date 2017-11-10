import argparse
import json
import torch
from torchvision import models, transforms
from torch.utils import data
from torch.autograd import Variable
from tqdm import tqdm
from PIL import Image
import os.path as osp


def get_model():
    model = models.vgg16(pretrained=True)
    model.features = torch.nn.DataParallel(model.features)
    modules = list(model.classifier.children())
    # restrict to the FC layer that gives us the 4096 embedding
    modules = modules[:-3]
    model.classifier = torch.nn.Sequential(*modules)
    return model


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

    model = get_model()
    model.cuda()
    model.eval()

    embeddings = {}

    print("Starting")
    for idx, (img, id) in enumerate(tqdm(data_loader, total=len(data_loader))):
        img_var = Variable(img).cuda()
        embedding = model(img_var)
        embeddings[str(id[0, 0])] = embedding.data.cpu()

    print("Done computing embeddings")

    torch.save(embeddings, "coco_{0}_vgg_fc7.pth".format(split))


parser = argparse.ArgumentParser("Preprocess COCO images")

parser.add_argument("file", help="COCO annotations file")
parser.add_argument("--root", help="The root directory of the train/val folders")
parser.add_argument("--split", default="train", choices=("train", "val"))

args = parser.parse_args()

# "/home/varun/datasets/MSCOCO/annotations/instances_{0}2014.json".format(split)
# "/home/varun/datasets/MSCOCO/{0}2014".format(split)
main(args.file, args.root, args.split)
