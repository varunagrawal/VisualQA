import json
import pickle
from collections import Counter
import torch
import torch.utils.data as data
from utils import text
import os
import os.path as osp
import numpy as np
from utils.image import coco_name_format
from PIL import Image


def get_train_dataloader(annotations, questions, images, args, vocab=None, raw_images=False,
                         transforms=None, shuffle=True):
    return data.DataLoader(VQADataset(annotations, questions, images, "train", args, raw_images=raw_images,
                                      vocab=vocab, transforms=transforms),
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 shuffle=shuffle)

def get_val_dataloader(annotations, questions, images, args, maps, vocab=None, raw_images=False,
                       transforms=None, shuffle=True):
    return data.DataLoader(VQADataset(annotations, questions, images, "val", args, raw_images=raw_images,
                                      vocab=vocab, transforms=transforms, maps=maps),
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 shuffle=shuffle)


class VQADataset(data.Dataset):
    def __init__(self, annotations, questions, images_dataset, split, args, raw_images=False, transforms=None,
                 vocab=None, normalize_img=True, maps=None, year=2014):

        # the data is saved as a dict where the key is the image_id and the value is the VGG feature vector
        if not raw_images:
            self.images_dataset = torch.load(images_dataset)
            print("Loaded {0} image embeddings dataset".format(split))

        self.split = split
        self.year = year

        self._process_dataset(annotations, questions, args, split, maps=maps)

        if vocab:
            self.vocab = vocab

        self.embed_question = args.embed_question

        self.raw_images = raw_images
        self.normalize_img = normalize_img
        self.transforms = transforms
        self.root = args.image_root

    def _process_dataset(self, annotations, questions, args,  split="train", maps=None):
        """
        Process the dataset and load it up.
        We should only do this for the training set.
        """
        self.data, self.vocab, \
        self.word_to_wid, self.wid_to_word, \
        self.ans_to_aid, self.aid_to_ans = process_vqa_dataset(questions, annotations, split, maps,
                                                               args.top_answer_limit, args.max_length)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        d = self.data[index]

        item = dict()

        if self.raw_images:
            item["image_id"] = d["image_id"]
            img = Image.open(osp.join(osp.expanduser(self.root), "{0}{1}".format(self.split, self.year),
                                      d["image_name"]))
            img = img.convert(mode='RGB')

            if self.transforms is not None:
                img = self.transforms(img)

        else:
            # Process Visual (image or features)
            # the preprocess script should have already saved these as Torch tensors
            item["image_id"] = d["image_id"]
            img = self.images_dataset[d["image_id"]].squeeze()

        item['image'] = img

        # Process Question (word token)
        item['question_id'] = d['question_id']
        if self.embed_question:
            item['question'] = torch.from_numpy(d['question_wids'])
        else:
            one_hot_vec = np.zeros((len(d["question_wids"]), len(self.vocab)))
            for k in range(len(d["question_wids"])):
                one_hot_vec[k, d['question_wids'][k]] = 1
            item['question'] = torch.from_numpy(one_hot_vec).float()

        item['answer_id'] = d['answer_id']
        item['answer_type'] = d['answer_type']

        return item


def process_vqa_dataset(questions_file, annotations_file, split, maps=None, top_answer_limit=1000, max_length=25,
                        year=2014):
    """
    Process the questions and annotations into a consolidated dataset.
    This is done only for the training set.
    :param questions_file:
    :param annotations_file:
    :param split: The dataset split.
    :param maps: Dict containing various mappings such as word_to_wid, wid_to_word, ans_to_aid and aid_to_ans.
    :param top_answer_limit:
    :param max_length:
    :param year: COCO Dataset release year.
    :return: The processed dataset ready to be used

    """
    cache_file = "vqa_{0}_dataset_cache.pickle".format(split)

    # Check if preprocessed cache exists. If yes, load it up, else preprocess the data
    if os.path.exists(cache_file):
        print("Found {0} set cache! Loading...".format(split))
        dataset, vocab, \
        word_to_wid, wid_to_word, \
        ans_to_aid, aid_to_ans = pickle.load(open(cache_file, 'rb'))

    else:
        # load the annotations and questions files
        print("Loading {0} annotations".format(split))
        with open(annotations_file) as ann:
            j = json.load(ann)
            annotations = j["annotations"]

        print("Loading {0} questions".format(split))
        with open(questions_file) as q:
            j = json.load(q)
            questions = j["questions"]

        # load up the dataset
        dataset = []
        for idx, q in enumerate(questions):
            d = dict()
            d["question_id"] = q["question_id"]
            d["question"] = q["question"]
            d["image_id"] = q["image_id"]
            d["image_name"] = coco_name_format(q["image_id"], split, year)

            d["answer"] = annotations[idx]["multiple_choice_answer"]
            answers = []
            for ans in annotations[idx]['answers']:
                answers.append(ans['answer'])
            d['answers_occurence'] = Counter(answers).most_common()

            d["question_type"] = annotations[idx]["question_type"]
            d["answer_type"] = annotations[idx]["answer_type"]

            dataset.append(d)

        # Get the top N answers so we can filter the dataset to only questions with these answers
        top_answers = text.get_top_answers(dataset, top_answer_limit)
        dataset = text.filter_dataset(dataset, top_answers)

        # Process the questions
        dataset = text.preprocess_questions(dataset)

        if split == "train":
            vocab = text.get_vocabulary(dataset)
            word_to_wid = {w:i for i, w in enumerate(vocab)}
            wid_to_word = {i:w for i, w in enumerate(vocab)}
            ans_to_aid = {a: i for i, a in enumerate(top_answers)}
            aid_to_ans = {i: a for i, a in enumerate(top_answers)}

        else: # split == "val":
            vocab = maps["vocab"]
            word_to_wid = maps["word_to_wid"]
            wid_to_word = maps["wid_to_word"]
            ans_to_aid = maps["ans_to_aid"]
            aid_to_ans = maps["aid_to_ans"]

        dataset = text.remove_tail_words(dataset, vocab)
        dataset = text.encode_questions(dataset, word_to_wid, max_length)
        dataset = text.encode_answers(dataset, ans_to_aid)

        print("Caching the processed data")
        pickle.dump([dataset, vocab, word_to_wid, wid_to_word, ans_to_aid, aid_to_ans],
                    open(cache_file, 'wb+'))

    return dataset, vocab, word_to_wid, wid_to_word, ans_to_aid, aid_to_ans
