import json
import pickle
from collections import Counter
import torch
import torch.utils.data as data
from utils import text
import os
import numpy as np
from utils.image import coco_name_format


def get_train_dataloader(annotations, questions, images, args, vocab=None, shuffle=True):
    return data.DataLoader(VQADataset(annotations, questions, images, "train", args, vocab=vocab),
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 shuffle=shuffle)

def get_val_dataloader(annotations, questions, images, args, maps, vocab=None, shuffle=True):
    return data.DataLoader(VQADataset(annotations, questions, images, "val", args, vocab=vocab, maps=maps),
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 shuffle=shuffle)


class VQADataset(data.Dataset):
    def __init__(self, annotations, questions, images_dataset, split, args, vocab=None, normalize_img=True, maps=None):

        # the data is saved as a dict where the key is the image_id and the value is the VGG feature vector
        self.images_dataset = torch.load(images_dataset)
        self.split = split

        cache_file = "vqa_{0}_dataset_cache.pickle".format(split)

        # Check if preprocessed cache exists. If yes, load it up, else preprocess the data
        if os.path.exists(cache_file):
            print("Found dataset cache! Loading...")
            self.data, self.vocab, \
            self.word_to_wid, self.wid_to_word, \
            self.ans_to_aid, self.aid_to_ans = pickle.load(open(cache_file, 'rb'))

        else:
            print("Loading {0} annotations".format(split))
            with open(annotations) as ann:
                j = json.load(ann)
                self.annotations = j["annotations"]

            print("Loading {0} questions".format(split))
            with open(questions) as q:
                j = json.load(q)
                self.questions = j["questions"]

            self._process_dataset(args, cache_file, split, maps=maps)

        if vocab:
            self.vocab = vocab

        self.embed_question = args.embed_question
        self.normalize_img = normalize_img

    def _process_dataset(self, args,  cache_file, split="train", maps=None):
        """
        Process the dataset.
        We should only do this for the training set
        """
        self.data, self.vocab, \
        self.word_to_wid, self.wid_to_word, \
        self.ans_to_aid, self.aid_to_ans = process_vqa_dataset(self.questions, self.annotations, split, args, maps)

        print("Caching the processed data")
        pickle.dump([self.data, self.vocab, self.word_to_wid, self.wid_to_word, self.ans_to_aid, self.aid_to_ans],
                    open(cache_file, 'wb+'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        d = self.data[index]

        item = dict()

        # Process Visual (image or features)
        # the preprocess script should have already saved these as Torch tensors
        item["image_id"] = d["image_id"]
        img = self.images_dataset[d["image_id"]].squeeze()
        if self.normalize_img:
            norm = img.mul(img)
            norm = norm.sum(dim=0, keepdim=True).sqrt()
            img = img.div(norm)
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

        return item


def process_vqa_dataset(questions, annotations, split, args, maps=None):
    """
    Process the questions and annotations into a consolidated dataset.
    This is done only for the training set.
    :param questions:
    :param annotations:
    :param split:
    :param args:
    :param maps: Dict containing various mappings such as word_to_wid, wid_to_word, ans_to_aid and aid_to_ans
    :return: The processed dataset ready to be used

    """
    dataset = []
    for idx, q in enumerate(questions):
        d = dict()
        d["question_id"] = q["question_id"]
        d["question"] = q["question"]
        d["image_id"] = q["image_id"]
        d["image_name"] = coco_name_format(q["image_id"], "train")

        d["answer"] = annotations[idx]["multiple_choice_answer"]
        answers = []
        for ans in annotations[idx]['answers']:
            answers.append(ans['answer'])
        d['answers_occurence'] = Counter(answers).most_common()

        dataset.append(d)

    # Get the top 1000 answers so we can filter the dataset to only questions with these answers
    top_answers = text.get_top_answers(dataset, args.top_answer_limit)
    dataset = text.filter_dataset(dataset, top_answers)

    # Process the questions
    dataset = text.preprocess_questions(dataset)
    vocab = text.get_vocabulary(dataset)

    if split == "train":
        word_to_wid = {w:i for i, w in enumerate(vocab)}
        wid_to_word = {i:w for i, w in enumerate(vocab)}

        ans_to_aid = {a: i for i, a in enumerate(top_answers)}
        aid_to_ans = {i: a for i, a in enumerate(top_answers)}

    else: # split == "val":
        word_to_wid = maps["word_to_wid"]
        wid_to_word = maps["wid_to_word"]
        ans_to_aid = maps["ans_to_aid"]
        aid_to_ans = maps["aid_to_ans"]

    dataset = text.remove_tail_words(dataset, vocab)
    dataset = text.encode_questions(dataset, word_to_wid, args.max_length)
    dataset = text.encode_answers(dataset, ans_to_aid)

    return dataset, vocab, word_to_wid, wid_to_word, ans_to_aid, aid_to_ans
