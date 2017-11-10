import json
from collections import Counter
import torch
import torch.utils.data as data
import utils
import os


def coco_name_format(image_id, split):
    image_name = "COCO_{0}2014_{1:012}.jpg".format(split, image_id)
    return image_name


def get_dataloader(annotations, questions, images, split, args, shuffle=True):
    return data.DataLoader(VQADataset(annotations, questions, images, split, args),
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 shuffle=shuffle)


class VQADataset(data.Dataset):
    def __init__(self, annotations, questions, images_dataset, split, args):
        print("Loading annotations")
        with open(annotations) as ann:
            j = json.load(ann)
            self.annotations = j["annotations"]

        print("Loading questions")
        with open(questions) as q:
            j = json.load(q)
            self.questions = j["questions"]

        # the images_dataset is the joblib file
        # the data is saved as a dict where the key is the image_id and the value is the VGG feature vector
        self.images_dataset = torch.load(images_dataset)
        self.split = split

        cache_file = "vqa_dataset_cache.json"
        if os.path.exists(cache_file):
            print("Found dataset cache! Loading...")
            self.data = json.load(open(cache_file))
        else:
            self.data = process_vqa_dataset(self.questions, self.annotations, split, args)
            print("Caching the processed data")
            json.dump(self.data, open(cache_file, 'w+'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        val = self.data[index]

        item = {}

        # Process Visual (image or features)
        # the preprocess script should have already saved these as Torch tensors
        item['visual'] = self.images_dataset[val["image_id"]]

        # Process Question (word token)
        item['question_id'] = val['question_id']
        item['question'] = torch.LongTensor(val['question_wids'])
        item['answer_id'] = val['answer_id']

        return item


def process_vqa_dataset(questions, annotations, split, args):
    """
    Process the questions and annotations into a consolidated dataset
    :param questions:
    :param annotations:
    :param images:
    :param split:
    :param args:
    :return: The processed dataset ready to be used

    """
    dataset = []
    for idx, q in enumerate(questions):
        d = {}
        d["question_id"] = q["question_id"]
        d["question"] = q["question"]
        d["image_id"] = q["image_id"]
        d["image_name"] = coco_name_format(q["image_id"], split)

        d["answer"] = annotations[idx]["multiple_choice_answer"]
        answers = []
        for ans in annotations[idx]['answers']:
            answers.append(ans['answer'])
        d['answers_occurence'] = Counter(answers).most_common()

        dataset.append(d)

    # Get the top 1000 answers so we can filter the dataset to only questions with these answers
    top_answers = utils.get_top_answers(dataset, args.top_answer_limit)
    dataset = utils.filter_dataset(dataset, top_answers)

    # Process the questions
    dataset = utils.preprocess_questions(dataset)
    vocab = utils.get_vocabulary(dataset)
    dataset = utils.remove_tail_words(dataset, vocab)

    word_to_wid = {w:i for i, w in enumerate(vocab)}
    wid_to_word = [w for w in vocab]

    dataset = utils.encode_questions(dataset, word_to_wid, args.max_length)

    ans_to_aid = {a: i for i, a in enumerate(top_answers)}
    aid_to_ans = [a for a in top_answers]

    dataset = utils.encode_answers(dataset, ans_to_aid)

    return dataset
