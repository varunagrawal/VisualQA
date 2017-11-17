import argparse
from models import Models
from utils import image, text
from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
import pickle


def parse_args():
    parser = argparse.ArgumentParser("VQA Demo")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("question", help="Question text")
    parser.add_argument("--model", default="DeeperLSTM")
    parser.add_argument("--weights")
    parser.add_argument("--preprocessed_cache", default="vqa_train_dataset_cache.pickle")

    return parser.parse_args()


def generate(output, aid_to_ans):
    """
    Return the answer given the Answer ID/label
    :param output: The answer label
    :param aid_to_ans:
    :return:
    """
    ans = aid_to_ans[output]
    return ans


def main():
    # TODO Test this

    args = parse_args()

    data, vocab, word_to_wid, wid_to_word, ans_to_aid, aid_to_ans = pickle.load(open(args.preprocessed_cache, 'rb'))

    # Get VGG model to process the image
    vision_model = image.get_model()
    # Get our VQA model
    model = Models[args.model].value(len(vocab))

    weights = torch.load(args.weights)
    model.load_state_dict(weights["model"])

    if torch.cuda.is_available():
        vision_model.cuda()
        model.cuda()

    vision_model.eval()
    model.eval()

    with open(args.image, 'rb') as f:
        img = Image.open(f)
        img = transforms.ToTensor()(img)

    img_var = Variable(img)
    if torch.cuda.is_available():
        img_var = img_var.cuda()

    img_features = vision_model(img_var)

    q = text.process_single_question(args.question, vocab, word_to_wid)

    one_hot_vec = np.zeros((len(q["question_wids"]), len(vocab)))
    for k in range(len(q["question_wids"])):
        one_hot_vec[k, q['question_wids'][k]] = 1

    q_var = Variable(torch.from_numpy(one_hot_vec))
    if torch.cuda.is_available():
        q_var = q_var.cuda()

    output = model(img_features, q_var)
    ans = generate(output, aid_to_ans)
    print("The answer is: {0}".format(ans))


if __name__ == "__main__":
    main()
