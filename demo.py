import argparse
from models import Models
from utils import image, text
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from dataset import process_vqa_dataset


def parse_args():
    parser = argparse.ArgumentParser("VQA Demo")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("question", help="Question text")
    parser.add_argument("questions", help="Path to VQA Questions training file")
    parser.add_argument("annotations", help="Path to COCO Annotations training file")
    parser.add_argument("--model", default="DeeperLSTM")
    parser.add_argument("--weights", default="weights/deeper_lstm_best_weights.pth.tar")
    parser.add_argument("--preprocessed_cache", default="vqa_train_dataset_cache.pickle")
    parser.add_argument("--embedding_arch", default="vgg")

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


def display_result(image, question, answer):
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), question)
    draw.text((10, 20), answer)
    print("{0}: {1}".format(question, answer))
    image.show()


def main():
    args = parse_args()

    print("Loading encoded data...")
    data, vocab, word_to_wid, wid_to_word, \
    ans_to_aid, aid_to_ans = process_vqa_dataset(args.questions, args.annotations, "train", maps=None)

    # Get VGG model to process the image
    vision_model, _ = image.get_model(args.embedding_arch)
    # Get our VQA model
    model = Models[args.model].value(len(vocab))
    # The final classifier
    classifier = nn.Softmax(dim=1)

    try:
        weights = torch.load(args.weights)
    except (Exception,):
        print("ERROR: Default weights missing. Please specify weights for the VQA model")
        exit(0)

    model.load_state_dict(weights["model"])

    if torch.cuda.is_available():
        vision_model.cuda()
        model.cuda()

    vision_model.eval()
    model.eval()

    img_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    print("Processing image")
    im = Image.open(args.image)
    img = img_transforms(im)
    img = img.unsqueeze(0)  # add batch dimension

    if torch.cuda.is_available():
        img = img.cuda()

    img_features = vision_model(img)

    print("Processing question")
    q = text.process_single_question(args.question, vocab, word_to_wid)

    # Convert the question to a sequence of 1 hot vectors over the vocab
    one_hot_vec = np.zeros((len(q["question_wids"]), len(vocab)))
    for k in range(len(q["question_wids"])):
        one_hot_vec[k, q['question_wids'][k]] = 1

    q = torch.from_numpy(one_hot_vec)
    if torch.cuda.is_available():
        q = q.cuda()

    # Add the batch dimension
    q = q.unsqueeze(0).float()

    # Get the model output and classify for the final value
    output = model(img_features, q)
    output = classifier(output).data

    _, ans_id = torch.max(output, dim=1)
    # index into ans_id since it is a tensor
    ans = generate(ans_id[0], aid_to_ans)

    display_result(im, args.question, ans)


if __name__ == "__main__":
    main()
