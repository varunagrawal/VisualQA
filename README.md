# VQA

Visual Question Answering in PyTorch


## Setup

We use Pipenv since it is the Python recommended way of managing dependencies.

- Create your virtual environment and install all your dependencies with `pipenv install` in the project root directory.
- Download the VQA (2.0) dataset from [visualqa.org](http://visualqa.org/)
- I have already pre-processed all the COCO images using VGG-16. You can find them [here]().


## Running

### Training

To run the training code, just type

```
python main.py <options> TRAIN_ANN TRAIN_QUES TRAIN_IMGS VAL_ANN VAL_QUES VAL_IMGS
```

You can get a list of options with `python main.py -h`.

Check out `run.sh` to get an idea of how to run the code.

> *NOTE* The code will take care of all the text preprocessing. Just sit back and relax.


The minimum arguments required are:

1. The VQA train annotations dataset
2. The VQA train open-ended questions dataset
3. Path to the COCO training image feature embeddings
4. The VQA val annotations dataset
5. The VQA val open-ended questions dataset
6. Path to the COCO val image feature embeddings

### Demo

Work in progress
