# VQA

Visual Question Answering in PyTorch


## Setup

- Run `pip install -r requirements.txt` to install all the required python packages.
- Download the VQA (2.0) dataset from [visualqa.org](http://visualqa.org/)

## Preprocess Images

We assume you use image embeddings, which you can process using `preprocess_images.py`.
```shell
python preprocess_images.py <path to instances_train2014.json> \
    --root <path to dataset root "train2014|val2014"> \
    --split <train|val> --arch <vgg|resnet152>
```
I have already pre-processed all the COCO images (both train and test sets) using the VGG-16 and ResNet-152 models. To download them, please go into the `image_embeddings` directory and run `make <model>`.</br>
Here `<model>` can be either `vgg` or `resnet` depending on which model's embeddings you need. E.g. `make vgg`

Alternatively, you can find them [here](https://1drv.ms/f/s!Au18pri6pxSNlop81AhX4bATqy1VJA).


## Running

### Training

To run the training and evaluation code with default values, just type

```
make
```

If you wish to only run the training code, you can run
```
make train
```

If you want to use the raw RGB images from COCO, you can type
```shell
make raw_images
```
This takes the same arguments as `make train`.

You can get a list of options with `make options` or `python main.py -h`.

Check out the `Makefile` to get an idea of how to run the code.

> *NOTE* The code will take care of all the text preprocessing. Just sit back and relax.


The minimum arguments required are:

1. The VQA train annotations dataset
2. The VQA train open-ended questions dataset
3. Path to the COCO training image feature embeddings
4. The VQA val annotations dataset
5. The VQA val open-ended questions dataset
6. Path to the COCO val image feature embeddings

### Evaluation

Evaluating the performance of the model on a fine-grained basis is important. Thus this repo supports evaluating 
answers to questions based on answer type (e.g. "yes/no" questions).

To evaluate the model, run
```shell
make evaluate
```

You are required to pass in the `--resume` argument to point to the trained model weights. The other arguments are 
the same as in training.

### Demo

We have a sample demo that you can run

```shell
make demo
```

You can use your own image or question:

```shell
python demo.py demo_img.jpg "what room is this?"
```


## Results

**NOTE** We train and evaluate on the balanced datasets.

The `DeeperLSTM` model in this repo achieves the following results:

    Overall Accuracy is: 49.15

    Per Answer Type Accuracy is the following:
    other : 38.12
    yes/no : 69.55
    number : 32.17
