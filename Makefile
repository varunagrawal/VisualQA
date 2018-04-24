# Using `pipenv` is highly recommended
# pipenv shell
.RECIPEPREFIX +=

ROOT=~/datasets/VQA2
TRAIN_ANN=${ROOT}/v2_mscoco_train2014_annotations.json
TRAIN_QUES=${ROOT}/v2_OpenEnded_mscoco_train2014_questions.json
TRAIN_IMGS=image_embeddings/coco_train_vgg_fc7.pth

VAL_ANN=${ROOT}/v2_mscoco_val2014_annotations.json
VAL_QUES=${ROOT}/v2_OpenEnded_mscoco_val2014_questions.json
VAL_IMGS=image_embeddings/coco_val_vgg_fc7.pth

ARCH=DeeperLSTM
BATCH=16

main:
    python main.py $(TRAIN_ANN) $(TRAIN_QUES) $(TRAIN_IMGS) $(VAL_ANN) $(VAL_QUES) $(VAL_IMGS) \
    --arch $(ARCH) --batch_size ${BATCH}

options:
    python main.py -h

demo:
    python demo.py demo_img.jpg "what room is this?" $(TRAIN_QUES) $(TRAIN_ANN)
