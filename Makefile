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

IMAGE_ROOT=~/datasets/coco
ARCH=DeeperLSTM
BATCH=16

main:
    python main.py $(TRAIN_ANN) $(TRAIN_QUES) $(VAL_ANN) $(VAL_QUES) --images $(TRAIN_IMGS) --val_images $(VAL_IMGS) \
    --arch $(ARCH) --batch_size ${BATCH}

raw_images:
    python main.py $(TRAIN_ANN) $(TRAIN_QUES) $(VAL_ANN) $(VAL_QUES) \
    --raw_images --image_root $(IMAGE_ROOT) --arch $(ARCH) --batch_size ${BATCH}

options:
    python main.py -h

evaluate:
    python evaluate.py ~/datasets/VQA2/v2_mscoco_train2014_annotations.json ~/datasets/VQA2/v2_OpenEnded_mscoco_train2014_questions.json --images image_embeddings/coco_train_vgg_fc7.pth ~/datasets/VQA2/v2_mscoco_val2014_annotations.json ~/datasets/VQA2/v2_OpenEnded_mscoco_val2014_questions.json --val_images image_embeddings/coco_val_vgg_fc7.pth --resume weights/vqa_checkpoint_DeeperLSTM_149.pth --batch_size 64

demo:
    python demo.py demo_img.jpg "what room is this?" $(TRAIN_QUES) $(TRAIN_ANN)
