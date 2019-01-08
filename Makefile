# Using `pipenv` is highly recommended
# pipenv shell
.RECIPEPREFIX +=

ROOT=~/datasets/VQA2
TRAIN_ANN=${ROOT}/v2_mscoco_train2014_annotations.json
TRAIN_QUES=${ROOT}/v2_OpenEnded_mscoco_train2014_questions.json
TRAIN_IMGS=image_embeddings/coco_train_vgg16_fc7.pth

VAL_ANN=${ROOT}/v2_mscoco_val2014_annotations.json
VAL_QUES=${ROOT}/v2_OpenEnded_mscoco_val2014_questions.json
VAL_IMGS=image_embeddings/coco_val_vgg_fc7.pth

IMAGE_ROOT=~/datasets/coco
ARCH=DeeperLSTM
BATCH=512
WORKERS=8

CHECKPOINT=weights/vqa_checkpoint_DeeperLSTM_199.pth

main: train evaluate

train:
    python main.py $(TRAIN_ANN) $(TRAIN_QUES) $(VAL_ANN) $(VAL_QUES) --images $(TRAIN_IMGS) --val_images $(VAL_IMGS) \
    --arch $(ARCH) --batch_size ${BATCH} --num_workers ${WORKERS} --image_root $(IMAGE_ROOT)

raw_images:
    python main.py $(TRAIN_ANN) $(TRAIN_QUES) $(VAL_ANN) $(VAL_QUES) \
    --raw_images --image_root $(IMAGE_ROOT) --arch $(ARCH) --batch_size 32

options:
    python main.py -h

evaluate:
    python evaluate.py $(TRAIN_ANN) $(TRAIN_QUES) $(VAL_ANN) $(VAL_QUES) --images $(TRAIN_IMGS) --val_images $(VAL_IMGS) \
      --batch_size 1 --resume $(CHECKPOINT) --num_workers ${WORKERS}

demo:
    python demo.py demo_img.jpg "what room is this?" $(TRAIN_QUES) $(TRAIN_ANN) --checkpoint $(CHECKPOINT)

preprocess:
    python preprocess_images.py $(IMAGE_ROOT)/annotations/instances_train2014.json --root $(IMAGE_ROOT) --split $(split)
