SAMPLING_RATIO=0.4

EPOCHS=10
BATCH_SIZE=200

IMAGE_SIZE=(224,224)

IMG_PATH="../dataset/after_4_bis/*/*.jpg"
EMBED_IMG_PATH="../dataset/reference_images/*.jpg"
LOG_DIR="logdir_train"

DEVICE="mps"
MODEL_SAVEPATH="./modelsave"