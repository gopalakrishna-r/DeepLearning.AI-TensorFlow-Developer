from pathlib import Path

ORIG_INPUT_DATASET = Path.cwd().parent.joinpath("dataset/orig")
BASE_PATH = "dataset/idc"

TRAIN_PATH = Path.cwd().parent.joinpath(BASE_PATH).joinpath("training")
TEST_PATH = Path.cwd().parent.joinpath(BASE_PATH).joinpath("testing")
VAL_PATH = Path.cwd().parent.joinpath(BASE_PATH).joinpath("validation")

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
IMAGE_SIZE = (48, 48)

NUM_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 5
INIT_LR = 1e-3
BS = 48
