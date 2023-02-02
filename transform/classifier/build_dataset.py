import os
import pathlib
import random
import shutil

from imutils import paths

import config

image_paths = list(paths.list_images(config.ORIG_INPUT_DATASET))
random.seed(42)
random.shuffle(image_paths)

i = int(len(image_paths) * config.TRAIN_SPLIT)
train_paths = image_paths[:i]
test_paths = image_paths[i:]

j = int(len(train_paths) * config.VAL_SPLIT)
val_paths = train_paths[:j]
train_paths = train_paths[j:]

datasets = [
    ("training", train_paths, config.TRAIN_PATH),
    ("validation", val_paths, config.VAL_PATH),
    ("testing", test_paths, config.TEST_PATH),
]
corrupted_file = []
for (dType, image_paths, base_output) in datasets:
    print(f"[INFO] building {dType} split")

    if not pathlib.Path.exists(base_output):
        print(f"[INFO] 'creating {base_output}' directory")
        pathlib.Path.mkdir(base_output, parents=True, exist_ok=True)

    for input_path in image_paths:
        filename = input_path.split(os.path.sep)[-1]
        label = filename[-5:-4]

        label_path = pathlib.Path(base_output).joinpath(label)

        if not pathlib.Path.exists(label_path):
            print(f"[INFO] creating {label_path}")
            pathlib.Path.mkdir(label_path, exist_ok=True)

        p = pathlib.Path(label_path).joinpath(filename)
        shutil.copy2(input_path, str(p))
        corrupted_file = []
        try:
            from PIL import Image

            v_image = Image.open(str(p))
            v_image.verify()
        except IOError:
            print(f"[error] while creating {str(p)}")
            corrupted_file.append(p)
            continue
        except Exception:
            print(f"general exception  while creating {str(p)}")
            corrupted_file.append(p)
            continue

for file in corrupted_file:
    file.unlink()
