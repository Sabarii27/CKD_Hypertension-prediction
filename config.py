BASE_DIR = '.'
DATA_DIR = f'{BASE_DIR}/data'
DRIVE_DIR = f'{DATA_DIR}/DRIVE'
STARE_DIR = f'{DATA_DIR}/STARE'
MODELS_DIR = f'{BASE_DIR}/models'
RESULTS_DIR = f'{BASE_DIR}/results'

import os
for d in [DATA_DIR, DRIVE_DIR, STARE_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

IMG_HEIGHT = 512
IMG_WIDTH = 512