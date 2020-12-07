from pathlib import Path
import os 

MODULE_FOLDER = Path(os.path.dirname(os.path.realpath(__file__)))
ROOT_FOLDER = MODULE_FOLDER.parent
OUTPUT_FOLDER = Path(ROOT_FOLDER, 'output')
OUTPUT_FOLDER.mkdir(parents=True,exist_ok=True)

CHECKPOINTS_PER_RUN = 10

img_dir = Path(ROOT_FOLDER, 'datasets', 'style_transfer')
if not img_dir.exists() or not img_dir.is_dir():
    raise FileNotFoundError('The image directory does not exist')

STYLE_FOLDER = Path(img_dir, 'style')
if not STYLE_FOLDER.exists() or not STYLE_FOLDER.is_dir():
    raise FileNotFoundError('The style directory does not exist')

CONTENT_FOLDER = Path(img_dir, 'content')
if not CONTENT_FOLDER.exists() or not CONTENT_FOLDER.is_dir():
    raise FileNotFoundError('The content directory does not exist')