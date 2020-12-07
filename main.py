from pathlib import Path
from datetime import datetime

from style_transfer import run_style_transfer
from style_transfer.config import STYLE_FOLDER, CONTENT_FOLDER, OUTPUT_FOLDER
from style_transfer._logging import config_logger

# Create output folder
run_folder = Path(OUTPUT_FOLDER, datetime.now().strftime('%Y%m%d_%H%M%S'))
run_folder.mkdir(parents=True,exist_ok=True)

logger = config_logger(output_folder=OUTPUT_FOLDER)

logger.info('Start run')

for content_path in CONTENT_FOLDER.iterdir():
    if not content_path.is_file():
        continue
    for style_path in STYLE_FOLDER.iterdir():
        if not style_path.is_file():
            continue
        logger.info(
            'Start style transfer:'
            f'\ncontent:{content_path}\nstyle:{style_path}'
            )
        run_style_transfer(
            content_path=content_path,
            style_path=style_path,
            output_folder=run_folder, 
            num_iterations=1000,
            content_weight=1e3, 
            style_weight=1e-2,
            verbose=True,
            )
            
logger.info('Finished run')
