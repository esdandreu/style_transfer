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

run_style_transfer(
    'turtle', 'kanagawa', 
    output_folder=run_folder, 
    num_iterations=600
    )
