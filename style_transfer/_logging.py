import logging
import __main__
import os
from pathlib import Path
from typing import Union
from datetime import datetime

logger = logging.getLogger(__name__)

def config_logger(
    output_folder: Union[Path, str],
    verbose: bool = False, 
    ):
    logger = logging.getLogger('style_transfer')
    logger.setLevel(logging.INFO if not verbose else logging.DEBUG)
    # STREAM
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter('%(name)s [%(levelname)s]: %(message)s')
        )
    logger.addHandler(console_handler)
    # FILE
    filename = f"{datetime.now().strftime('%Y%m%dT%H%M%S')}.log"
    output_folder = Path(output_folder, 'logs')
    output_folder.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(
        filename=Path(output_folder, filename), encoding='utf-8'
        )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter('%(name)s [%(levelname)s] %(asctime)s: %(message)s')
        )
    logger.addHandler(file_handler)
    return logger

def stats_logger(
    path: Union[Path, str],
    ):
    logger = logging.getLogger('stats')
    logger.setLevel(logging.INFO)
    # STREAM
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter('%(name)s (%(asctime)s): %(message)s')
        )
    logger.addHandler(console_handler)
    # FILE
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(
        filename=path, encoding='utf-8'
        )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter('%(message)s')
        )
    logger.addHandler(file_handler)
    return logger