from typing import List, Union, Optional
from pathlib import Path
from PIL import Image

import logging
import pandas as pd

from style_transfer.utils import layers_codename

logger = logging.getLogger(__name__)

CONTENT = 'content_path'
STYLE = 'style_path'
CONTENT_LAYERS = 'content_layers'
STYLE_LAYERS = 'style_layers'
PRE_TRAINING = 'pre_training'
LEARNING_RATE = 'learning_rate'
BETA_1 = 'beta_1'
BETA_2 = 'beta_2'
EPSILON = 'epsilon'
AMSGRAD = 'amsgrad'
CONTENT_WEIGHT = 'content_weight'
STYLE_WEIGHT = 'style_weight'
NUM_ITERATIONS = 'num_iterations'

def str2bool(string: str) -> bool:
    if string == "True":
        return True
    elif string == "False":
        return False
    raise ValueError(f'Could not convert into bool, {string = }')

def value2str(value):
    if isinstance(value, tuple):
        value = value[1]
        string = value[0]
    elif isinstance(value, Path):
        string = value.stem
    elif isinstance(value, float):
        string = f'{value:.4e}'
    else:
        string = str(value)
    return string, value

class Experiment:
    _options = {}

    def __init__(self, folder: Union[Path, str]):
        self.folder = Path(folder)
        self.clean_data()

    # Build path
    def output_folder(
        self,
        content_path: str, 
        style_path: str,
        content_layers: Union[List[str],str] = ['block5_conv2'],
        style_layers: Union[List[str],str] = [
            'block1_conv1',
            'block2_conv1',
            'block3_conv1', 
            'block4_conv1', 
            'block5_conv1'
            ],
        pre_training: bool = True,
        learning_rate: float = 5,
        beta_1: float = 0.99,
        beta_2: float = 0.999,
        epsilon: float = 1e-07,
        amsgrad: bool = False,
        content_weight: float = 1e3, 
        style_weight: float = 1e-2,
        num_iterations: int = 1000,
        ) -> Path:
        kwargs = {}
        for key, value in locals().items():
            if key == 'self':
                continue
            elif (
                key in [CONTENT_LAYERS, STYLE_LAYERS] 
                and isinstance(value, list)
                ):
                value = layers_codename(value)
            kwargs[key], _ = value2str(value)
        return self._output_folder(folder=self.folder, **kwargs)

    def _output_folder(self, folder: Path, **kwargs) -> Path:
        parameter = False
        folders = {}
        for x in folder.iterdir():
            if x.is_dir():
                folders[x.name] = x
            elif not parameter and not x.suffix:
                parameter = x.stem
        if parameter:
            if folders:
                if value:=kwargs.get(parameter,None):
                    if f:=folders.get(value, None):
                        return self._output_folder(folder=f, **kwargs) 
                    raise ValueError(
                        f'{value = } of {parameter = } not found in experiment'
                        f'. Available options: {[x for x in folders.keys()]}'
                        )
                raise ValueError(
                    'Found parameter not in parameter list '
                    f'{[p for p in kwargs.keys()]}'
                    )
            else:
                raise RuntimeError('Experiment folder is not clean')
        return folder

    def options(self, parameter: str) -> List[Union[str,bool,float,int]]:
        if (options:=self._options.get(parameter, None)) is None:
            options = self._find_options(self.folder, parameter)
            for fun in [int, float, str2bool]:
                try:
                    options = [fun(x) for x in options]
                    break
                except ValueError:
                    continue
            self._options[parameter] = options
        return options

    def _find_options(self, folder: Path, parameter: str) -> List[str]:
        folders = []
        found_parameter = False
        for x in folder.iterdir():
            if x.is_dir() and x.stem != 'logs':
                folders.append(x)
            elif not found_parameter and not x.suffix:
                if x.stem == parameter:
                    found_parameter = True
        if found_parameter:
            return [x.name for x in folders]
        else:
            try:
                return self._find_options(folders[0], parameter)
            except IndexError:
                raise ValueError(f'Could not find options for {parameter = }')

    def clean_data(self):
        if not self.folder.is_dir():
            raise ValueError(f'{self.folder} is not a valid experiment folder')
        return self._clean_folder(self.folder)

    def _clean_folder(self, folder: Path):
        remove = True
        removefiles = [] # Files to remove without extension (Parameter names)
        for x in folder.iterdir():
            if x.is_dir():
                remove = self._clean_folder(folder=x) and remove
            elif x.suffix:
                remove = False
            else:
                removefiles.append(x)
        if remove:
            [x.unlink() for x in removefiles] # Clean useless files
            folder.rmdir() # Folder should now be empty, we can remove it
            logger.info(f'Removed {folder = }')
        return remove

    def loss_plot(
        self,
        content_path: str, 
        style_path: str,
        content_layers: Union[List[str],str] = ['block5_conv2'],
        style_layers: Union[List[str],str] = [
            'block1_conv1',
            'block2_conv1',
            'block3_conv1', 
            'block4_conv1', 
            'block5_conv1'
            ],
        pre_training: bool = True,
        learning_rate: float = 5,
        beta_1: float = 0.99,
        beta_2: float = 0.999,
        epsilon: float = 1e-07,
        amsgrad: bool = False,
        content_weight: float = 1e3, 
        style_weight: float = 1e-2,
        num_iterations: int = 1000,
        ):
        folder = self.output_folder(
            **{k: v for k, v in locals().items() if k != 'self'}
            )
        filename = f'{content_path}_{style_path}.csv'
        return self._loss_plot(csv_file=Path(folder, filename))

    def _loss_plot(self, csv_file: Path):
        return pd.read_csv(csv_file,
            usecols=['loss', 'style_loss', 'content_loss']
            ).plot()

    def image(
        self,
        content_path: str, 
        style_path: str,
        iterations: Optional[int] = None,
        content_layers: Union[List[str],str] = ['block5_conv2'],
        style_layers: Union[List[str],str] = [
            'block1_conv1',
            'block2_conv1',
            'block3_conv1', 
            'block4_conv1', 
            'block5_conv1'
            ],
        pre_training: bool = True,
        learning_rate: float = 5,
        beta_1: float = 0.99,
        beta_2: float = 0.999,
        epsilon: float = 1e-07,
        amsgrad: bool = False,
        content_weight: float = 1e3, 
        style_weight: float = 1e-2,
        num_iterations: int = 1000,
        ):
        folder = self.output_folder(**{
            k: v for k, v in locals().items() 
            if k not in ['self', 'iterations']
            })
        filename = (
            f'{content_path}_{style_path}'
            f'{"_"+str(iterations) if iterations is not None else ""}.png'
            )
        return Image.open(Path(folder, filename))
        
