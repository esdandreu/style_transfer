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

PARAMETERS = [
    CONTENT,
    STYLE,
    CONTENT_LAYERS,
    STYLE_LAYERS,
    PRE_TRAINING,
    LEARNING_RATE,
    BETA_1,
    BETA_2,
    EPSILON,
    AMSGRAD,
    CONTENT_WEIGHT,
    STYLE_WEIGHT,
    NUM_ITERATIONS,
]

def str2bool(string: str) -> bool:
    if string == "True":
        return True
    elif string == "False":
        return False
    raise ValueError(f'Could not convert string "{string}" into bool')

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
    # _options = {}

    def __init__(self, folder: Union[Path, str]):
        self.folder = Path(folder)
        self.clean_data()

    # Build path
    def output_folder(
        self,
        content_path: str, 
        style_path: str,
        content_layers: Union[List[str],str] = ['block5_conv2'],
        style_layers: Union[List[str],str] = '5_B12345_L11111',
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
                value = kwargs.get(parameter,None)
                if value:
                    f = folders.get(value, None)
                    if f:
                        return self._output_folder(folder=f, **kwargs) 
                    raise ValueError(
                        f'Value = {value} of parameter = {parameter} not found'
                        ' in experiment. Available options: '
                        f'{[x for x in folders.keys()]}'
                        )
                raise ValueError(
                    'Found parameter not in parameter list '
                    f'{[p for p in kwargs.keys()]}'
                    )
            else:
                raise RuntimeError('Experiment folder is not clean')
        return folder

    def options(
        self,
        parameter: str,
        content_path: str, 
        style_path: str,
        content_layers: Union[List[str],str] = ['block5_conv2'],
        style_layers: Union[List[str],str] = '5_B12345_L11111',
        pre_training: bool = True,
        learning_rate: float = 5,
        beta_1: float = 0.99,
        beta_2: float = 0.999,
        epsilon: float = 1e-07,
        amsgrad: bool = False,
        content_weight: float = 1e3, 
        style_weight: float = 1e-2,
        num_iterations: int = 1000,
        ) -> List[Union[str,bool,float,int]]:
        kwargs = {}
        for key, value in locals().items():
            if key in ['self', 'parameter', parameter]:
                continue
            elif (
                key in [CONTENT_LAYERS, STYLE_LAYERS] 
                and isinstance(value, list)
                ):
                value = layers_codename(value)
            kwargs[key], _ = value2str(value)
        options = self._options(
            folder=self.folder, 
            parameter=parameter,
            **kwargs
            )
        for fun in [int, float, str2bool]:
            try:
                options = [fun(x) for x in options]
                break
            except ValueError:
                continue
        return options

    def _options(self, folder: Path, parameter: str, **kwargs) -> List[str]:
        f_parameter = False
        folders = {}
        for x in folder.iterdir():
            if x.is_dir():
                folders[x.name] = x
            elif not f_parameter and not x.suffix:
                f_parameter = x.stem
        if f_parameter:
            if folders:
                if f_parameter == parameter:
                    out = []
                    for value, f in folders.items():
                        try:
                            # Check if the folder finishes in a valid input
                            self._output_folder(
                                folder=f, 
                                **{parameter: value},
                                **kwargs
                                )
                            out.append(value)
                        except ValueError:
                            continue
                    return out
                else:
                    f_parameter
                    value = kwargs.get(f_parameter,None)
                    if value:
                        logger.info(folders)
                        f = folders.get(value, None)
                        if f:
                            return self._options(folder=f,
                                parameter=parameter,
                                **kwargs
                                ) 
                        raise ValueError(
                            f'Value = {value} of f_parameter = {f_parameter} '
                            'not found in experiment. Available options: '
                            f'{[x for x in folders.keys()]}'
                            )
                    raise ValueError(
                        'Found parameter not in f_parameter list '
                        f'{[p for p in kwargs.keys()]}'
                        )
        raise RuntimeError('Experiment folder is not clean')

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
            logger.info(f'Removed folder = {folder}')
        return remove

    def loss_plot(
        self,
        content_path: str, 
        style_path: str,
        content_layers: Union[List[str],str] = ['block5_conv2'],
        style_layers: Union[List[str],str] = '5_B12345_L11111',
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
        style_layers: Union[List[str],str] = '5_B12345_L11111',
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
        
