from style_transfer import Experiment, config_logger
from style_transfer.experiment import CONTENT_WEIGHT, STYLE_WEIGHT, CONTENT_LAYERS

import matplotlib.pyplot as plt
from pathlib import Path


logger = config_logger(Path('output'))

import cv2
folder = Path('datasets')
folder.mkdir(exist_ok=True)

for img_name in [r'content\turtle', r'style\kanagawa']:
    filename = Path(folder, 'style_transfer', f'{img_name}.jpg')
    img = cv2.imread(str(filename))
    for k in [5,11,19,43,101,151,211,401]:
        logger.info(k)
        blur = cv2.GaussianBlur(img,(k,k),0)
        filename = Path(folder, 'blur', f'{img_name}_blur{k}.jpg')
        filename.parent.mkdir(exist_ok=True, parents=True)
        assert cv2.imwrite(str(filename),blur)

# folder = Path('results', 'parameters_experiments')
# r = Experiment(folder)

# logger.info(f'{r.options(STYLE_WEIGHT, "turtle", "kanagawa") = }')
# logger.info(f'{r.options(CONTENT_LAYERS, "turtle", "kanagawa") = }')

# logger.info(f'{r.output_folder("turtle", "kanagawa") = }')

# im = r.image("turtle", "kanagawa").show()

# fig = r.loss_plot("turtle", "kanagawa")
# fig.set_yscale('log')
# plt.show()