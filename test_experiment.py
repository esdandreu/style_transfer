from style_transfer import Experiment, config_logger
from style_transfer.experiment import CONTENT_WEIGHT, STYLE_WEIGHT, CONTENT_LAYERS

import matplotlib.pyplot as plt
from pathlib import Path


logger = config_logger(Path('output'))

# import cv2
# folder = Path('datasets')
# folder.mkdir(exist_ok=True)

# for img_name in [r'content\turtle', r'style\kanagawa']:
#     filename = Path(folder, 'style_transfer', f'{img_name}.jpg')
#     img = cv2.imread(str(filename))
#     for w,h in [(1280, 960), (640,480), (320,240), (160, 120), (80,60)]:
#         logger.info((w,h))
#         blur = cv2.resize(img,(w,h))
#         filename = Path(folder, 'resolution', f'{img_name}_res{w}x{h}.jpg')
#         filename.parent.mkdir(exist_ok=True, parents=True)
#         assert cv2.imwrite(str(filename),blur)

folder = Path('results', 'parameters_experiments')
r = Experiment(folder)

logger.info(f'{r.options(STYLE_WEIGHT, "turtle", "kanagawa") = }')
logger.info(f'{r.options(CONTENT_LAYERS, "turtle", "kanagawa") = }')

logger.info(f'{r.output_folder("turtle", "kanagawa") = }')

im = r.image("turtle", "kanagawa").show()

fig = r.loss_plot("turtle", "kanagawa")
fig.set_yscale('log')
plt.show()