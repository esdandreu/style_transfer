from style_transfer import Experiment, config_logger
from style_transfer.experiment import CONTENT_WEIGHT, STYLE_WEIGHT, CONTENT_LAYERS

import matplotlib.pyplot as plt
from pathlib import Path


logger = config_logger(Path('output'))

folder = Path('results', '20201221_105909 - Copy')
r = Experiment(folder)

# logger.info(f'{r.options(STYLE_WEIGHT, "turtle", "kanagawa") = }')
# logger.info(f'{r.options(CONTENT_LAYERS, "turtle", "kanagawa") = }')

# logger.info(f'{r.output_folder("turtle", "kanagawa") = }')

# im = r.image("turtle", "kanagawa").show()

# fig = r.loss_plot("turtle", "kanagawa")
# fig.set_yscale('log')
# plt.show()