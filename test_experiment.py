from style_transfer import Experiment, config_logger
from style_transfer.experiment import CONTENT_WEIGHT

import matplotlib.pyplot as plt
from pathlib import Path


logger = config_logger(Path('output'))

folder = Path('results', 'turtle_kanagawa_weights')
r = Experiment(folder)

logger.info(f'{r.options(CONTENT_WEIGHT) = }')

logger.info(f'{r.output_folder("turtle", "kanagawa") = }')

im = r.image("turtle", "kanagawa").show()

fig = r.loss_plot("turtle", "kanagawa")
fig.set_yscale('log')
plt.show()