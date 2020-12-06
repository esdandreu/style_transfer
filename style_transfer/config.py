from pathlib import Path

img_dir = Path('datasets', 'style_transfer')
if not img_dir.exists() or not img_dir.is_dir():
    raise FileNotFoundError('The image directory does not exist')