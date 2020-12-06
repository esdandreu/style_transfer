from pathlib import Path

img_dir = Path('datasets', 'tutorial')
if not img_dir.exists() or not img_dir.is_dir():
    raise FileNotFoundError('The image directory does not exist')
else:
    print('\n'.join([str(f) for f in img_dir.iterdir() if f.is_file()]))