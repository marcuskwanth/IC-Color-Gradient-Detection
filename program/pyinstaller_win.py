# Run only on Windows!

import PyInstaller.__main__

PyInstaller.__main__.run([
    'mainprogram.py',
    '--windowed',
])