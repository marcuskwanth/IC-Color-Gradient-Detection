# Run only on macOS!

import PyInstaller.__main__

PyInstaller.__main__.run([
    'mainprogram.py',
    '--windowed',
])