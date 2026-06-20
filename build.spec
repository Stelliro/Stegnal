# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

# 1. Setup Source Data
datas = [('src/umbra', 'umbra')]
binaries = []

# 2. Force Include Missing Modules (The Fix)
hiddenimports = [
    'scipy.signal', 
    'scipy.io', 
    'scipy.io.wavfile',
    'scipy.special.cython_special', 
    'umbra.encoding',
    'umbra.decoding',
    'umbra.audio'
]

# 3. Collect CustomTkinter assets
tmp_ret = collect_all('customtkinter')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # 4. Exclude Heavy Libraries to keep size small
    excludes=[
        'torch', 'torchvision', 'torchaudio', 
        'cupy', 'cupyx', 
        'cv2', 'opencv-python', 
        'matplotlib', 'pandas', 
        'PyQt5', 'PyQt6', 'PySide6', 
        'IPython', 'notebook', 'tkinter.test',
        'nvidia', 'nvidia-cuda-runtime-cu11'
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='Umbra_Terminal',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)