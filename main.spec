# main.spec

# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main.py'],  # your real entry point
    pathex=[],
    binaries=[],
    datas=[
        ('assets/app.db', 'assets'),
        ('assets/user_data.json', 'assets'),
        ('Models/*.pt', 'Models'),
        ('Models/*.h5', 'Models'),
        ('Models/*.dat', 'Models'),
        ('ui/*.py', 'ui'),
        ('core/*.py', 'core'),
        ('database/*.py', 'database'),
        ('utils/*.py', 'utils'),
    ],
    hiddenimports=[
        'unittest',
        'unittest.mock',
        'torch',
        'torch.ao.quantization',
        'torch.fx.passes.shape_prop',
        'torch.fx.passes.infra.pass_base',
        'customtkinter',
        'tkinter',
        'sqlite3',
        'cv2',
        'numpy',
        'tensorflow',
        'torch',
        'torchvision',
        'mediapipe',
        'ultralytics',
        'dlib',
        'PIL',
        'matplotlib',
        'pandas',
        'scipy',
        'psutil',
        'pystray',
        'win10toast',
        'win11toast',
        'sounddevice',
        'requests',
        'json',
        'os',
        'sys',
        'threading',
        'time',
        'datetime',
        'logging',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        'matplotlib.tests',
        'numpy.tests',
        'PIL.tests',
        'tkinter.test',
        'unittest',
        'test',
        'tests',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FacialEmotionRecognition',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # no terminal popup
    # icon='assets/icon.ico',  # Add an icon if you have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='FacialEmotionRecognition'
)
