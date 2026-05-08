# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for sports recognition pipeline (onedir mode)."""

import os

block_cipher = None

# Project root (where this spec file lives)
PROJECT_DIR = SPECPATH

# Model files and config to bundle into saved_models/
saved_models_files = [
    'combined_model_3s_seed42.pth',
    'combined_model_3s_seed123.pth',
    'combined_model_3s_seed456.pth',
    'combined_model_5s_seed42.pth',
    'combined_model_5s_seed123.pth',
    'combined_model_5s_seed456.pth',
    'combined_model_8s_seed42.pth',
    'combined_model_8s_seed123.pth',
    'combined_model_8s_seed456.pth',
    'combined_model_best.pth',
    'norm_params_3s.pkl',
    'norm_params_5s.pkl',
    'norm_params_8s.pkl',
    'norm_params.pkl',
    'ensemble_config.json',
]

datas = []
for f in saved_models_files:
    src = os.path.join(PROJECT_DIR, 'saved_models', f)
    if os.path.exists(src):
        datas.append((src, 'saved_models'))
    else:
        print(f"WARNING: {src} not found, skipping")

a = Analysis(
    [os.path.join(PROJECT_DIR, 'main.py')],
    pathex=[PROJECT_DIR, os.path.join(PROJECT_DIR, 'src')],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'openpyxl',
        'scipy.signal',
        'scipy.ndimage',
        'scipy.interpolate',
        'imu_activity_pipeline.input',
        'imu_activity_pipeline.output',
        'imu_activity_pipeline.inference',
        'imu_activity_pipeline.config',
        'imu_activity_pipeline.data_utils',
        'imu_activity_pipeline.models_nn',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'torchvision',
        'torchaudio',
        'IPython',
        'jupyter',
        'notebook',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main',
)
