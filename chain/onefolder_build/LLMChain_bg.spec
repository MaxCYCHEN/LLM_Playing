# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import copy_metadata

datas = []
datas += copy_metadata('urllib3')

a = Analysis(
    ['LLMChain_bg.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['tiktoken_ext.openai_public', 'tiktoken_ext', 'langchain.docstore.in_memory', 'langchain.schema.document'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['pdf2image', 'pdfminer.six', 'onnxruntime', 'onnx', 'cryptography', 'scipy', 'safetensors', 'tokenizers', 'jupyter', 'transformers'],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='LLMChain_bg',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='LLMChain_bg',
)
