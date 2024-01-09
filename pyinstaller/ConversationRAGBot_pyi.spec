# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files

datas = []
datas += collect_data_files('gradio_client')
datas += collect_data_files('gradio')


a = Analysis(
    ['ConversationRAGBot_pyi.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['tiktoken_ext.openai_public', 'tiktoken_ext', 'langchain.docstore.in_memory', 'langchain.schema.document'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
	module_collection_mode={
        'gradio': 'py',  # Collect gradio package as source .py files
    },
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ConversationRAGBot_pyi',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
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
    upx=False,
    upx_exclude=[],
    name='ConversationRAGBot_pyi',
)
