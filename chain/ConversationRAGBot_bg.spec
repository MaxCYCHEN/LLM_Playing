# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['ConversationRAGBot_bg.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['tiktoken_ext.openai_public', 'tiktoken_ext', 'langchain.docstore.in_memory', 'langchain.schema.document'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ConversationRAGBot_bg',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
