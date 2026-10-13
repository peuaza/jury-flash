# -*- mode: python ; coding: utf-8 -*-
a = Analysis(['jury_cli.py'],
    pathex=['.'],
    binaries=[],
    datas=[('_det_chat_np.pkl','.'), ('_det_applied_np.pkl','.'), ('_det_classic_np.pkl','.'), ('_det_lit_np.pkl','.'), ('_det_multi14.pkl','.'), ('_det_narr_np.pkl','.'), ('_det_flash2_np.pkl','.'), ('_det_struct_np.pkl','.'), ('_det_multi_np.pkl','.'), ('_det_rewrite_np.pkl','.'), ('_det_tok_np.pkl','.'), ('_det_expo_np.pkl','.'), ('_det_narrative_np.pkl','.'), ('_det_info_np.pkl','.'), ('_det_auto_np.pkl','.'), ('_det_auto_c_np.pkl','.'), ('_det_auto_p_np.pkl','.'), ('_finetune_replay.npz','.'), ('_ngram_ref.pkl','.'), ('tokenizer.json','.'), ('tokenizer_config.json','.'), ('_tok_lm.npz','.'), ('_train_bank.npz','.'), ('jury_learn.py','.'), ('jury_ft_cli.py','.'), ('_ref145.pkl','.'), ('jury_diag.py','.'), ('jury_research.py','.'), ('jury_cmds2.py','.')],
    hiddenimports=['structure_feats','infoflow','ngramfeat','jury_det','toklenfeat','jury_prep','cpu_proxy2','toklm','jury_finetune','jury_ft_cli','jury_conf','jury_session','tokenizers'],
    hookspath=[], runtime_hooks=[],
    excludes=['torch','transformers','sklearn','scipy','matplotlib','pandas','PIL','tkinter',
              'IPython','jupyter','notebook','pytest','setuptools','pip'],
    noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, [],
    name='jury', debug=False, bootloader_ignore_signals=False,
    strip=False, upx=True, console=True, disable_windowed_traceback=False,
    target_arch=None, codesign_identity=None, entitlements_file=None)
