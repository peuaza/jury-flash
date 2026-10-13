# -*- coding: utf-8 -*-
"""pack_models.py — 把根目录的模型文件打包成 models-v1.0.0.zip（上传到 Release）

在**包含模型文件的目录**下运行：
    python tools/pack_models.py
"""
import io, os, sys, zipfile

MODELS = ["_det_multi14.pkl", "_det_lit_np.pkl", "_det_expo_np.pkl", "_det_narr_np.pkl",
          "_det_info_np.pkl", "_det_classic_np.pkl", "_det_applied_np.pkl",
          "_det_chat_np.pkl", "_det_auto_c_np.pkl", "_det_auto_p_np.pkl",
          "_tok_lm.npz", "_ngram_ref.pkl", "tokenizer.json", "tokenizer_config.json",
          "_train_bank.npz", "_ref145.pkl"]
OUT = "models-v1.0.0.zip"


def main():
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        n = 0
        for f in MODELS:
            if os.path.exists(f):
                z.write(f, f)
                n += 1
                print("  + %s" % f)
            else:
                print("  - 缺少 %s（跳过）" % f)
    print("  => %s（%d 个文件，%.1f MB）" % (OUT, n, os.path.getsize(OUT) / 1048576))
    return 0


if __name__ == "__main__":
    sys.exit(main())
