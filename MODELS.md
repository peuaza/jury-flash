# 模型文件（不随 git 提交，走 GitHub Release）

运行检测器需要以下文件放在**仓库根目录**（与 `jury_cli.py` 同层）：

| 文件 | 大小 | 说明 |
|---|---|---|
| `tokenizer.json` | ~19 MB | BPE 分词器（来自千问 Qwen 系，Apache-2.0；见 `NOTICE`） |
| `tokenizer_config.json` | 小 | 同上 |
| `_tok_lm.npz` | ~9.6 MB | token n-gram 语言模型（本项目统计·人类诗语料） |
| `_ngram_ref.pkl` | 小 | n-gram 参考分布（供 ngram 特征） |
| `_train_bank.npz` | ~10.6 MB | 训练特征库（供 `jury learn --fit` 重训） |
| `_det_multi14.pkl` | ~0.5 MB | 主模型（14 类 MLP 集成） |
| `_det_*.pkl` | 小 | 各体裁辅助头 |
| `_ref145.pkl` | 小 | 特征参考分布（供 `analysis`） |

## 获取方式（二选一）

**A. 自动下载**

```bash
python tools/download_models.py --repo <用户名>/jury-flash --tag v1.0.0
```

**B. 手动**：从 Release 页面下载 `models-v1.0.0.zip`，解压到仓库根目录。

## 或者：直接用预编译的 `jury.exe`

Release 里的 `jury.exe` **已内置全部模型**，无需下载。
