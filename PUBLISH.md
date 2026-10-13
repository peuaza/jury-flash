# 发布指南（把 jury-flash 开源到 GitHub）

> 本文给出**从零到发布**的完整步骤。假设你已注册 GitHub。

---

## 0. 先确认三件事

```bash
cd jury-flash

# ① 绝不提交密钥 —— 确认没有被 git 跟踪
ls -la | grep -i apikey          # 应当**没有**任何 .apikey 文件

# ② 确认 .gitignore 生效
git init
git status --short | head -20    # 不应出现 *.pkl / *.npz / tokenizer.json / .apikey*
```

**若看到 `tokenizer.json` 或 `*.pkl` 出现在待提交列表 ⇒ `.gitignore` 未生效，务必先修。**

---

## 1. 初始化本地仓库

```bash
cd jury-flash
git init
git branch -M main

git add .
git commit -m "jury-flash-CN v1.0.0：中文 AI 文本检测器（纯 CPU，离线）"
```

**检查提交内容**：

```bash
git show --stat HEAD | tail -30
```

应当只包含 `.py / .md / .txt / .gitignore / jury.spec`。

---

## 2. 在 GitHub 建仓库

1. 打开 https://github.com/new
2. **Repository name**：`jury-flash`（或 `jury-flash-CN`）
3. **Description**：中文 AI 文本检测器 · 纯 CPU · 离线 · 7 体裁 × 人机
4. **Public**
5. ⚠ **不要**勾选 "Add a README"（本地已有）
6. 点 **Create repository**

---

## 3. 推上去

```bash
# 把 <你的用户名> 换掉
git remote add origin https://github.com/<你的用户名>/jury-flash.git
git push -u origin main
```

若提示验证：用 **Personal Access Token** 当密码
（GitHub → Settings → Developer settings → Tokens (classic) → 勾 `repo`）。

---

## 4. 打包模型（上传到 Release，不进 git）

```bash
# 回到含模型文件的目录（intern_decision/）
python tools/pack_models.py
# 产出 models-v1.0.0.zip（约 55 MB）
```

若在 `intern_decision/` 下没有 `tools/`，可直接用：

```bash
python -c "import zipfile,os;M=['_det_multi14.pkl','_det_lit_np.pkl','_det_expo_np.pkl','_det_narr_np.pkl','_det_info_np.pkl','_det_classic_np.pkl','_det_applied_np.pkl','_det_chat_np.pkl','_det_auto_c_np.pkl','_det_auto_p_np.pkl','_tok_lm.npz','_ngram_ref.pkl','tokenizer.json','tokenizer_config.json','_train_bank.npz','_ref145.pkl'];z=zipfile.ZipFile('models-v1.0.0.zip','w',zipfile.ZIP_DEFLATED);[z.write(f,f) for f in M if os.path.exists(f)];z.close();print('ok')"
```

---

## 5. 建 Release 并上传两个资产

1. 仓库页 → 右侧 **Releases** → **Create a new release**
2. **Choose a tag**：输入 `v1.0.0` → 点 "Create new tag"
3. **Release title**：`jury-flash-CN v1.0.0`
4. **Describe**：可粘贴 `CHANGELOG.md` 的 v1.0.0 部分（**建议包含 `ai_usage.md` 的要点**）
5. **Attach binaries**：上传两个文件
   - `models-v1.0.0.zip`（模型，约 55 MB）
   - `jury.exe`（预编译程序，约 58 MB，**已内置模型，用户无需再下载**）
6. 点 **Publish release**

---

## 6. 收尾

```bash
cd jury-flash
# 在 README 顶部加一句「下载 jury.exe」的链接（可选）
git add README.md
git commit -m "README: 加入 Release 下载说明"
git push
```

**建议再补两个文件（可选但推荐）**：

- `.github/ISSUE_TEMPLATE/bug_report.md`：请用户提交**误报/漏报样本**
- `.github/workflows/`：一个简单的 `python -c "import ast; ..."` 语法检查

---

## 7. 发布后自检清单

- [ ] 仓库里**没有**任何 `.apikey*` 文件
- [ ] 仓库里**没有** `*.pkl / *.npz / tokenizer.json`（它们在 Release）
- [ ] Release 里有 `jury.exe` 与 `models-v1.0.0.zip`
- [ ] `NOTICE` 里标注了**千问分词器（Apache-2.0）**
- [ ] `ai_usage.md` 已明确「判分不用大模型，但训练语料含 AI 生成样本」
- [ ] README 的「已知局限」章节保留（**不要**宣称 100% 准确）

---

## 8. 后续如何更新版本

```bash
# 1) 改代码 / 重训模型 → 新 EXE
# 2) 更新 CHANGELOG.md（新增 v1.0.1 段落）
git add . && git commit -m "v1.0.1: ..."
git tag v1.0.1 && git push --tags
# 3) 在 GitHub 建 v1.0.1 Release，上传新的 jury.exe 与 models zip
```

---

## 附：常见问题

**Q：模型 55 MB 太大了，能不放 Release 吗？**
A：可以只放 `jury.exe`（58 MB，已内置模型）。源码用户再单独下载模型 zip。

**Q：可以放 Git LFS 吗？**
A：可以，但 LFS 有免费额度限制；**Release 附件更适合二进制**。

**Q：要不要一起开源训练语料？**
A：⚠ **不建议直接开源**：① 体积大 ② 第三方许可参差 ③ 含 AI 生成样本。
建议只发布**模型 + 溯源清单**（`NOTICE` 已说明）。
