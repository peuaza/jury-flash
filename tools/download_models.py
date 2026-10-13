# -*- coding: utf-8 -*-
"""download_models.py — 从 GitHub Release 下载模型文件到仓库根目录

用法：
    python tools/download_models.py --repo 用户名/jury-flash --tag v1.0.0
    python tools/download_models.py --url https://.../models-v1.0.0.zip
"""
import argparse, io, os, sys, zipfile, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
UA = "Mozilla/5.0 (compatible; jury-flash-downloader)"


def fetch(url, timeout=900):
    r = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(r, timeout=timeout) as x:
        return x.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", help="GitHub 仓库，如 用户名/jury-flash")
    ap.add_argument("--tag", default="v1.0.0", help="Release 标签（默认 v1.0.0）")
    ap.add_argument("--asset", default="models-v1.0.0.zip", help="Release 中的资产名")
    ap.add_argument("--url", help="直接给 zip 下载地址（跳过 --repo）")
    a = ap.parse_args()
    if a.url:
        url = a.url
    elif a.repo:
        url = "https://github.com/%s/releases/download/%s/%s" % (a.repo, a.tag, a.asset)
    else:
        print("  请给 --repo 或 --url")
        return 2
    print("  下载 %s" % url)
    try:
        data = fetch(url)
    except Exception as e:
        print("  x 下载失败：%s" % str(e)[:200])
        print("    可改为手动下载并解压到仓库根目录")
        return 2
    zp = os.path.join(ROOT, "_models_tmp.zip")
    with io.open(zp, "wb") as f:
        f.write(data)
    print("  解压到 %s" % ROOT)
    with zipfile.ZipFile(zp) as z:
        z.extractall(ROOT)
    os.remove(zp)
    print("  完成。可运行：python jury_cli.py score -f examples/示例_AI诗.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
