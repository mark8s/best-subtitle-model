# awol-subtitle-model

本地字幕翻译测试（SRT → SRT，时间轴不变）。

通过 `config.yaml` 切换模型；当前默认：`nllb-600m-int8`。

> **模型不进 git。** 需要自行下载到 `models/` 目录后再测。

---

## 模型列表

| 配置 id | 模型 | 格式 / 大小 | 下载地址 | 放置目录 |
|---|---|---|---|---|
| `nllb-600m-int8` | NLLB-200 Distilled 600M（Meta）INT8 | CTranslate2，约 591MB | [Hugging Face: osa911/nllb-200-distilled-600M-ct2-int8](https://huggingface.co/osa911/nllb-200-distilled-600M-ct2-int8) | `models/nllb-600m-int8/` |

原模型：[facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M)（CC-BY-NC-4.0，仅个人非商业测试）。

### 下载方式

**方式 A：huggingface-cli（推荐）**

```bash
pip install -U "huggingface_hub[cli]"

huggingface-cli download osa911/nllb-200-distilled-600M-ct2-int8 \
  --local-dir models/nllb-600m-int8
```

**方式 B：git lfs**

```bash
git lfs install
git clone https://huggingface.co/osa911/nllb-200-distilled-600M-ct2-int8 models/nllb-600m-int8
```

下载后目录应包含：

```text
models/nllb-600m-int8/
  model.bin
  sentencepiece.bpe.model
  shared_vocabulary.json
  config.json
```

在 `config.yaml` 里用 `active_model: nllb-600m-int8` 启用（已是默认）。

---

## 本地（Windows）

```bash
cd /d/mark/awol/awol-subtitle-model

py -3.11 -m venv .venv
source .venv/Scripts/activate          # Git Bash
# 或 PowerShell: .\.venv\Scripts\Activate.ps1

python -m pip install -U pip
pip install -r requirements.txt

python translate_srt.py --list-models
```

测试（日语 → 简体中文）：

```bash
python translate_srt.py \
  -i input/The.Pursuit.of.Happyness.ja.srt \
  -o output/The.Pursuit.of.Happyness.zh.srt \
  --src jpn_Jpan \
  --tgt zho_Hans \
  --device cpu
```

> 若 `python` 无输出：先激活 `.venv`，或用 `.venv/Scripts/python.exe ...`

---

## Linux / NAS

不要复用 Windows 的 `.venv`，用本机 Python 重建：

```bash
cd ~/awol-subtitle-model   # 按实际路径改

/usr/awol/python/bin/python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install -U pip
pip install -r requirements.txt

# 若本机还没有模型，先下载到 models/nllb-600m-int8/
python translate_srt.py --list-models
```

测试：

```bash
python translate_srt.py \
  -i input/The.Pursuit.of.Happyness.ja.srt \
  -o output/The.Pursuit.of.Happyness.zh.srt \
  --src jpn_Jpan \
  --tgt zho_Hans \
  --device cpu
```

无 NVIDIA 显卡时用 `--device cpu`（Intel 核显当前这套不支持）。

---

## 常用命令

```bash
# 只指定输入，输出默认到 output/
python translate_srt.py -i input/xxx.srt --src jpn_Jpan --tgt zho_Hans

# 指定输出
python translate_srt.py -i input/xxx.srt -o output/xxx.zh.srt --src jpn_Jpan --tgt zho_Hans

# 切换模型
python translate_srt.py --model-id nllb-600m-int8 ...
```

| 参数 | 含义 | 默认 |
|---|---|---|
| `-i` | 输入 srt / 目录 | `input/` |
| `-o` | 输出 srt / 目录 | `output/` |
| `--src` | 源语言 | config（当前 `eng_Latn`） |
| `--tgt` | 目标语言 | config（当前 `zho_Hans`） |
| `--device` | `cpu` / `cuda` / `auto` | `auto` |

语言码：`jpn_Jpan` 日语，`zho_Hans` 简体中文，`eng_Latn` 英语。  
**不会自动识别语言**，日语字幕必须加 `--src jpn_Jpan`。
