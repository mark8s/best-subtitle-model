# awol-subtitle-model

本地字幕翻译测试（SRT → SRT，时间轴不变）。

通过 `config.yaml` 切换模型；当前默认：`nllb-600m-int8`。

> **模型不进 git。** 需要自行下载到 `models/` 目录后再测。

---

## 模型列表

| 配置 id | 模型 | 格式 / 大小 | 下载地址 | 放置目录 | 语言码 |
|---|---|---|---|---|---|
| `nllb-600m-int8` | NLLB-200 Distilled 600M INT8 | CTranslate2，约 591MB | [osa911/nllb-200-distilled-600M-ct2-int8](https://huggingface.co/osa911/nllb-200-distilled-600M-ct2-int8) | `models/nllb-600m-int8/` | `jpn_Jpan` / `zho_Hans` / `eng_Latn` |
| `m2m100-418m-int8` | M2M100 418M INT8 | CTranslate2，约 468MB | [jncraton/m2m100_418M-ct2-int8](https://huggingface.co/jncraton/m2m100_418M-ct2-int8) | `models/m2m100-418m-int8/` | `ja` / `zh` / `en` |

- NLLB 原模型：[facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M)（CC-BY-NC-4.0）
- M2M100 原模型：[facebook/m2m100_418M](https://huggingface.co/facebook/m2m100_418M)（MIT）

### 下载方式

```bash
pip install -U "huggingface_hub[cli]"

# NLLB
hf download osa911/nllb-200-distilled-600M-ct2-int8 \
  --local-dir models/nllb-600m-int8

# M2M100
hf download jncraton/m2m100_418M-ct2-int8 \
  --local-dir models/m2m100-418m-int8
```

M2M100 目录应包含：`model.bin`、`sentencepiece.bpe.model`、`vocab.json`、`tokenizer_config.json` 等。

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

NLLB 测试（日语 → 简体中文）：

```bash
python translate_srt.py \
  -i input/The.Pursuit.of.Happyness.ja.srt \
  -o output/The.Pursuit.of.Happyness.zh.srt \
  --src jpn_Jpan --tgt zho_Hans --device cpu
```

M2M100 测试（注意语言码不同）：

```bash
python translate_srt.py --model-id m2m100-418m-int8 \
  -i input/The.Pursuit.of.Happyness.ja.srt \
  -o output/The.Pursuit.of.Happyness.m2m.zh.srt \
  --src ja --tgt zh --device cpu
```

> 若 `python` 无输出：先激活 `.venv`，或用 `.venv/Scripts/python.exe ...`

---

## Linux / NAS

```bash
cd ~/awol-subtitle-model

/usr/awol/python/bin/python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install -U pip
pip install -r requirements.txt
python translate_srt.py --list-models
```

```bash
# NLLB
python translate_srt.py \
  -i input/The.Pursuit.of.Happyness.ja.srt \
  -o output/The.Pursuit.of.Happyness.zh.srt \
  --src jpn_Jpan --tgt zho_Hans --device cpu

# M2M100
python translate_srt.py --model-id m2m100-418m-int8 \
  -i input/The.Pursuit.of.Happyness.ja.srt \
  -o output/The.Pursuit.of.Happyness.m2m.zh.srt \
  --src ja --tgt zh --device cpu
```

无 NVIDIA 显卡时用 `--device cpu`。

---

## 常用命令

```bash
python translate_srt.py -i input/xxx.srt --src jpn_Jpan --tgt zho_Hans
python translate_srt.py --model-id m2m100-418m-int8 -i input/xxx.srt --src ja --tgt zh
```

| 参数 | 含义 | 默认 |
|---|---|---|
| `-i` | 输入 srt / 目录 | `input/` |
| `-o` | 输出 srt / 目录 | `output/` |
| `--model-id` | 用哪个模型 | `config.active_model` |
| `--src` / `--tgt` | 源 / 目标语言 | 看模型配置 |
| `--device` | `cpu` / `cuda` / `auto` | `auto` |

**语言码注意：**

| 模型 | 日语 | 中文 | 英语 |
|---|---|---|---|
| NLLB | `jpn_Jpan` | `zho_Hans` | `eng_Latn` |
| M2M100 | `ja` | `zh` | `en` |

不会自动识别语言。M2M100 若误传了 NLLB 码（如 `jpn_Jpan`），脚本会尝试自动映射到 `ja`。
