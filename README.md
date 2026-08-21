# awol-subtitle-model

本地字幕测试工具：

- **翻译**：`translate_srt.py`（SRT → SRT）
- **听写**：`transcribe_audio.py`（音频/视频 → SRT，faster-whisper）

通过 `config.yaml` 切换翻译模型；当前默认：`nllb-600m-int8`。

> **模型不进 git。** 需要自行下载到 `models/`（或 Whisper 首次自动缓存）后再测。

---

## 模型列表

| 配置 id | 模型 | 格式 / 大小 | 下载地址 | 放置目录 | 语言码 |
|---|---|---|---|---|---|
| `nllb-600m-int8` | NLLB-200 Distilled 600M INT8 | CTranslate2，约 591MB | [osa911/nllb-200-distilled-600M-ct2-int8](https://huggingface.co/osa911/nllb-200-distilled-600M-ct2-int8) | `models/nllb-600m-int8/` | `jpn_Jpan` / `zho_Hans` / `eng_Latn` |
| `m2m100-418m-int8` | M2M100 418M INT8 | CTranslate2，约 468MB | [jncraton/m2m100_418M-ct2-int8](https://huggingface.co/jncraton/m2m100_418M-ct2-int8) | `models/m2m100-418m-int8/` | `ja` / `zh` / `en` |
| （ASR，独立） | faster-whisper | 随型号变化 | [Systran/faster-whisper-*](https://huggingface.co/Systran) 自动缓存或下到 `models/` | 见下方 | `ja` / `zh` / `en` / `it` / `ko` … |

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

### faster-whisper（音频/视频 → SRT，独立脚本）

脚本：`transcribe_audio.py`，不和翻译模型串联。解码靠 **PyAV**（一般不必单独装系统 ffmpeg）。

**安装**

```bash
pip install faster-whisper
# 或：pip install -r requirements.txt
```

**模型下载**（二选一）

| 方式 | 说明 |
|---|---|
| 首次运行自动下 | `--model small` 等型号名；缓存到 `~/.cache/huggingface/hub/`，**不会**出现在项目 `models/` |
| 手动放到项目里 | 见下方命令，再用 `--model models/...` |

```bash
pip install -U "huggingface_hub[cli]"

hf download Systran/faster-whisper-small \
  --local-dir models/faster-whisper-small
# 更大：Systran/faster-whisper-large-v3 → models/faster-whisper-large-v3
```

| 型号 | 用途 |
|---|---|
| `tiny` / `base` | 最快冒烟 |
| `small` | 默认，日常试听 |
| `medium` / `large-v3` | 更好 / 最好也最慢 |
| `distil-large-v3` | 接近 large，更快 |

**运行**

```bash
# 自动检语言；模型首次自动下载到缓存
python transcribe_audio.py -i input/demo.wav

python transcribe_audio.py -i input/demo.mp3 -o output/demo.srt \
  --model small --device cpu

# 使用项目内模型目录
python transcribe_audio.py -i input/demo.mp3 \
  --model models/faster-whisper-small

# 已知语言时指定更稳；--task translate = 译成英语字幕（不是译中文）
python transcribe_audio.py -i input/demo.wav --language it
python transcribe_audio.py -i input/demo.wav --language it --task translate
```

**语言怎么定（重要）**

- 不写 `--language`：听开头约 30s 做语种分类，日志：`Detected language : xx (0.xx)`。
- **置信度低（例如低于 0.7～0.8）不可信**，片头静音/配乐/多语混杂时容易检错；应抽对白片段再检，或人工指定后整片 `--language xx` 锁死。
- 写了 `--language`：跳过检测。SRT **无语言字段**，以日志或文本为准。
- `--task transcribe`（默认）→ 源语言字幕；`--task translate` → **英语**字幕（不是中文）。
- 语种列表同 [OpenAI Whisper](https://github.com/openai/whisper)（约 99 种短码：`it`/`ja`/`zh`/`en`…）。

**速度 / 设备**

- GPU 仅 **NVIDIA CUDA**（`--device cuda`）。Intel 核显不可用，NAS 无独显时用 `--device cpu`。
- CPU 参考：约 2 小时音频 + `small`，RTF≈0.08 量级（约数分钟～十几分钟），属正常。
- 可加快：`--beam-size 1`、`--vad`；要质量再加大型号。

**无字幕片推荐流程（ASR → 翻译）**

```text
音频/视频 → Whisper（确认语种）→ 源语言 SRT → NLLB/M2M100 → 中文 SRT
```

```bash
# 1) 短片段自动检语种，看 Detected language 与置信度
ffmpeg -i movie.mkv -t 180 -vn -acodec pcm_s16le -ar 16000 -ac 1 input/clip.wav
python transcribe_audio.py -i input/clip.wav --model models/faster-whisper-small --device cpu

# 2) 确认语种后锁死整片（例：意大利语 it）
python transcribe_audio.py -i input/movie.wav -o output/movie.it.srt \
  --model models/faster-whisper-small --language it --vad --device cpu

# 3) 同一源语言去做翻译（Whisper 短码 → 翻译模型码）
python translate_srt.py --model-id m2m100-418m-int8 \
  -i output/movie.it.srt -o output/movie.zh.srt --src it --tgt zh --device cpu
# NLLB 则：--src ita_Latn --tgt zho_Hans
```

Whisper → 翻译常用映射：

| Whisper | M2M100 | NLLB |
|---|---|---|
| `it` | `it` | `ita_Latn` |
| `ja` | `ja` | `jpn_Jpan` |
| `zh` | `zh` | `zho_Hans` |
| `en` | `en` | `eng_Latn` |
| `ko` | `ko` | `kor_Hang` |

时间轴：Whisper 分段与人工字幕不会逐条对齐，属正常；乱轴+乱字时优先检查语种是否检错。

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
  -o output/The.Pursuit.of.Happyness.zh2.srt \
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
  -o output/The.Pursuit.of.Happyness.zh2.srt \
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
