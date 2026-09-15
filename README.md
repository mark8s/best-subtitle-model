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
| （ASR，独立） | faster-whisper（默认 `large-v3`） | 随型号变化 | 见下方 | 见下方 | `ja` / `zh` / `en` / `it` / `ko` … |

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
pip install "faster-whisper>=1.1.0"
# 或：pip install -r requirements.txt
```

**模型下载**（二选一）

| 方式 | 说明 |
|---|---|
| 首次运行自动下 | 型号名如 `large-v3`；缓存到 `~/.cache/huggingface/hub/`，**不会**出现在项目 `models/`。机器必须能访问 Hugging Face |
| 本机下载后拷到服务器 | **推荐无网 NAS**。下到 `models/faster-whisper-large-v3/`，脚本默认会用这个目录，**不必再写 `--model`** |

在**能上网的电脑**上下载（Windows / Git Bash 或 Linux 均可）：

```bash
cd /path/to/awol-subtitle-model
pip install -U "huggingface_hub[cli]"

hf download Systran/faster-whisper-large-v3 \
  --local-dir models/faster-whisper-large-v3
```

目录里应有 `model.bin`、`config.json`、`tokenizer.json` 等。把整个文件夹拷到服务器的 `models/faster-whisper-large-v3/`。

其它常用（非默认）：

```bash
hf download Systran/faster-whisper-small --local-dir models/faster-whisper-small
hf download mobiuslabsgmbh/faster-whisper-large-v3-turbo \
  --local-dir models/faster-whisper-large-v3-turbo
```

| 型号 | 用途 |
|---|---|
| `tiny` / `base` / `small` | 最快冒烟 / 日常试听 |
| `large-v3` | **默认**；质量最好，也最慢、最占内存 |
| `large-v3-turbo` / `turbo` | 更快，质量通常不如 large-v3 |
| `medium` / `distil-large-v3` | 折中 / 蒸馏加速 |

默认权重：[Systran/faster-whisper-large-v3](https://huggingface.co/Systran/faster-whisper-large-v3)。Turbo 见 [mobiuslabsgmbh/faster-whisper-large-v3-turbo](https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo)。

**运行**

`models/faster-whisper-large-v3/` 存在时，下面这条就会走本地权重，无需 `--model`：

```bash
python transcribe_audio.py -i input/malena.wav
python transcribe_audio.py -i input/demo.wav
```

```bash
python transcribe_audio.py -i input/demo.mp3 -o output/demo.srt --device cpu

# 强制指定目录（一般不必）
python transcribe_audio.py -i input/demo.mp3 \
  --model models/faster-whisper-large-v3

# 已知语言时指定更稳（听写原语言字幕，不是译成中文）
python transcribe_audio.py -i input/为奴十二年.wav --language en
python transcribe_audio.py -i "input/为奴十二年-中文.wav" --language zh
python transcribe_audio.py -i input/demo.wav --language it

# --task translate = Whisper 自带「译成英语」，不是中文
python transcribe_audio.py -i input/demo.wav --language it --task translate
```

默认启用 VAD，并关闭上一段文本继承，以减少静音、音乐片段中的重复幻觉。特殊情况下可调整：

```bash
# 关闭 VAD
python transcribe_audio.py -i input/demo.wav --no-vad

# 重新启用跨段文本上下文
python transcribe_audio.py -i input/demo.wav --condition-on-previous-text
```

转写过程中，每完成一段都会立即保存到同目录的 `.partial.srt`。可随时按 `Ctrl+C` 中断并查看，例如：

```text
output/movie.partial.srt
```

全部完成后，脚本会把它自动改名为正式的 `output/movie.srt`。部分字幕可以查看，但重新运行仍会从头转写。

**语言怎么定（重要）**

`--language` 指定的是**音频里说的语言**，默认 `--task transcribe` 会生成**同语言字幕**：

| 音频 | 命令 | 输出 |
|---|---|---|
| 英语对白 | `--language en` | 英文字幕 |
| 中文对白 | `--language zh` | 中文字幕 |
| 意大利语对白 | `--language it` | 意大利文字幕 |

```bash
python transcribe_audio.py -i input/为奴十二年.wav --language en
python transcribe_audio.py -i "input/为奴十二年-中文.wav" --language zh
```

这不是「翻译成中文」。英语片要中文字幕：先 `--language en` 出英文字幕，再用 `translate_srt.py`。

- 不写 `--language`：听开头约 30s 做语种分类，日志：`Detected language : xx (0.xx)`。
- **置信度低（例如低于 0.7～0.8）不可信**，片头静音/配乐/多语混杂时容易检错；应抽对白片段再检，或人工指定后整片 `--language xx` 锁死。
- 写了 `--language`：跳过检测。SRT **无语言字段**，以日志或文本为准。
- `--task transcribe`（默认）→ 源语言字幕；`--task translate` → **英语**字幕（不是中文）。
- 语种列表同 [OpenAI Whisper](https://github.com/openai/whisper)（约 99 种短码：`it`/`ja`/`zh`/`en`…）。

**速度 / 设备**

- GPU 仅 **NVIDIA CUDA**（`--device cuda`）。Intel 核显不可用，NAS 无独显时用 `--device cpu`。
- CPU 参考：约 2 小时音频 + `small`，RTF≈0.08 量级（约数分钟～十几分钟），属正常。默认 `large-v3` 更慢、更占内存，NAS 上请保证 `available` 明显高于 turbo 时的需求。
- 可用 `--beam-size 1` 加快；VAD 已默认启用，会跳过无对白区间。

**用 ffmpeg 抽 wav**（16kHz 单声道 PCM，给 Whisper 用）

`-vn` = no video，不输出画面，只抽音频。

```bash
# 整片（默认第一条音轨）
ffmpeg -i input/movie.mkv -vn -acodec pcm_s16le -ar 16000 -ac 1 input/movie.wav

# 指定音轨（例：stream index 2，中文 chi）
ffmpeg -i "input/为奴十二年.2013.1080p.h265.mkv" -map 0:2 -vn \
  -acodec pcm_s16le -ar 16000 -ac 1 "input/为奴十二年-中文.wav"

# 只抽前 3 分钟（先检语种）
ffmpeg -i input/movie.mkv -t 180 -vn -acodec pcm_s16le -ar 16000 -ac 1 input/clip.wav
```

**无字幕片推荐流程（ASR → 翻译）**

```text
音频/视频 → Whisper（确认语种）→ 源语言 SRT → NLLB/M2M100 → 中文 SRT
```

```bash
# 1) 短片段自动检语种，看 Detected language 与置信度
ffmpeg -i input/movie.mkv -t 180 -vn -acodec pcm_s16le -ar 16000 -ac 1 input/clip.wav
python transcribe_audio.py -i input/clip.wav --device cpu

# 2) 整片抽音，确认语种后锁死（例：意大利语 it）
ffmpeg -i input/movie.mkv -vn -acodec pcm_s16le -ar 16000 -ac 1 input/movie.wav
python transcribe_audio.py -i input/movie.wav -o output/movie.it.srt \
  --language it --device cpu

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

# /DATA 上常无法 symlink，用 --copies；先清掉半成品
rm -rf .venv
/usr/awol/python/bin/python3.11 -m venv --copies .venv
source .venv/bin/activate

python -m pip install -U pip
pip install -r requirements.txt
python translate_srt.py --list-models
```

必须用 `/usr/awol/python/bin/python3.11`（3.11）。系统里的 `python3` 可能是 3.14，不要拿它建这个项目的 venv。

若 `ensurepip` 仍失败（exit 127），改用不带 pip 的 venv，再用官方脚本装 pip。`--copies` 不会带上 `libpython`，需要拷进 `.venv/lib/`，否则会报 `libpython3.11.so.1.0: No such file`：

```bash
rm -rf .venv
/usr/awol/python/bin/python3.11 -m venv --copies --without-pip .venv
mkdir -p .venv/lib
cp -a /usr/awol/python/lib/libpython3.11.so* .venv/lib/
source .venv/bin/activate
python -c "import sys; print(sys.version)"
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
python /tmp/get-pip.py
pip install -r requirements.txt
```

拷完库仍找不到时，当前 shell 可先：`export LD_LIBRARY_PATH=/usr/awol/python/lib:$LD_LIBRARY_PATH`

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
