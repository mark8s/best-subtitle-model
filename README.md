# awol-subtitle-model

网页 Demo：选影片和音轨，OpenVINO Whisper 生成原语言字幕，再用 M2M100 或 NLLB 翻译。模型不进 git。

影片默认放在 `/DATA/videos`。服务监听 `0.0.0.0:8080`，日志在 `logs/demo.log`。

## 1. 环境

只用 `/usr/awol/python/bin/python3.11`。系统 `python3` 可能是 3.14。OpenVINO 放在项目外的 `/DATA/ov-venv`，不要和别的项目混用 `.so`。

```bash
ls -l /dev/dri
# 应有 card0 和 renderD128
```

`/DATA` 上经常不能建 symlink，venv 要 `--copies`，并把 `libpython` 拷进去：

```bash
/usr/awol/python/bin/python3.11 -m venv --copies --without-pip /DATA/ov-venv
mkdir -p /DATA/ov-venv/lib
cp -a /usr/awol/python/lib/libpython3.11.so* /DATA/ov-venv/lib/
source /DATA/ov-venv/bin/activate
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
python /tmp/get-pip.py
```

`sys.executable` 必须是 `/DATA/ov-venv/bin/python`。报找不到 `libpython3.11.so.1.0` 就是上面的 `cp` 没做。

```bash
cd /DATA/awol-subtitle-model
python -m pip install -r requirements-openvino.txt
python -m pip install -r requirements-demo.txt
```

系统 PATH 里还要有 `ffmpeg` 和 `ffprobe`。

```bash
python -c "import openvino; print(openvino.__version__)"
python -c "from openvino import Core; print(Core().available_devices)"
```

应能看到 `GPU`。只有 `CPU` 时先看 `/dev/dri` 和核显驱动（`xe` 或 `i915`）。

## 2. 模型

下载到 `models/`。Whisper 是听写，M2M100 和 NLLB 二选一或都下。Silero 只有约 2MB，用来在送进 Whisper 之前跳过静音。

```bash
hf download OpenVINO/whisper-large-v3-int8-ov \
  --local-dir models/openvino-whisper-large-v3-int8

hf download jncraton/m2m100_418M-ct2-int8 \
  --local-dir models/m2m100-418m-int8

hf download osa911/nllb-200-distilled-600M-ct2-int8 \
  --local-dir models/nllb-600m-int8

mkdir -p models/silero-vad
curl -L -o models/silero-vad/silero_vad.onnx \
  https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
```

没有 `hf` 时先 `python -m pip install -U "huggingface_hub[cli]"`。无网机器在能上网的电脑下载后，把这四个目录原样拷过来。

## 3. 启动

```bash
source /DATA/ov-venv/bin/activate
cd /DATA/awol-subtitle-model
python -m demo_server
```

浏览器打开 `http://机器IP:8080`。8080 已被占用时：

```bash
fuser -k 8080/tcp
python -m demo_server
```

影片不在 `/DATA/videos` 时，启动前设置 `MEDIA_ROOT`。

听写先用 Silero 找出人声，一段确定后就送给 Whisper，不必等整片扫完。翻译在网页里选 M2M100 或 NLLB。
