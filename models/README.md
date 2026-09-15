# 模型目录说明

把要测试的模型放在本目录下，每个模型一个子文件夹，例如：

```
models/
  nllb-600m-int8/
  m2m100-418m-int8/
  faster-whisper-large-v3/        # 默认 ASR；存在且含 model.bin 时 transcribe_audio.py 自动用，不必 --model
```

翻译模型在根目录 `config.yaml` 注册；ASR 用独立脚本 `transcribe_audio.py`，不必注册。

CTranslate2 NLLB / M2M100 至少需要：

- `model.bin`
- `sentencepiece.bpe.model`

权重文件较大，默认不提交到 git（见根目录 `.gitignore`）。
