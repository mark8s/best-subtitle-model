# 模型目录说明

把要测试的模型放在本目录下，每个模型一个子文件夹，例如：

```
models/
  nllb-600m-int8/
    model.bin
    sentencepiece.bpe.model
    shared_vocabulary.json
    config.json
  another-model/
    ...
```

然后在项目根目录的 `config.yaml` 里注册，并设置 `active_model`。

CTranslate2 NLLB 模型至少需要：

- `model.bin`
- `sentencepiece.bpe.model`

权重文件较大，默认不提交到 git（见根目录 `.gitignore`）。
