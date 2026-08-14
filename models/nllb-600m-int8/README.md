---
license: cc-by-nc-4.0
language:
- multilingual
tags:
- translation
- ctranslate2
- nllb
- int8
- 8-bit
- quantized
base_model: facebook/nllb-200-distilled-600M
---

# NLLB-200 Distilled 600M — CTranslate2 INT8 (8-bit)

INT8 quantized version of [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) for use with [CTranslate2](https://github.com/OpenNMT/CTranslate2).

## License

This model is licensed under **CC-BY-NC-4.0** (Creative Commons Attribution-NonCommercial 4.0).
- Original model by Meta AI (NLLB Team)
- Converted to CTranslate2 INT8 format

**For personal, non-commercial use only.**

## Files

- `model.bin` — CTranslate2 INT8 quantized model (~591MB)
- `sentencepiece.bpe.model` — SentencePiece BPE tokenizer
- `config.json` — CTranslate2 model config
- `shared_vocabulary.json` — Shared vocabulary

## Attribution

Original model: [No Language Left Behind (NLLB)](https://ai.meta.com/research/no-language-left-behind/)
by Meta AI. Paper: [arxiv.org/abs/2207.04672](https://arxiv.org/abs/2207.04672)
