"""OpenVINO NLLB 翻译（Intel GPU / CPU）。

权重是 OpenVINO IR，目录默认 models/openvino-nllb-600m-int8。
和 CTranslate2 的 models/nllb-600m-int8 不能混用。
"""

from __future__ import annotations

from pathlib import Path


class OpenVinoNllb:
    def __init__(self, model_dir: Path, device: str = "GPU") -> None:
        model_dir = model_dir.resolve()
        encoder = model_dir / "openvino_encoder_model.xml"
        if not encoder.is_file():
            raise FileNotFoundError(
                f"未找到 OpenVINO NLLB：{encoder}。"
                "请在 ov-venv 里执行 python export_openvino_nllb.py"
            )
        try:
            from optimum.intel import OVModelForSeq2SeqLM
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "未安装 OpenVINO 翻译依赖。在 ov-venv 里先装 CPU 版 torch，再执行 "
                "python -m pip install -r requirements-openvino-translate.txt"
            ) from exc

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = OVModelForSeq2SeqLM.from_pretrained(model_dir, device=device)

    def translate(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        outputs: list[str] = [""] * len(texts)
        indexes: list[int] = []
        payload: list[str] = []
        for index, text in enumerate(texts):
            stripped = text.strip()
            if not stripped:
                outputs[index] = text
                continue
            indexes.append(index)
            payload.append(stripped)
        if not payload:
            return outputs

        bos_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
        unk_id = getattr(self.tokenizer, "unk_token_id", None)
        if not isinstance(bos_id, int) or bos_id < 0 or bos_id == unk_id:
            raise RuntimeError(f"目标语言码无效: {tgt_lang}")

        self.tokenizer.src_lang = src_lang
        encoded = self.tokenizer(
            payload,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        generated = self.model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            forced_bos_token_id=bos_id,
            max_new_tokens=256,
            num_beams=2,
        )
        sequences = generated.sequences if hasattr(generated, "sequences") else generated
        decoded = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
        if len(decoded) != len(indexes):
            raise RuntimeError(f"翻译条数不一致：送入 {len(indexes)}，得到 {len(decoded)}")
        for index, line in zip(indexes, decoded):
            outputs[index] = line.strip()
        return outputs

    def __call__(self, src: str, tgt: str, texts: list[str]) -> list[str]:
        return self.translate(texts, src, tgt)
