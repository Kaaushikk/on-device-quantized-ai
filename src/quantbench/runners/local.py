import numpy as np
from quantbench.common import ARTIFACTS, SUPPORTED_VARIANTS, read_json, verify_files


class LocalRunner:
    def __init__(self, variant, threads=1):
        if variant not in SUPPORTED_VARIANTS:
            raise ValueError("Unsupported variant")
        self.variant = variant
        self.threads = threads
        self.manifest = read_json(ARTIFACTS / f"{variant}.manifest.json")
        verify_files(self.manifest)

    def load(self):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(ARTIFACTS / "model", local_files_only=True)
        if self.variant == "pytorch_fp32":
            import torch
            from transformers import AutoModelForSequenceClassification
            torch.set_num_threads(self.threads)
            torch.set_num_interop_threads(1)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                ARTIFACTS / "model", local_files_only=True, attn_implementation="eager")
            self.model.eval()
            if self.model.config.id2label != {0: "NEGATIVE", 1: "POSITIVE"}:
                raise ValueError("Unexpected label mapping")
        else:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = self.threads
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self.model = ort.InferenceSession(str(ARTIFACTS / self.manifest["graph"]),
                                             sess_options=options, providers=["CPUExecutionProvider"])
            if self.model.get_providers() != ["CPUExecutionProvider"]:
                raise ValueError("Unexpected execution provider")

    def preprocess(self, texts, length=256, pad_to_max=True):
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Provide a nonempty batch of nonblank strings")
        if len(texts) > 32 or not 2 <= length <= 512:
            raise ValueError("Batch limit 32; sequence length must be between 2 and 512")
        encoded = self.tokenizer(list(texts), truncation=True, padding="max_length" if pad_to_max else True,
                                 max_length=length, return_tensors="np")
        return {key: np.asarray(encoded[key], dtype=np.int64) for key in ("input_ids", "attention_mask")}

    def prepare(self, inputs):
        if self.variant == "pytorch_fp32":
            import torch
            return {key: torch.from_numpy(value) for key, value in inputs.items()}
        return inputs

    def predict(self, inputs):
        if self.variant == "pytorch_fp32":
            import torch
            with torch.inference_mode():
                return self.model(**inputs).logits.numpy()
        return self.model.run(["logits"], inputs)[0]

    def metadata(self):
        return {"variant": self.variant, "threads": self.threads,
                "provider": "CPU" if self.variant == "pytorch_fp32" else "CPUExecutionProvider",
                "artifact": self.manifest}
