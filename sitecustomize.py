"""Runtime compatibility patches for subprocesses launched from this checkout."""

try:
    from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer

    if not hasattr(Qwen2Tokenizer, "all_special_tokens_extended"):

        @property
        def all_special_tokens_extended(self):
            return list(self.added_tokens_decoder.values())

        Qwen2Tokenizer.all_special_tokens_extended = all_special_tokens_extended
except ImportError:
    pass

try:
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

    if not hasattr(Qwen3VLTextConfig, "tie_word_embeddings"):
        Qwen3VLTextConfig.tie_word_embeddings = False
except ImportError:
    pass
