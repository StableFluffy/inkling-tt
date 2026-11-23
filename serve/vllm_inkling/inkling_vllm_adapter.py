import logging
import os

import torch

from inkling_tt import paths
from serve.runtime import InklingRuntime

log = logging.getLogger("inkling.vllm")

WEIGHTS = os.environ.get("INKLING_TT_WEIGHTS", paths.hf_sd())


class InklingForCausalLM:
    def embed_input_ids(self, input_ids):
        raise NotImplementedError("Use the TT prefill/decode interface")

    def forward(self, input_ids, positions, **kwargs):
        raise NotImplementedError("Use the TT prefill/decode interface")

    def compute_logits(self, hidden_states, **kwargs):
        raise NotImplementedError("Use the TT prefill/decode interface")

    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_chunked_prefill": True,
        "supports_async_decode": False,
        "supports_sample_on_device": False,
    }

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len=2048, tt_data_parallel=1, optimizations=None
    ):
        if max_batch_size != 1 or tt_data_parallel != 1:
            raise ValueError(
                f"Inkling TT adapter serves one sequence: max_num_seqs=1, got {max_batch_size} "
                f"(tt_data_parallel={tt_data_parallel})"
            )
        if mesh_device.get_num_devices() != 1:
            raise ValueError("Inkling TT adapter runs on a single p150 (1x1 mesh)")
        obj = cls()
        obj.mesh_device = mesh_device
        obj.vocab_out = hf_config.get_text_config().vocab_size
        obj.rt = InklingRuntime(mesh_device, WEIGHTS, lmax=max_seq_len, log=log.info)
        return obj

    @classmethod
    def get_max_tokens_all_users(cls, **kwargs):
        return kwargs["max_model_len"] * kwargs["max_num_seqs"]

    def allocate_kv_cache(self, shape, dtype, num_layers):
        return [None] * num_layers

    def warmup_model_prefill(self, kv_cache, enable_trace, can_sample_on_device, **kwargs):
        return None

    def warmup_model_decode(self, kv_cache, enable_trace, max_batch_size, num_blocks, can_sample_on_device, **kwargs):
        if enable_trace:
            self.rt.capture(log=log.info)

    def _pad(self, logits):
        out = torch.full((self.vocab_out,), -float("inf"))
        out[: logits.numel()] = logits
        return out

    def prefill_forward(
        self,
        tokens,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
        empty_slots=None,
        enable_trace=True,
        sampling_params=None,
        start_pos=None,
        **kwargs,
    ):
        if sampling_params is not None:
            raise ValueError("Inkling TT adapter samples on host only")
        rows = len(prompt_lens)
        if rows != 1:
            raise ValueError(f"Inkling TT adapter prefill expects one row, got {rows}")
        start = 0 if start_pos is None else int(start_pos[0])
        end = int(prompt_lens[0])
        ids = tokens[0, start:end].tolist()
        logits = self.rt.feed(ids, start)
        return self._pad(logits).reshape(1, 1, -1)

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        read_from_device=True,
        sampling_params=None,
        **kwargs,
    ):
        if sampling_params is not None:
            raise ValueError("Inkling TT adapter samples on host only")
        pos = torch.as_tensor(start_pos).flatten().tolist()
        out = torch.zeros(len(pos), 1, self.vocab_out)
        for row, p in enumerate(pos):
            if p < 0:
                continue
            if row != 0:
                raise ValueError("Inkling TT adapter decodes one live row")
            out[row, 0] = self._pad(self.rt.step(int(tokens[row, 0]), int(p)))
        return out

    def release_request(self, slot):
        self.rt.pos = None

    def release_persistent_capture(self):
        self.rt.release()
