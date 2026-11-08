# Upstream references

The Mamba files come from [state-spaces/mamba](https://github.com/state-spaces/mamba/tree/e9594ce1c732d97440f0332fdc43170a2294dbfa) at commit `e9594ce1c732d97440f0332fdc43170a2294dbfa`. Their copyright belongs to Dao AI Lab and Goombalab. The accompanying `LICENSE` is the upstream Apache 2.0 license. These files retain their original contents.

- `mamba3_step_fn.py` supplies `selective_state_update_fused_ref_v2`.
- `mamba3_mimo_rotary_step.py` supplies `apply_rotary_qk_inference_reference`.
- `mamba3.py` establishes the preprocessing, rotary pairing convention, carried state, and decode scope.
- `test_mamba3_mimo_step.py` documents the upstream tests and their GPU requirements.

The project extracts the two pure PyTorch reference functions using Python's AST, then executes their unmodified bodies on CPU. This avoids importing their surrounding GPU-only modules. Dense projections and BC normalization fall outside the recurrence API. Inputs include post-normalization BC, post-softplus DT, post-sigmoid trapezoidal gates, and negative A. The CPU kernel includes rotary transformation, recurrence, skip connection, SiLU gating, and MIMO output mixing.

`stream.c` is John D. McCalpin's STREAM 5.10 C source, downloaded from [the official source distribution](https://www.cs.virginia.edu/stream/FTP/Code/stream.c). Its license and publication conditions appear in the file. The array size and repetition count are compile-time settings. No kernel source modifications are applied.

The main experiment fetches ggml from llama.cpp commit `8bb3cf0d216a5ae48360179f2931daf284be4d01`. `scripts/prepare_ggml.py` exports the committed tree when a local checkout is provided, guards a Windows API unavailable in the local MinGW headers, and inserts optional phase timers into the actual upstream GDN implementation. Its arithmetic remains unchanged. llama.cpp uses the MIT license, retained in the fetched source.

The assignment handout is an input document and is excluded from the public project files.
