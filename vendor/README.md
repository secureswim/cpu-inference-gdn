# Upstream references

`stream.c` is John D. McCalpin's STREAM 5.10 C source, downloaded from [the official source distribution](https://www.cs.virginia.edu/stream/FTP/Code/stream.c). Its license and publication conditions appear in the file. The array size and repetition count are compile-time settings. No kernel source modifications are applied.

The main experiment fetches ggml from llama.cpp commit `8bb3cf0d216a5ae48360179f2931daf284be4d01`. `scripts/prepare_ggml.py` exports the committed tree when a local checkout is provided, guards a Windows API unavailable in the local MinGW headers, and inserts optional phase timers into the actual upstream GDN implementation. Its arithmetic remains unchanged. llama.cpp uses the MIT license, retained in the fetched source.

The assignment handout is an input document and is excluded from the public project files.
