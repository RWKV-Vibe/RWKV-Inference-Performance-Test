<div align="center">

<h1> RWKV Inference Performance Testing Guide </h1>

[![中文](https://img.shields.io/badge/Language-中文-orange.svg)](./README.md)
[![English](https://img.shields.io/badge/Language-English-blue.svg)](./README_EN.md)

</div>

This guide will help you configure and test RWKV model inference performance on your local device. We also welcome you to submit issues reporting RWKV inference performance data.

We provide testing methods based on three different inference tools: [web-rwkv testing](#web-rwkv) | [RWKV pip testing](#rwkv-pip) | [llama.cpp testing](#llamacpp).

|Testing Method|Required Model Format|Supported GPU Types|
|---|---|---|
|web-rwkv|`.st`|All GPUs supporting vulkan, including integrated graphics|
|RWKV pip|`.pth`|NVIDIA GPUs supporting CUDA (CPU mode available but not recommended for testing)|
|llama.cpp|`.gguf`|All types of GPUs, including integrated graphics and CPU|

Before starting, please ensure you have the following:

- Sufficient storage space for downloading model files
- Basic command line operation skills
- Python environment installed (required for test method 2)

## web-rwkv

### Test Preparation

1. Download web-rwkv tool: Visit [web-rwkv releases](https://github.com/cryscan/web-rwkv/releases) page, download the latest version suitable for your operating system, and extract it in an empty directory
2. Get RWKV7-G1 2.9B model: [Click to download](https://huggingface.co/shoumenchougou/RWKV-ST-model/resolve/main/rwkv7-g1-2.9b-20250519-ctx4096.st?download=true) `rwkv7-g1-2.9b-20250519-ctx4096.st` RWKV model file for performance testing
3. Move the downloaded model file to the `dist` folder in the web-rwkv extracted directory

### Inference Performance Testing

In the `web-rwkv/dist` directory, right-click and select "Open in Integrated Terminal". Then execute the following commands for testing different quantization precisions:

1. fp16 precision inference performance test:

```bash
./bench.exe --model "./rwkv7-g1-2.9b-20250519-ctx4096.st"
```
2. INT8 quantization inference performance test:
```bash
./bench.exe --model "./rwkv7-g1-2.9b-20250519-ctx4096.st" --quant 31
```
3. NF4 quantization inference performance test:
```bash
./bench.exe --model "./rwkv7-g1-2.9b-20250519-ctx4096.st" --quant-nf4 31
```

>[!WARNING]
> `--quant` and `--quant-nf4` are quantization layers, recommended to keep the default value `31`

**Move the cursor and select your inference device (recommended to use the default `vulkan` backend)**

![web-rwkv-result](./img/web-rwkv-seclet-adapter.png)

After testing, the terminal will output a performance report in the following format:

| model                                                    | quant_int8 | quant_float4 |    test |            t/s |
|----------------------------------------------------------|-----------:|-------------:|--------:|---------------:|
| rwkv7-g1-2.9b-20250519-ctx4096.st                        |          0 |            0 |   pp512 |        1022.89 |
| rwkv7-g1-2.9b-20250519-ctx4096.st                        |          0 |            0 |   tg128 |          95.98 |

Where **t/s** represents inference speed (tokens/second). Please copy this table from the terminal, paste it into a new issue, and provide your **CPU and GPU model**.

---

## RWKV pip

Test performance data by calling the [RWKV pip repository](https://pypi.org/project/rwkv/) through Python code for inference.

To test based on RWKV pip, you need to download a RWKV7-G1 2.9B model in `.pth` format first:

- [ModelScope Download](https://modelscope.cn/models/RWKV/rwkv7-g1/resolve/master/rwkv7-g1-2.9b-20250519-ctx4096.pth)
- [Hugging Face Download](https://huggingface.co/BlinkDL/rwkv7-g1/resolve/main/rwkv7-g1-2.9b-20250519-ctx4096.pth?download=true)

### Preparing Test Environment

> [!TIP]
> Recommended to use [AnaConda](https://anaconda.org/anaconda/conda) for Python environment management

Run the following commands to create a new conda environment, install necessary Python packages, and clone this repository:

```bash
conda create -n rwkv-pip-test python=3.12
conda activate rwkv-pip-test
pip install torch --upgrade --extra-index-url https://download.pytorch.org/whl/cu128
pip install rwkv psutil prompt_toolkit tokenizers
git clone https://github.com/ehooon/RWKV-Inference-Performance-Test.git
```

### Testing Inference Performance

Open `rwkv-pip-test.py` file and edit the key parameters:

| Parameter Name | Function Description | Available Values | Notes |
|---------|---------|--------|------|
| `args.strategy` | Running device and precision | `cuda fp16`<br>`cpu fp16`<br>`cuda fp32` | Recommended to use `cuda fp16` |
| `args.MODEL_NAME` | Model file path | Complete path to model file | Only needs `.pth` model name, without file extension |

**Configuration Example:**

```python
args.strategy = 'cuda fp16'
args.MODEL_NAME = '/path/to/your/rwkv-model'
```

After configuration, run the following command in terminal to start the test script:

```bash
python rwkv-pip-test.py
```

After program startup, you can interact with the model through an interactive chat interface. After each round of dialogue, the terminal will display the model's inference speed and VRAM usage. Such as

```
────────────────────────────────────────────────────────────
[Current Generation]: 111 tokens | Time: 4.69s | Speed: 23.66 tokens/s
[Total Statistics]: 582 tokens | Average Speed: 22.48 tokens/s
[Current VRAM Usage]: 5.52GB/23.99GB (23.0%) | Cache: 5.75GB
GPU cache cleared
────────────────────────────────────────────────────────────
```

Please copy the performance data from the terminal, paste it into a new issue, and provide your **CPU and GPU model**.

>[!WARNING]
> Please record the performance data from the second or third round of dialogue to eliminate interference.

## llama.cpp

> ⚠️ TBD

## 🙏 Acknowledgments

Thanks to the following developers and projects for supporting this guide:

- [@BlinkDL](https://github.com/BlinkDL) - RWKV architecture author
- [@cryscan](https://github.com/cryscan) - Developer of [web-rwkv](https://github.com/cryscan/web-rwkv) project

Special thanks to all contributors in the RWKV open source community for enabling this excellent language model architecture to continuously develop and improve.

---

*This guide is continuously being updated. If you have any questions or suggestions, please feel free to submit an Issue or Pull Request.*