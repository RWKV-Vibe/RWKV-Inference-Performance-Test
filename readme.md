<div align="center">

<h1> RWKV 推理性能测试指南 </h1>

[![中文](https://img.shields.io/badge/Language-中文-orange.svg)](./README.md)
[![English](https://img.shields.io/badge/Language-English-blue.svg)](./README_EN.md)

</div>

本指南将帮助您在本地设备上配置并测试 RWKV 模型的推理性能，我们也欢迎您提交 issue 报告 RWKV 推理性能数据。

我们提供了基于三种不同推理工具的测试方法：[web-rwkv 测试](#web-rwkv) | [RWKV pip 测试](#rwkv-pip) | [llama.cpp 测试](#llamacpp)。

|测试方法|需要的模型格式|支持的显卡类型|
|---|---|---|
|web-rwkv|`.st`|支持 vulkan 的所有显卡，包括核显|
|RWKV pip|`.pth`|支持 CUDA 的 NVIDIA 显卡，虽然有 CPU 模式，但不建议测试|
|llama.cpp|`.gguf` | 所有类型的显卡，包括核显和 CPU |


在开始之前，请确保您具备以下条件：

- 系统具备足够的存储空间用于下载模型文件
- 具备基本的命令行操作能力
- 已安装 Python 环境（测试二需要）

## web-rwkv

### 测试准备

1. 下载 web-rwkv 工具：访问 [web-rwkv releases](https://github.com/cryscan/web-rwkv/releases) 页面，下载适合您操作系统的最新版本压缩包，在一个空白目录中解压
2. 获取 RWKV7-G1 2.9B 模型：[点击下载](https://huggingface.co/shoumenchougou/RWKV-ST-model/resolve/main/rwkv7-g1-2.9b-20250519-ctx4096.st?download=true) `rwkv7-g1-2.9b-20250519-ctx4096.st` RWKV 模型文件，用于性能测试
3. 将下载的模型文件移动到 web-rwkv 解压目录下的 `dist` 文件夹中

### 推理性能测试

在 `web-rwkv/dist` 目录下，右键选择"在集成终端中打开"。然后分别执行入以下指令，进行不同量化精度的测试：

1. fp16 精度推理性能测试：

```bash
./bench.exe --model "./rwkv7-g1-2.9b-20250519-ctx4096.st"
```
2. INT8 量化推理性能测试：
```bash
./bench.exe --model "./rwkv7-g1-2.9b-20250519-ctx4096.st" --quant 31
```
3. NF4 量化推理性能测试
```bash
./bench.exe --model "./rwkv7-g1-2.9b-20250519-ctx4096.st" --quant-nf4 31
```

>[!WARNING]
> `--quant` 和 `--quant-nf4` 是量化层数，推荐保持默认值 `31`

**移动光标并选择您的推理设备（推荐使用默认的 `vulkan` 后端）**

![web-rwkv-result](./img/web-rwkv-seclet-adapter.png)

测试完成后，终端将输出如下格式的性能报告：

| model                                                    | quant_int8 | quant_float4 |    test |            t/s |
|----------------------------------------------------------|-----------:|-------------:|--------:|---------------:|
| rwkv7-g1-2.9b-20250519-ctx4096.st                        |          0 |            0 |   pp512 |        1022.89 |
| rwkv7-g1-2.9b-20250519-ctx4096.st                        |          0 |            0 |   tg128 |          95.98 |

其中 **t/s** 表示推理速度（tokens/秒），请将从终端复制此表格，将其粘贴到新的 issue 中，并提供您的 **CPU 和 GPU 型号**。

---

## RWKV pip 

通过 Python 代码调用 [RWKV pip 仓库](https://pypi.org/project/rwkv/)进行推理，以测试性能数据。

要基于 RWKV pip 测试，需要提前下载一个 `.pth` 格式的 RWKV7-G1 2.9B 模型：

- [魔搭平台下载](https://modelscope.cn/models/RWKV/rwkv7-g1/resolve/master/rwkv7-g1-2.9b-20250519-ctx4096.pth)
- [Hugging Face 下载](https://huggingface.co/BlinkDL/rwkv7-g1/resolve/main/rwkv7-g1-2.9b-20250519-ctx4096.pth?download=true)

### 准备测试环境

> [!TIP]
> 推荐使用 [AnaConda](https://anaconda.org/anaconda/conda) 管理 Python 环境

运行以下命令新建一个 conda 环境，安装必要的 Python 环境，然后克隆此仓库：

```bash
conda create -n rwkv-pip-test python=3.12
conda activate rwkv-pip-test
pip install torch --upgrade --extra-index-url https://download.pytorch.org/whl/cu128
pip install rwkv psutil prompt_toolkit tokenizers
git clone https://github.com/ehooon/RWKV-Inference-Performance-Test.git
```

### 测试推理性能

打开 `rwkv-pip-test.py` 文件，编辑中的关键参数：

| 参数名称 | 功能描述 | 可选值 | 说明 |
|---------|---------|--------|------|
| `args.strategy` | 运行设备和精度 | `cuda fp16`<br>`cpu fp16`<br>`cuda fp32` | 推荐使用 `cuda fp16`  |
| `args.MODEL_NAME` | 模型文件路径 | 模型文件的完整路径 | 仅需要`.pth` 模型名称，不需要文件后缀 |

**配置示例：**

```python
args.strategy = 'cuda fp16'
args.MODEL_NAME = '/path/to/your/rwkv-model'
```

配置完成后，在终端中运行以下命令启动测试脚本：

```bash
python rwkv-pip-test.py
``` 

程序启动后，您可以通过交互式聊天界面与模型对话。每轮对话过后，终端会显示模型的响应速度和显存占用。

```
────────────────────────────────────────────────────────────
[Current Generation]: 111 tokens | Time: 4.69s | Speed: 23.66 tokens/s
[Total Statistics]: 582 tokens | Average Speed: 22.48 tokens/s
[Current VRAM Usage]: 5.52GB/23.99GB (23.0%) | Cache: 5.75GB
GPU cache cleared
────────────────────────────────────────────────────────────
```

请将从终端复制性能数据，将其粘贴到新的 issue 中，并提供您的 **CPU 和 GPU 型号**。

>[!WARNING]
> 请记录第二轮或第三轮对话的性能数据，以排除干扰。

## llama.cpp

> ⚠️ TBD 

## 🙏 致谢

感谢以下开发者和项目为本指南提供的支持：

- [@BlinkDL](https://github.com/BlinkDL) - RWKV 架构作者
- [@cryscan](https://github.com/cryscan) - [web-rwkv](https://github.com/cryscan/web-rwkv) 项目的开发者

特别感谢 RWKV 开源社区的所有贡献者，让这个优秀的语言模型架构得以不断发展和完善。

---

*本指南持续更新中，如有问题或建议，欢迎提交 Issue 或 Pull Request。*