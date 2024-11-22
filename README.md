# Nexa-SDK-Enterprise

Enterprise SDK for Nexa AI's model serving on edge devices.

## Python Package Installation

We have released pre-built wheels for various Python versions, platforms, and backends for convenient installation on our [index page](https://nexaai.github.io/nexa-sdk/whl/).

### CPU Installation

```bash
pip install nexa-enterprise --prefer-binary --index-url https://nexaai.github.io/nexa-sdk-enterprise/whl/cpu --extra-index-url https://pypi.org/simple --no-cache-dir
```

### GPU Installation (Vulkan)

To install with Vulkan support, make sure you have [Vulkan SDK 1.3.261.1 or later](https://vulkan.lunarg.com/sdk/home) installed.

For **Windows PowerShell**:

```bash
$env:CMAKE_ARGS="-DGGML_VULKAN=on"; pip install nexa-enterprise --prefer-binary --index-url https://nexaai.github.io/nexa-sdk-enterprise/whl/vulkan --extra-index-url https://pypi.org/simple --no-cache-dir
```

For **Windows Command Prompt**:

```bash
set CMAKE_ARGS="-DGGML_VULKAN=on" & pip install nexa-enterprise --prefer-binary --index-url https://nexaai.github.io/nexa-sdk-enterprise/whl/vulkan --extra-index-url https://pypi.org/simple --no-cache-dir
```

For **Windows Git Bash**:

```bash
CMAKE_ARGS="-DGGML_VULKAN=on" pip install nexa-enterprise --prefer-binary --index-url https://nexaai.github.io/nexa-sdk-enterprise/whl/vulkan --extra-index-url https://pypi.org/simple --no-cache-dir
```

### Local Build (You may need to acquire access to this private repo first.)

How to clone this repo:

```bash
git clone --recursive https://github.com/NexaAI/nexa-sdk-enterprise
```

If you forget to use `--recursive`, you can use below command to add submodule:

```bash
git submodule update --init --recursive
```

Then you can build and install the package:

```bash
pip install -e .
```

## Unit Tests

```
python -m tests.test_text_generation
python -m tests.test_disk_kv_cache
```
