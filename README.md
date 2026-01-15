# 🔥 vLLMulator

A calculator for determining parallelism requirements when running large language models with [vLLM](https://github.com/vllm-project/vllm).

## 🚀 Features

- **Model Configuration Calculator**: Determine optimal tensor parallel (TP) and pipeline parallel (PP) settings
- **Memory Estimation**: Calculate VRAM requirements for different model configurations
- **Hardware Compatibility**: Check which GPU configurations can run specific models
- **Extensive Model Database**: Pre-configured with popular models from major providers (Meta, Mistral, Qwen, DeepSeek, etc.)
- **Real-time Updates**: Model database is regularly updated with the latest releases

## 🎯 Use Cases

- **Infrastructure Planning**: Determine hardware requirements before deploying models
- **Cost Optimization**: Find the most efficient GPU configuration for your workload
- **Performance Tuning**: Optimize parallelism settings for maximum throughput
- **Model Selection**: Compare memory requirements across different model sizes

## 🌐 Live Demo

Try it out: **[https://vanpelt.github.io/vllmulator/](https://vanpelt.github.io/vllmulator/)**

## 🛠️ Local Development

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/vanpelt/vllmulator.git
   cd vllmulator/houston
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

3. Update model database (optional):
   ```bash
   uv run python enrich_models.py
   ```

4. Serve locally:
   ```bash
   python -m http.server 8000
   ```

   Open http://localhost:8000 in your browser.

## 📊 How It Works

The calculator uses model metadata from Hugging Face to determine:

- **Parameter count** and **model architecture**
- **Memory requirements** based on precision (FP16, BF16, etc.)
- **Optimal parallelism configurations** for different GPU setups
- **Attention mechanisms** (GQA, MLA, etc.) that affect memory usage

## 🤝 Contributing

Contributions welcome! The model database is automatically updated, but feel free to:

- Report bugs or suggest features via GitHub Issues
- Submit pull requests for improvements
- Add support for new model architectures

## ⚠️ Disclaimer

This tool provides estimates based on available model metadata. Actual memory usage may vary depending on:
- Specific vLLM version and configuration
- Additional overhead from attention mechanisms
- Runtime optimizations and caching

Always test with your specific setup before production deployment.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.