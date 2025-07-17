# jqmL 🧠⚛️

A modular Julia project for hybrid **Classical + Quantum Machine Learning**, featuring a pretrained **SqueezeNet** feature extractor, a custom **10-class classifier**, and integration-ready quantum interfaces via **Yao.jl**.

---

## 📦 Project Overview

This project extracts deep visual features from images using a pretrained convolutional neural network (SqueezeNet), and feeds them into a custom classifier — which can be either classical (Flux) or quantum-based (Yao.jl).

It's designed for experimentation, benchmarking, and eventual deployment in **hybrid quantum-classical classification pipelines**.

---

## 🚀 Getting Started

### 1. Clone the project
```bash
git clone https://github.com/yourusername/jqml.git
cd jqml
```

### 2. Launch Julia with this environment
```bash
julia --project
```

### 3. Install dependencies
```julia
using Pkg
Pkg.instantiate()
```

### 4. Run demo script
```julia
include("scripts/visualize_demo.jl")
```

---

## 🧱 Project Structure

```bash
jqmL/
├── src/                     # Source modules
│   ├── JQML.jl              # Main module file
│   ├── models/              # Feature extractor (SqueezeNet)
│   ├── classifiers/         # Classical and quantum classifier builders
│   ├── utils/               # ImageNet label mapping
├── scripts/                 # Demo and inference scripts
│   ├── visualize_demo.jl
│   └── squeezenet_inference.jl
├── assets/                  # Sample image assets
│   └── cat.jpg
├── TODO.md                  # Roadmap and next steps
├── Project.toml
├── Manifest.toml
└── README.md
```

---

## 🧠 Features

- ✅ Pretrained SqueezeNet feature extractor
- ✅ Modular classifier with Flux.jl
- ✅ 10-class ImageNet subset classification
- ✅ Bar chart + image visualization demo
- ✅ Ready to integrate quantum circuits via Yao.jl
- ✅ Designed for hybrid experiments and extensibility

---

## ⚛️ Quantum Integration (coming soon)

The next step is to integrate a **quantum classifier** using [Yao.jl](https://github.com/QuantumBFS/Yao.jl), allowing for a hybrid model that performs classification on reduced feature vectors extracted from SqueezeNet.

Stay tuned for:
- `quantum_interface.jl` module
- `qml_inference.jl` comparison script
- Full QML vs Classical performance benchmark

---

## 📊 Demo Preview

![demo-plot](docs/demo_example.png) <!-- Optional if you add screenshots -->

---

## 📜 License

MIT License – see [LICENSE](./LICENSE)

---

## 🤝 Contributing

Want to help? Feel free to fork this repo, suggest improvements, or contribute quantum classifier designs. PRs are welcome!

---

## ✨ Acknowledgments

- [`Metalhead.jl`](https://github.com/FluxML/Metalhead.jl) for pretrained models  
- [`Flux.jl`](https://fluxml.ai) for neural network layers  
- [`Yao.jl`](https://yaoquantum.org/) for quantum computing in Julia
