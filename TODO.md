# TODO – jqmL: Classical + Quantum Feature-based Classification

## ✅ Completed
- [x] Modular project structure under `src/`
- [x] Pretrained SqueezeNet feature extractor
- [x] Custom 10-class classifier
- [x] ImageNet label mapping module
- [x] Visual demo with input image + class probability bar chart
- [x] Fixed image plotting issue with RGB conversion
- [x] Moved scripts/ outside src/

---

## 🧪 In Progress
- [ ] Evaluate 10-class classifier on sample images
- [ ] Add more demo images to `assets/`
- [ ] Load local image dataset subset (for test loop)

---

## 🧠 Next Steps – Quantum Integration (Yao.jl)
- [ ] Create `src/classifiers/quantum_interface.jl`
  - [ ] Function to extract small patch (e.g., `Vector{Float32}` of 4–8 dims)
  - [ ] Placeholder `run_quantum_classifier(x::Vector)` function using Yao.jl
- [ ] Add hybrid inference script `scripts/qml_inference.jl`
- [ ] Compare classical vs quantum predictions

---

## 📊 Visual & Reporting
- [ ] Add Pluto notebook for interactive demo
- [ ] Add evaluation script: `scripts/eval_qml_vs_classical.jl`
- [ ] Add ROC, accuracy comparison plot (Flux.jl + Plots)

---

## 🧰 Infrastructure
- [ ] Create `test/` folder with unit tests for all modules
- [ ] Add pre-trained weights export (optional, for reproducibility)
- [ ] Update `README.md` with architecture + usage examples
- [ ] Add LICENSE and CITATION file (optional)
