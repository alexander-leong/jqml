push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using .JQML
using Flux, Statistics
using Printf

# Load modules
extractor = SqueezeNetExtractor.get_feature_extractor()
labels = MNISTLoader.get_mnist_batch(5)  # Load 5 MNIST samples

# Prepare classifier
# Feature shape: (13, 13, 512) → flattened = 86592
sample_input = Flux.flatten(extractor(reshape(labels[1][1], 224, 224, 3, 1)))
classifier = TenClassClassifier.build_classifier(size(sample_input, 1))

println("Running MNIST inference...\n")

for (i, (img, true_label)) in enumerate(labels)
    img_batch = reshape(img, 224, 224, 3, 1)  # Add batch dim
    features = extractor(img_batch)
    flat_features = Flux.flatten(features)
    ŷ = classifier(flat_features)
    
    pred_index = argmax(ŷ)[1]  # 1-based
    probs = round.(ŷ; digits=3)

    @printf "[%d] True: %d → Predicted: %d\nProbabilities: %s\n\n" i true_label pred_index probs
end
