# SqueezeNet Feature Extractor + Custom 10-Class Classifier (Transfer Learning Demo)
using Metalhead
using Flux
using Statistics
using Images, FileIO, ImageTransformations, Downloads
using JSON

# ----------------------------
# 1. Load and Preprocess Image
# ----------------------------
img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Push_van_cat.jpg/320px-Push_van_cat.jpg"
img_path = "cat.jpg"
Downloads.download(img_url, img_path)

img = load(img_path)
img_resized = imresize(img, (224, 224))  # SqueezeNet expects 224x224
img_array = Float32.(channelview(img_resized))         # (3, 224, 224)
img_hwcn = permutedims(img_array, (2, 3, 1))            # (224, 224, 3)
img_batch = reshape(img_hwcn, size(img_hwcn)..., 1)     # (224, 224, 3, 1)

# ----------------------------
# 2. Load Pretrained SqueezeNet and Extract Features
# ----------------------------
model = Metalhead.SqueezeNet(pretrain = true)
feature_extractor = model.layers[1:end-1]
extractor_model = Chain(feature_extractor...)

# Forward pass to extract features
features = extractor_model(img_batch)  # (13, 13, 512, 1)
flat_features = Flux.flatten(features) # (86592, 1)

# ----------------------------
# 3. Define Lightweight Classifier (10-Class)
# ----------------------------
classifier = Chain(
    Dense(size(flat_features, 1), 128, relu),
    Dense(128, 10),
    softmax,
)

# ----------------------------
# 4. Forward Pass Through Classifier
# ----------------------------
y_hat = classifier(flat_features)
@show size(y_hat)
@show y_hat

# ----------------------------
# 5. Define Subset of ImageNet Classes
# ----------------------------
imagenet_subset_ids = [281, 282, 283, 284, 285, 207, 208, 817, 511, 656]  # e.g. cat, dog, car, etc.

# Load label mapping
labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
all_labels = readlines(Downloads.download(labels_url))
imagenet_labels = Dict(i => all_labels[i+1] for i in 0:length(all_labels)-1)

# Map prediction to label
top_idx = argmax(y_hat)[1]
predicted_class = imagenet_subset_ids[top_idx]
println("Predicted class index: ", predicted_class)
println("Predicted label: ", imagenet_labels[predicted_class])
