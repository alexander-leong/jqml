using Metalhead
using Statistics
using Images, FileIO, ImageTransformations, Downloads
using JSON

# Download test image
img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Push_van_cat.jpg/320px-Push_van_cat.jpg"
img_path = "cat.jpg"
Downloads.download(img_url, img_path)

# Load and resize image
img = load(img_path)
img_resized = imresize(img, (224, 224))  # MobileNet expects 224x224

# Convert to Float32 and reshape
img_array = Float32.(channelview(img_resized))  # Shape: (3, 224, 224)
img_hwcn = permutedims(img_array, (2, 3, 1))     # Now: (224, 224, 3)
img_batch = reshape(img_hwcn, size(img_hwcn)..., 1)  # Add batch dim

@show size(img_batch)  # (224, 224, 3, 1)

# Load SqueezeNet model pretrained on ImageNet-1k
model = Metalhead.SqueezeNet(pretrain = true)

# Run inference
y_hat = model(img_batch)

# Extract predicted index (convert to 0-based)
ci = argmax(y_hat)            # CartesianIndex(i, 1)
pred_index = ci[1] - 1        # Convert to 0-based for PyTorch labels

println("Top class index: ", pred_index)
pred_label = imagenet_labels[pred_index]
println("Predicted label: ", pred_label)

# Download ImageNet labels
labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels_path = Downloads.download(labels_url)
labels = readlines(labels_path)

# Convert to Dict for reference (optional)
imagenet_labels = Dict(i => labels[i + 1] for i in 0:length(labels)-1)

# Show label
pred_label = imagenet_labels[pred_index]
println("Predicted label: ", pred_label)
