using JQML.SqueezeNetExtractor
using JQML.TenClassClassifier
using JQML.ImageNetMapping
using Images, FileIO, ImageTransformations, Flux

# Load image
img = load("assets/cat.jpg")
img_resized = imresize(img, (224, 224))
img_array = Float32.(channelview(img_resized))
img_hwcn = permutedims(img_array, (2,3,1))
img_batch = reshape(img_hwcn, size(img_hwcn)..., 1)

# Feature extraction
extractor = SqueezeNetExtractor.get_feature_extractor()
features = extractor(img_batch)
flat_features = Flux.flatten(features)

# Classifier
classifier = TenClassClassifier.build_classifier(size(flat_features, 1))
y_hat = classifier(flat_features)

# Labels
labels = ImageNetMapping.load_imagenet_labels()
subset_ids = [281, 282, 283, 284, 285, 207, 208, 817, 511, 656]
top_idx = argmax(y_hat)[1]
predicted_label = labels[subset_ids[top_idx]]

println("Predicted: ", predicted_label)