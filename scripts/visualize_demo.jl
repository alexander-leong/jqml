using Plots, Images, FileIO, Colors
using .JQML

# Load image
img = load("assets/cat.jpg")
img_resized = imresize(img, (224, 224))

img_rgb = RGB.(img_resized)
p1 = plot(img_rgb, title="Input Image", axis=false, ticks=nothing, size=(300, 300))

# Simulate classifier output (replace with real model inference)
probs = rand(10)
labels = JQML.ImageNetMapping.load_imagenet_labels()
subset_ids = [281, 282, 283, 284, 285, 207, 208, 817, 511, 656]
bar_labels = [labels[i] for i in subset_ids]

p2 = bar(bar_labels, probs,
    title="Predicted Probabilities (10-class subset)",
    xlabel="Class", ylabel="Probability",
    legend=false, rotation=45, size=(800, 300))

# Combine both
plot(p1, p2, layout = @layout([a; b]))
