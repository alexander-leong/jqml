using MLDatasets
using Images, ImageTransformations, Plots

# Load MNIST data
images, labels = MNIST.traindata()

# 1. Get image and label
i = 15  # or any index from 1 to 60_000
img_gray = images[:, :, i]
label = labels[i]

# 2. Fix orientation (transpose image)
img_corrected = permutedims(img_gray, (2, 1))  # swap axes

# 3. Create fake RGB and resize
img_rgb = cat(img_corrected, img_corrected, img_corrected; dims=3)
img_resized = imresize(img_rgb, (224, 224))

# 4. Convert for plotting
img_plot = colorview(RGB, permutedims(img_resized, (3,1,2)))

# 5. Plot
plot(img_plot, axis=false, title="MNIST sample - Label: $label")