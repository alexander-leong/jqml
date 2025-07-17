push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using .JQML
using Plots, Colors

# Get one sample
data = MNISTLoader.get_mnist_batch(1)
img, label = data[1]

# Convert to Matrix{RGB} for Plots.jl
img_rgb = colorview(RGB, permutedims(img, (3,1,2)))

plot(img_rgb, axis=false, title="MNIST - Label: $label")