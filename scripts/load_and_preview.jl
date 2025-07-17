push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using .JQML
using Plots

data = LocalLoader.load_dataset_from_folder("assets", limit=4)

# Show the first image
img, name = data[1]
plot(RGB.(img), axis=false, title=name)