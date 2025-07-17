module JQML

# === Include module files ===
include("models/squeezenet_extractor.jl")
include("classifiers/classifier_10class.jl")
include("utils/imagenet_mapping.jl")
include("data/local_loader.jl")

# === Use the submodules ===
using .SqueezeNetExtractor
using .TenClassClassifier
using .ImageNetMapping
using .LocalLoader

# === Export the submodules (optional, for cleaner usage) ===
export SqueezeNetExtractor, TenClassClassifier, ImageNetMapping, LocalLoader

end