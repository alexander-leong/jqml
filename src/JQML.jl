module JQML

include("models/squeezenet_extractor.jl")
include("classifiers/classifier_10class.jl")
include("utils/imagenet_mapping.jl")

# Explicitly import submodules so you can use `JQML.SqueezeNetExtractor`
using .SqueezeNetExtractor
using .TenClassClassifier
using .ImageNetMapping

# Optional: export names to avoid typing `JQML.SqueezeNetExtractor` in scripts
export SqueezeNetExtractor, TenClassClassifier, ImageNetMapping

end