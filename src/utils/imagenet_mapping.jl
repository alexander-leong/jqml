module ImageNetMapping

using Downloads

"""
    load_imagenet_labels() -> Dict{Int, String}

Downloads and returns a mapping of ImageNet class indices to class names.
"""
function load_imagenet_labels()
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    lines = readlines(Downloads.download(url))
    return Dict(i => lines[i + 1] for i in 0:length(lines)-1)
end

end # module
