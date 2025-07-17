module LocalLoader

using Images, FileIO

"""
    load_image(filepath::String) -> Matrix{RGB}

Loads an image and resizes it to 224x224 (for SqueezeNet).
"""
function load_image(filepath::String)
    img = load(filepath)
    return imresize(img, (224, 224))
end

"""
    load_dataset_from_folder(folder::String; limit::Int=10)

Loads a limited number of images from a folder.
Returns a vector of (img, filename) tuples.
"""
function load_dataset_from_folder(folder::String; limit::Int=10)
    files = filter(f -> any(endswith(f, ext) for ext in [".jpg", ".png"]), readdir(folder; join=true))
    selected = files[1:min(end, limit)]
    return [(load_image(f), basename(f)) for f in selected]
end

end
