module MNISTLoader

using MLDatasets
using ImageTransformations

"""
    preprocess_mnist_image(img::AbstractMatrix{<:Number})::Array{Float32, 3}

Takes a grayscale 28×28 image, transposes it, converts to fake RGB, and resizes to (224, 224, 3).
"""
function preprocess_mnist_image(img)
    img_corrected = permutedims(img, (2, 1))
    img_rgb = cat(img_corrected, img_corrected, img_corrected; dims=3)
    return imresize(img_rgb, (224, 224))
end

"""
    get_mnist_batch(n::Int=10)

Returns a vector of (image, label) pairs, where each image is a (224, 224, 3) Array{Float32, 3}.
"""
function get_mnist_batch(n::Int=10)
    x, y = MNIST.traindata()
    return [(preprocess_mnist_image(x[:, :, i]), y[i]) for i in 1:n]
end

end # module
