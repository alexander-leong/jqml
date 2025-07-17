module TenClassClassifier

using Flux

function build_classifier(input_dim::Int)
    return Chain(
        Dense(input_dim, 128, relu),
        Dense(128, 10),
    )
end

end
