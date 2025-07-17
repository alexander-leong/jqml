module SqueezeNetExtractor

using Metalhead, Flux

function get_feature_extractor()
    model = Metalhead.SqueezeNet(pretrain=true)
    return Chain(model.layers[1:end-1]...)
end

end