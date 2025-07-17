push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using .JQML
using Flux, Statistics
using Printf

function accuracy(model, extractor, data)
    correct = 0
    total = 0

    for (img, label) in data
        x = Flux.flatten(extractor(reshape(img, 224, 224, 3, 1)))
        y_pred = model(x)
        pred_class = argmax(y_pred)

        if pred_class == label + 1  # onehot labels are 0-based, argmax is 1-based
            correct += 1
        end

        total += 1
    end

    return correct / total
end


# === Load modules
extractor = SqueezeNetExtractor.get_feature_extractor()  # frozen
train_data = MNISTLoader.get_mnist_batch(100)

# === Build classifier model
sample_input = Flux.flatten(extractor(reshape(train_data[1][1], 224, 224, 3, 1)))
classifier = TenClassClassifier.build_classifier(size(sample_input, 1))

# === Define loss function (model-aware)
loss_fn(model, x, y) = Flux.crossentropy(model(x), y)

# === Setup optimizer state for classifier
opt = Flux.ADAM(1e-3)
state = Flux.setup(opt, classifier)

# === Training loop
epochs = 5
@info "Starting training..."

for epoch in 1:epochs
    losses = Float32[]

    for (img, label) in train_data
        x = Flux.flatten(extractor(reshape(img, 224, 224, 3, 1)))  # frozen FE
        y = Flux.onehot(label, 0:9)

        # ⬇️ Modern pattern
        loss, grads = Flux.withgradient(classifier) do m
            loss_fn(m, x, y)
        end

        Flux.update!(state, classifier, grads[1])  # grads is a tuple: (grads, _context)
        push!(losses, loss)
    end

    avg_loss = mean(losses)
    train_acc = accuracy(classifier, extractor, train_data)

    @printf "Epoch %d - Avg loss: %.4f | Accuracy: %.2f%%\n" epoch avg_loss 100 * train_acc
end

@info "Done training."
