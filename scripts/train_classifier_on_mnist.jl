push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using .JQML
using Flux, Statistics
using Printf

function val_metrics(model, extractor, data, loss_fn)
    total_loss = 0.0
    correct = 0
    total = 0

    for (img, label) in data
        x = Flux.flatten(extractor(reshape(img, 224, 224, 3, 1)))
        y = Flux.onehot(label, 0:9)

        y_pred = model(x)
        total_loss += loss_fn(model, x, y)
        pred_class = argmax(y_pred)

        if pred_class == label + 1
            correct += 1
        end

        total += 1
    end

    return total_loss / total, correct / total
end


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

split_idx = Int(floor(0.8 * length(train_data)))
train_set = train_data[1:split_idx]
val_set = train_data[split_idx+1:end]

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
    val_loss, val_acc = val_metrics(classifier, extractor, val_set, loss_fn)

    @printf "Epoch %d - Train Loss: %.4f | Train Acc: %.2f%% | Val Loss: %.4f | Val Acc: %.2f%%\n" epoch avg_loss 100*train_acc val_loss 100*val_acc

end

@info "Done training."
