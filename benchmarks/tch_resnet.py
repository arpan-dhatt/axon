import torch
import torch.nn as nn

from benchmarker import benchmarker

if __name__ == "__main__":
    batch_sizes = [max(e, 1) for e in range(0, 129, 16)]
    image_sizes = [e for e in range(32, 128 + 1, 8)]
    print(batch_sizes, len(batch_sizes))
    print(image_sizes, len(image_sizes))

    TRIALS = 20
    INIT_WARMUP = 100
    WARMUP = 10
    NUM_LAYERS = 4

    def make_model(batch_size, image_size):
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False).to("mps")
        return model.half().eval()

    def make_input(batch_size, image_size):
        return (torch.normal(0.0, 1.0,
                            size=(batch_size, 3, image_size, image_size), dtype=torch.float16)
                .to("mps"))

    benchmarker(("Batch Size", batch_sizes), ("Image Size", image_sizes),
                make_model, make_input, "resnet.torch.gpu.npz", sem="tch",
                trace=[(1, 64), (64, 64)])