# Julia wrapper for Tensorboard

[![Build Status](https://travis-ci.org/zenna/Tensorboard.jl.svg?branch=master)](https://travis-ci.org/zenna/Tensorboard.jl)

[![codecov.io](http://codecov.io/github/zenna/Tensorboard.jl/coverage.svg?branch=master)](http://codecov.io/github/zenna/Tensorboard.jl?branch=master)

Tensorboard.jl is an interface to Tensorflow's Tensorboard.
Currently it is implemented as a wrapper over the Python library tensorboardX.

## Limitations

`add_image!` does not work, since it expects a PyTorch Tensor and not a numpy array, PRs are welcome!

## Installation 

Install [TensorboardX](https://github.com/lanpa/tensorboard-pytorch):

```python
pip install tensorboardX
```

or build from source:

```
pip install git+https://github.com/lanpa/tensorboard-pytorch
```

Install Tensorboard.jl from a julia repl

```julia
Pkg.clone("https://github.com/zenna/Tensorboard.jl")
```

## Example Usage

```julia
using Tensorboard
import MLDatasets: MNIST

writer = SummaryWriter()
sample_rate = 44100
freqs = [262, 294, 330, 349, 392, 440, 440, 440, 440, 440, 440]

# Fake resnet
resnetparams = Dict("W1" => rand(100, 100),
                    "W2" => randn(10, 5) * 10,
                    "b1" => ones(100) * 5)

for n_iter = 1:100
  dummy_s1 = rand(1)
  dummy_s2 = rand(1)
  # data grouping by `slash`
  add_scalar!(writer, "data/scalar1", dummy_s1[1], n_iter)
  add_scalar!(writer, "data/scalar2", dummy_s2[1], n_iter)
  add_scalars!(writer, "data/scalar_group", Dict("xsinx" => n_iter * sin(n_iter),
                                                 "xcosx" => n_iter * cos(n_iter),
                                                 "arctanx" => atan(n_iter)),
               n_iter)
  dummy_img = rand(3, 64, 64)
  if n_iter % 10 == 0
    # x = vutils.make_grid(dummy_img, normalize=true, scale_each=true)
    # add_image!(writer, "Image", dummy_img, n_iter)
    dummy_audio = [cos(freqs[div(n_iter, 10)] * pi * i / sample_rate) for i = 1:sample_rate * 2]
    
    add_audio!(writer, "myAudio", dummy_audio, n_iter, sample_rate=sample_rate)
    add_text!(writer, "Text", "text logged at step:" * string(n_iter), n_iter)

    for (name, param) in resnetparams
      param = param + n_iter
      add_histogram!(writer, name, param, n_iter)
    end

    # needs tensorboard 0.4RC or later
    add_pr_curve!(writer, "xoxo", rand(0:1, 100), rand(100), n_iter) 
  end
end

dataset, labels = MNIST.traindata()
images = float(permutedims(dataset[:, :, 1:100], (3, 1, 2)))
images = reshape(images, (100, 1, 28, 28))
label = labels[1:100]
features = reshape(images, (100, 784))
add_embedding!(writer, features, metadata=label, label_img=images)

# export scalar data to JSON for external processing
export_scalars_to_json(writer, "./all_scalars.json")
close(writer)
```

## Running TensorBoard

cd into the directory passed to `SummaryWriter` (by default this will be `runs`) and launch `tensorboard`

```bash
cd runs
tensorboard --logdir=./
```

point your browser to the the address shown
