"Wrapper of TensorboardX"
module TensorboardX
using PyCall
@pyimport tensorboardX

"Summary Writer"
struct SummaryWriter
  pyobj::PyObject
  function SummaryWriter(log_dir=nothing, comment="")
    new(tensorboardX.SummaryWriter(log_dir, comment))
  end
end

add_scalar(s::SummaryWriter, args...) = s.pyobj["add_scalar"](args...)
add_histogram(s::SummaryWriter, args...) = s.pyobj["add_histogram"](args...)
add_histogram(s::SummaryWriter, args...) = s.pyobj["add_histogram"](args...)

end