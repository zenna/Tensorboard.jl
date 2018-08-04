"Wrapper of TensorboardX"
module Tensorboard
using PyCall
@pyimport tensorboardX

"Summary Writer"
struct SummaryWriter
  pyobj::PyObject
  function SummaryWriter(log_dir=nothing, comment="")
    new(tensorboardX.SummaryWriter(log_dir, comment))
  end
end

add_scalar!(s::SummaryWriter, args...; kwargs...) = s.pyobj["add_scalar"](args...; kwargs...)
add_scalars!(s::SummaryWriter, args...; kwargs...) = s.pyobj["add_scalars"](args...; kwargs...)
add_histogram!(s::SummaryWriter, args...; kwargs...) = s.pyobj["add_histogram"](args...; kwargs...)
add_text!(s::SummaryWriter, args...; kwargs...) = s.pyobj["add_text"](args...; kwargs...)
add_image!(s::SummaryWriter, args...; kwargs...) = s.pyobj["add_image"](args...; kwargs...)
add_audio!(s::SummaryWriter, args...; kwargs...) = s.pyobj["add_audio"](args...; kwargs...)
add_pr_curve!(s::SummaryWriter, args...; kwargs...) = s.pyobj["add_pr_curve"](args...; kwargs...)
add_embedding!(s::SummaryWriter, args...; kwargs...) = s.pyobj["add_embedding"](args...; kwargs...)
Base.close(s::SummaryWriter, args...; kwargs...) = s.pyobj["close"](args...; kwargs...)
export_scalars_to_json(s::SummaryWriter, args...; kwargs...) = s.pyobj["export_scalars_to_json"](args...; kwargs...)

export SummaryWriter,
       add_scalar!,
       add_scalars!,
       add_histogram!,
       add_image!,
       add_audio!,
       add_text!,
       add_pr_curve!,
       close,
       export_scalars_to_json,
       add_embedding!

end
