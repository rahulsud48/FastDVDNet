"""
Convert an NCHW ONNX model to NHWC format.

Usage:
    python onnx_to_nhwc.py --input model.onnx
    python onnx_to_nhwc.py --input model.onnx --output model_nhwc.onnx
"""

import argparse
import numpy as np
import onnx
import onnxruntime as ort


def insert_nhwc_transposes(
    onnx_path: str,
    out_path:  str = None,
) -> str:
    import onnx.helper as oh
    from onnx import TensorProto

    if out_path is None:
        out_path = onnx_path.replace(".onnx", "_nhwc.onnx")

    model = onnx.load(onnx_path)
    graph = model.graph

    new_nodes  = []
    new_inputs = []
    new_outputs = []

    for inp in graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        if len(shape) == 4:
            nhwc_name = inp.name + "_nhwc_in"
            nhwc_shape = [shape[0], shape[2], shape[3], shape[1]]
            new_type = oh.make_tensor_type_proto(
                TensorProto.FLOAT,
                nhwc_shape if all(s > 0 for s in nhwc_shape) else [None] * 4
            )
            new_inputs.append(oh.make_value_info(nhwc_name, new_type))
            new_nodes.append(oh.make_node(
                "Transpose",
                inputs=[nhwc_name],
                outputs=[inp.name],
                perm=[0, 3, 1, 2],
                name=f"pre_transpose_{inp.name}",
            ))
        else:
            new_inputs.append(inp)

    for out in graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        if len(shape) == 4:
            nhwc_name = out.name + "_nhwc"
            nhwc_shape = [shape[0], shape[2], shape[3], shape[1]]
            new_type = oh.make_tensor_type_proto(
                TensorProto.FLOAT,
                nhwc_shape if all(s > 0 for s in nhwc_shape) else [None] * 4
            )
            new_nodes.append(oh.make_node(
                "Transpose",
                inputs=[out.name],
                outputs=[nhwc_name],
                perm=[0, 2, 3, 1],
                name=f"post_transpose_{out.name}",
            ))
            new_outputs.append(oh.make_value_info(nhwc_name, new_type))
        else:
            new_outputs.append(out)

    pre_transposes  = [n for n in new_nodes if n.name.startswith("pre_")]
    post_transposes = [n for n in new_nodes if n.name.startswith("post_")]

    new_graph = oh.make_graph(
        nodes=pre_transposes + list(graph.node) + post_transposes,
        name=graph.name,
        inputs=new_inputs,
        outputs=new_outputs,
        initializer=list(graph.initializer),
    )
    new_model = oh.make_model(new_graph, opset_imports=model.opset_import)
    new_model.ir_version = model.ir_version

    onnx.checker.check_model(new_model)
    onnx.save(new_model, out_path)
    print(f"NHWC ONNX saved → {out_path}")
    return out_path


def validate(original: str, nhwc: str, h: int = 96, w: int = 96):
    sess_nchw = ort.InferenceSession(original, providers=["CPUExecutionProvider"])
    sess_nhwc = ort.InferenceSession(nhwc,     providers=["CPUExecutionProvider"])

    np.random.seed(42)
    feed_nchw = {}
    feed_nhwc = {}
    for inp in sess_nchw.get_inputs():
        shape = [d if (d and d > 0) else 1 for d in inp.shape]
        # restore spatial dims from args
        shape[2], shape[3] = h, w
        arr = np.random.randn(*shape).astype(np.float32)
        feed_nchw[inp.name] = arr
        feed_nhwc[inp.name + "_nhwc_in"] = arr.transpose(0, 2, 3, 1)

    out_nchw = sess_nchw.run(None, feed_nchw)[0]
    out_nhwc = sess_nhwc.run(None, feed_nhwc)[0].transpose(0, 3, 1, 2)

    max_diff = float(np.max(np.abs(out_nchw - out_nhwc)))
    print(f"Max |NCHW − NHWC| diff: {max_diff:.2e}  "
          f"({'OK' if max_diff < 1e-4 else 'WARNING'})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",  "-i", required=True, help="Input NCHW .onnx file")
    p.add_argument("--output", "-o", default=None,  help="Output NHWC .onnx file (default: <input>_nhwc.onnx)")
    p.add_argument("--no-validate", action="store_true", help="Skip validation")
    args = p.parse_args()

    nhwc_path = insert_nhwc_transposes(args.input, args.output)

    if not args.no_validate:
        validate(args.input, nhwc_path)
