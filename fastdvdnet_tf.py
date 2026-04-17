"""
Definition of the FastDVDnet model in TensorFlow/Keras
Depthwise-separable version:
each Conv2d(3x3) is replaced by:
    depthwise 3x3 + pointwise 1x1
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ---------------------------------------------------------------------------
# Building-block layers
# ---------------------------------------------------------------------------

def ds_conv(x, out_ch, stride=1, name="dsconv"):
    """Depthwise 3x3 + Pointwise 1x1 (mirrors PyTorch DSConv)."""
    x = layers.DepthwiseConv2D(
        kernel_size=3,
        strides=stride,
        padding="same",
        use_bias=False,
        name=f"{name}_dw"
    )(x)
    x = layers.Conv2D(
        out_ch,
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=False,
        name=f"{name}_pw"
    )(x)
    return x


def cv_block(x, out_ch, name="cvblock"):
    """(DSConv => BN => ReLU) x 2"""
    x = ds_conv(x, out_ch, name=f"{name}_ds0")
    x = layers.BatchNormalization(name=f"{name}_bn0")(x)
    x = layers.ReLU(name=f"{name}_relu0")(x)
    x = ds_conv(x, out_ch, name=f"{name}_ds1")
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.ReLU(name=f"{name}_relu1")(x)
    return x


def input_cv_block(x, num_in_frames, out_ch, name="incvblock"):
    """
    Grouped Conv (num_in_frames groups) => BN => ReLU
    + DSConv => BN => ReLU
    
    TF doesn't support grouped Conv2D directly for arbitrary groups,
    so we split the input into per-frame chunks, apply the same Conv2D
    to each, then concatenate — which is equivalent to a grouped convolution.
    """
    interm_ch = 30
    in_ch_per_group = 3 + 1   # 3 colour + 1 noise channel per frame

    # Split channels into num_in_frames groups and apply the same conv
    splits = tf.split(x, num_or_size_splits=num_in_frames, axis=-1)
    group_outputs = []
    for i, split in enumerate(splits):
        g = layers.Conv2D(
            interm_ch,
            kernel_size=3,
            padding="same",
            use_bias=False,
            name=f"{name}_grouped_conv_{i}"
        )(split)
        group_outputs.append(g)
    x = layers.Concatenate(axis=-1, name=f"{name}_concat")(group_outputs)

    x = layers.BatchNormalization(name=f"{name}_bn0")(x)
    x = layers.ReLU(name=f"{name}_relu0")(x)

    x = ds_conv(x, out_ch, name=f"{name}_ds0")
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.ReLU(name=f"{name}_relu1")(x)
    return x


def down_block(x, out_ch, name="downblock"):
    """Downscale (stride-2 DSConv) + CvBlock"""
    x = ds_conv(x, out_ch, stride=2, name=f"{name}_ds")
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.ReLU(name=f"{name}_relu")(x)
    x = cv_block(x, out_ch, name=f"{name}_cv")
    return x


def up_block(x, out_ch, name="upblock"):
    """CvBlock + DSConv + PixelShuffle (depth-to-space)"""
    in_ch = x.shape[-1]
    x = cv_block(x, in_ch, name=f"{name}_cv")
    x = ds_conv(x, out_ch * 4, name=f"{name}_ds")
    # PixelShuffle equivalent: tf.nn.depth_to_space with block_size=2
    x = layers.Lambda(
        lambda t: tf.nn.depth_to_space(t, block_size=2),
        name=f"{name}_pixel_shuffle"
    )(x)
    return x


def output_cv_block(x, out_ch, name="outcvblock"):
    """DSConv => BN => ReLU => DSConv"""
    in_ch = x.shape[-1]
    x = ds_conv(x, in_ch, name=f"{name}_ds0")
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.ReLU(name=f"{name}_relu")(x)
    x = ds_conv(x, out_ch, name=f"{name}_ds1")
    return x


# ---------------------------------------------------------------------------
# DenBlock
# ---------------------------------------------------------------------------

def build_den_block(num_input_frames=3, name="denblock"):
    """
    Builds the denoising block as a Keras functional model.
    Inputs: in0, in1, in2  (each [B, H, W, 3])
            noise_map       ([B, H, W, 1])
    Output: denoised frame  ([B, H, W, 3])
    """
    chs_lyr0 = 32
    chs_lyr1 = 64
    chs_lyr2 = 128

    in0       = keras.Input(shape=(None, None, 3),  name="in0")
    in1       = keras.Input(shape=(None, None, 3),  name="in1")
    in2       = keras.Input(shape=(None, None, 3),  name="in2")
    noise_map = keras.Input(shape=(None, None, 1),  name="noise_map")

    # Concatenate as: [in0, noise, in1, noise, in2, noise]  →  channels-last
    x_cat = layers.Concatenate(axis=-1, name="input_cat")(
        [in0, noise_map, in1, noise_map, in2, noise_map]
    )

    x0 = input_cv_block(x_cat, num_input_frames, chs_lyr0, name="inc")

    x1 = down_block(x0, chs_lyr1, name="downc0")
    x2 = down_block(x1, chs_lyr2, name="downc1")

    x2 = up_block(x2, chs_lyr1, name="upc2")
    x1 = up_block(
        layers.Add(name="skip1")([x1, x2]),
        chs_lyr0,
        name="upc1"
    )

    residual = output_cv_block(
        layers.Add(name="skip0")([x0, x1]),
        3,
        name="outc"
    )

    out = layers.Subtract(name="output")([in1, residual])

    return keras.Model(inputs=[in0, in1, in2, noise_map], outputs=out, name=name)


# ---------------------------------------------------------------------------
# FastDVDnet
# ---------------------------------------------------------------------------

def build_fastdvdnet(num_input_frames=3):
    """
    Builds FastDVDnet as a Keras functional model.
    
    Inputs:
        frames:    [B, H, W, num_input_frames*3]  (frames concatenated on channel axis)
        noise_map: [B, H, W, 1]
    Output:
        denoised:  [B, H, W, 3]
    """
    frames    = keras.Input(shape=(None, None, num_input_frames * 3), name="frames")
    noise_map = keras.Input(shape=(None, None, 1), name="noise_map")

    # Split into individual frames (each 3 channels)
    frame_splits = [
        frames[:, :, :, 3*m : 3*m+3]
        for m in range(num_input_frames)
    ]
    # Use Lambda to avoid issues with symbolic tensor slicing
    in0 = layers.Lambda(lambda t: t[:, :, :, 0:3],  name="split_f0")(frames)
    in1 = layers.Lambda(lambda t: t[:, :, :, 3:6],  name="split_f1")(frames)
    in2 = layers.Lambda(lambda t: t[:, :, :, 6:9],  name="split_f2")(frames)

    den = build_den_block(num_input_frames=3, name="temp")
    out = den([in0, in1, in2, noise_map])

    model = keras.Model(inputs=[frames, noise_map], outputs=out, name="FastDVDnet")
    return model


# ---------------------------------------------------------------------------
# Main: build, summarise, and convert to TFLite
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # 1. Build and summarise
    # -----------------------------------------------------------------------
    model = build_fastdvdnet(num_input_frames=3)
    model.summary(expand_nested=True)

    # -----------------------------------------------------------------------
    # 2. Quick sanity-check forward pass
    # -----------------------------------------------------------------------
    BATCH, H, W = 1, 96, 96
    dummy_frames    = np.random.randn(BATCH, H, W, 9).astype(np.float32)
    dummy_noise_map = np.random.randn(BATCH, H, W, 1).astype(np.float32)

    out = model([dummy_frames, dummy_noise_map], training=False)
    print(f"\nForward pass OK — output shape: {out.shape}")  # (1, 96, 96, 3)

    # -----------------------------------------------------------------------
    # 3. Convert to TFLite
    #
    #    We use a concrete function with a *fixed* input shape so that the
    #    TFLite flatbuffer can be fully statically shaped (required for most
    #    on-device runtimes).  If you need dynamic shapes, see the
    #    "dynamic-shape" variant below.
    # -----------------------------------------------------------------------

    # -- 3a. Fixed-shape TFLite (recommended for deployment) ----------------
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, H, W, 9], dtype=tf.float32, name="frames"),
        tf.TensorSpec(shape=[1, H, W, 1], dtype=tf.float32, name="noise_map"),
    ])
    def serving_fn(frames, noise_map):
        return model([frames, noise_map], training=False)

    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [serving_fn.get_concrete_function()],
        trackable_obj=model
    )

    # Optional: enable optimisations (size + latency ↓, tiny accuracy drop)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    tflite_path = "fastdvdnet.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"\nTFLite model saved → {tflite_path}  "
          f"({len(tflite_model)/1024:.1f} KB)")

    # -- 3b. Dynamic-shape TFLite (uncomment if you need variable H/W) ------
    #
    # saved_model_dir = "fastdvdnet_saved_model"
    # model.export(saved_model_dir)          # TF 2.13+; use save() on older TF
    #
    # converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # tflite_model = converter.convert()
    # with open("fastdvdnet_dynamic.tflite", "wb") as f:
    #     f.write(tflite_model)

    # -----------------------------------------------------------------------
    # 4. Verify the TFLite model with the Interpreter
    # -----------------------------------------------------------------------
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("\nTFLite input tensors:")
    for d in input_details:
        print(f"  [{d['index']}] {d['name']:30s}  shape={d['shape']}  dtype={d['dtype']}")

    print("TFLite output tensors:")
    for d in output_details:
        print(f"  [{d['index']}] {d['name']:30s}  shape={d['shape']}  dtype={d['dtype']}")

    # Run one inference through the interpreter
    interpreter.set_tensor(input_details[0]['index'], dummy_frames)
    interpreter.set_tensor(input_details[1]['index'], dummy_noise_map)
    interpreter.invoke()

    tflite_out = interpreter.get_tensor(output_details[0]['index'])
    print(f"\nTFLite inference OK — output shape: {tflite_out.shape}")

    # Max absolute difference between Keras and TFLite outputs
    keras_out = model([dummy_frames, dummy_noise_map], training=False).numpy()
    max_diff = np.max(np.abs(keras_out - tflite_out))
    print(f"Max |Keras − TFLite| diff: {max_diff:.6f}  "
          f"({'OK' if max_diff < 1e-3 else 'WARNING: large diff'})")
