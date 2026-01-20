# ROS Package [bob_flux2k](https://github.com/bob-ros2/bob_flux2k)

ROS 2 node for the **FLUX.2-klein** text-to-image and image-to-image models from Black Forest Labs.

## Supported Models

This node supports both the 4B and 9B variants of the FLUX.2-klein family:
- **[FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)**: Optimized for speed and lower VRAM usage (Apache 2.0 license).
- **[FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)**: Higher quality and better prompt following (requires ~22GB VRAM, fits on RTX 4090).

> [!IMPORTANT]
> To use the **9B model**, you must manually visit the [model page](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B), log in to Hugging Face, and **agree to the terms and non-commercial license**. You will also need to be logged in via `huggingface-cli login` on your machine for the node to download the weights.

You can switch models via the `repo_id` ROS parameter or the `FLUX2K_REPO_ID` environment variable.

## Usage

### Run the node
You can run the node directly or provide a configuration file.

**Basic run:**
```bash
ros2 run bob_flux2k tti --ros-args -p repo_id:=black-forest-labs/FLUX.2-klein-4B
```

**Using a configuration file:**
```bash
ros2 run bob_flux2k tti --ros-args --params-file src/bob_flux2k/config/tti.yaml
```

### Send a prompt
```bash
ros2 topic pub /prompt std_msgs/msg/String "{data: 'A futuristic city in the style of cyberpunk'}" --once
```

## Parameters

### Dynamic Parameters
These can be adjusted while the node is running using the **`rqt_reconfigure`** GUI.

| Parameter | Env Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `num_inference_steps` | `FLUX2K_NUM_STEPS` | `4` | Number of denoising steps. |
| `guidance_scale` | `FLUX2K_GUIDANCE` | `1.0` | Prompt following strength. |
| `num_images_per_prompt`| `FLUX2K_NUM_IMAGES` | `1` | Batch generation count. |
| `max_sequence_length` | `FLUX2K_MAX_SEQ` | `512` | Max prompt length. |
| `frame_id` | `FLUX2K_FRAME_ID` | `flux2k` | The frame_id for the output. |

### Static Parameters
These are set at **startup** and cannot be changed while the node is running.

| Parameter | Env Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `repo_id` | `FLUX2K_REPO_ID` | `...` | Model repository ID. |
| `model_dir` | `FLUX2K_MODEL_DIR` | `./models` | Cache directory. |
| `device` | `FLUX2K_DEVICE` | `cuda:0` | Computing device. |

### Dynamic Reconfiguration GUI

To start the configuration GUI and adjust the dynamic parameters, run:
```bash
ros2 run rqt_reconfigure rqt_reconfigure
```

## Configuration File

An example configuration file is provided at `config/tti.yaml`. This file contains all parameters with descriptions and can be used to set the initial behavior of the node without providing long command-line arguments.

## Hardware Optimization

The node is optimized for NVIDIA consumer GPUs (like the RTX 4090) using `torch.bfloat16`. It also utilizes `enable_model_cpu_offload()` to ensure memory efficiency, allowing it to coexist with other GPU-intensive nodes.
