import os
import sys
import time
import random
import string
import gc

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.parameter import ParameterType

import torch
from diffusers import Flux2KleinPipeline
from diffusers.utils import load_image
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
from PIL import Image as PILImage

"""
ROS node for the **FLUX.2-klein** text-to-image and image-to-image models from Black Forest Labs.
"""

class Flux2Knode(Node):
    """
    A ROS node that interfaces with the FLUX.2-klein model for text-to-image 
    and image-to-image generation.
    """

    def __init__(self):
        super().__init__('tti')
        self.get_logger().info("Initializing Flux2-klein ROS node...")

        self.bridge = CvBridge()

        # --- ROS 2 Parameter Declaration ---
        # We follow the pattern from flux2_node.py
        self.declare_parameter(
            'repo_id',
            os.environ.get('FLUX2K_REPO_ID', 'black-forest-labs/FLUX.2-klein-4B'),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='The Huggingface repo id to use.'
            )
        )
        self.declare_parameter(
            'model_dir',
            os.environ.get('FLUX2K_MODEL_DIR', os.path.join(os.getcwd(), 'models')),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='Directory to cache models.'
            )
        )
        self.declare_parameter(
            'device',
            os.environ.get('FLUX2K_DEVICE', 'cuda:0'),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='The torch device to use.'
            )
        )
        self.declare_parameter(
            'prompt',
            os.environ.get('FLUX2K_PROMPT', ''),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='An initial prompt to generate an image at startup.'
            )
        )
        self.declare_parameter(
            'input_image',
            os.environ.get('FLUX2K_INPUT_IMAGE', ''),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='Path to an input image to condition the generation.'
            )
        )
        self.declare_parameter(
            'once',
            os.environ.get('FLUX2K_ONCE', 'false').lower() in ['1','true'],
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description='If true, the node will exit after generating the first image.'
            )
        )
        self.declare_parameter(
            'image_path',
            os.environ.get('FLUX2K_IMAGE_PATH', 'generated_images/auto'),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='Path for saving the output image.'
            )
        )
        self.declare_parameter(
            'seed',
            int(os.environ.get('FLUX2K_SEED', -1)),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='Seed for the random number generator.'
            )
        )
        self.declare_parameter(
            'image_counter_start',
            int(os.environ.get('FLUX2K_IMAGE_COUNTER_START', 1)),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='The starting value for the image filename counter.'
            )
        )
        self.declare_parameter(
            'height',
            int(os.environ.get('FLUX2K_HEIGHT', 1024)),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='Height of the generated image.'
            )
        )
        self.declare_parameter(
            'width',
            int(os.environ.get('FLUX2K_WIDTH', 1024)),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='Width of the generated image.'
            )
        )
        self.declare_parameter(
            'num_inference_steps',
            int(os.environ.get('FLUX2K_NUM_STEPS', 4)),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='Number of denoising steps.'
            )
        )
        self.declare_parameter(
            'guidance_scale',
            float(os.environ.get('FLUX2K_GUIDANCE', 1.0)),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='Creative control (higher = stricter prompt following).'
            )
        )
        self.declare_parameter(
            'num_images_per_prompt',
            int(os.environ.get('FLUX2K_NUM_IMAGES', 1)),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='Number of images to generate per prompt.'
            )
        )
        self.declare_parameter(
            'max_sequence_length',
            int(os.environ.get('FLUX2K_MAX_SEQ', 512)),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='Max prompt sequence length.'
            )
        )
        self.declare_parameter(
            'frame_id',
            os.environ.get('FLUX2K_FRAME_ID', 'flux2k'),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='The frame_id for the generated image messages.'
            )
        )
        self.declare_parameter(
            'keep_loaded',
            os.environ.get('FLUX2K_KEEP_LOADED', 'true').lower() in ['1','true'],
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description='If true, keeps the model in memory between prompts.'
            )
        )
        self.declare_parameter(
            'cpu_offload',
            os.environ.get('FLUX2K_CPU_OFFLOAD', 'true').lower() in ['1','true'],
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description='If true, uses model CPU offloading to save VRAM.'
            )
        )

        # --- Get Parameters ---
        self.repo_id = self.get_parameter('repo_id').value
        self.model_dir = self.get_parameter('model_dir').value
        self.device = self.get_parameter('device').value
        self.image_path = self.get_parameter('image_path').value
        self.prompt = self.get_parameter('prompt').value
        self.seed = self.get_parameter('seed').value
        self.input_image = self.get_parameter('input_image').value
        self.once = self.get_parameter('once').value
        self.image_counter = self.get_parameter('image_counter_start').value
        self.height = self.get_parameter('height').value
        self.width = self.get_parameter('width').value
        self.torch_dtype = torch.bfloat16
        self.pipe = None
        self.current_repo_id = None
        self.current_cpu_offload = None

        # Ensure model_dir exists
        if self.model_dir and not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)
            self.get_logger().info(f"Created model directory: {self.model_dir}")

        # Ensure image_path directory exists if using auto
        if 'auto' in self.image_path:
            out_dir = os.path.dirname(self.image_path)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)

        self.get_logger().info(f"Using repo_id: '{self.repo_id}'")
        self.get_logger().info(f"Using device: '{self.device}'")

        # --- Subscriber ---
        self.subscription = self.create_subscription(
            String,
            'prompt',
            self.prompt_callback,
            1000)

        # --- Publisher ---
        self.image_pub = self.create_publisher(Image, 'generated_image', 10)

        # --- Handle initial prompt ---
        if self.prompt:
            self.get_logger().info("Processing initial prompt...")
            # Use Timer to avoid blocking constructor/initialization
            self.create_timer(0.1, lambda: self._initial_prompt_timer_callback(), oneshot=True)

    def _initial_prompt_timer_callback(self):
        self.prompt_callback(String(data=self.prompt))

    def _get_image_output_path(self):
        if self.image_path.endswith('auto'):
            directory = os.path.dirname(self.image_path)
            random_chars = string.ascii_letters + string.digits
            random_string = ''.join(random.choice(random_chars) for _ in range(10))
            filename = f"flux2k_{self.image_counter:04d}_{random_string}.png"
            return os.path.join(directory, filename)
        else:
            return self.image_path

    def prompt_callback(self, msg):
        prompt = msg.data
        self.get_logger().info(f"Received prompt")
        self.get_logger().debug(f"Prompt: {prompt}")
        
        callback_start = time.time()

        try:
            # --- Model Loading / Caching Logic ---
            cpu_offload = self.get_parameter('cpu_offload').value
            repo_id = self.repo_id # Static for now, but we check anyway

            # Reload if not loaded or if configuration changed
            if self.pipe is None or self.current_repo_id != repo_id or self.current_cpu_offload != cpu_offload:
                if self.pipe is not None:
                    self.get_logger().info("Configuration changed or pipe exists, clearing old pipeline...")
                    del self.pipe
                    gc.collect()
                    torch.cuda.empty_cache()

                start_time = time.time()
                self.get_logger().info(f"Loading Flux2-klein pipeline ({repo_id})...")
                
                self.pipe = Flux2KleinPipeline.from_pretrained(
                    repo_id,
                    torch_dtype=self.torch_dtype,
                    cache_dir=self.model_dir
                )

                if cpu_offload:
                    self.get_logger().info("Optimization: Enabling model CPU offload...")
                    self.pipe.enable_model_cpu_offload()
                else:
                    self.get_logger().info(f"Optimization: Moving whole model to {self.device}...")
                    self.pipe.to(self.device)
                
                self.get_logger().info(f"Pipeline loaded in {time.time() - start_time:.2f}s")
                self.current_repo_id = repo_id
                self.current_cpu_offload = cpu_offload
            else:
                self.get_logger().debug("Using cached pipeline.")

            # Seed handling
            current_seed = self.seed
            if current_seed == -1:
                current_seed = random.randint(0, 2**32 - 1)

            generator = torch.Generator(device=self.device).manual_seed(current_seed)
            # Prepare call arguments
            # We fetch parameter values dynamically to support RQT reconfiguration
            num_steps = self.get_parameter('num_inference_steps').value
            guidance = self.get_parameter('guidance_scale').value
            num_images = self.get_parameter('num_images_per_prompt').value
            max_seq = self.get_parameter('max_sequence_length').value

            self.get_logger().debug(
                f"Inference Params: steps={num_steps}, guidance={guidance:.2f}, num_images={num_images}, max_seq={max_seq}, current_seed={current_seed}")

            call_kwargs = {
                "prompt": prompt,
                "height": self.height,
                "width": self.width,
                "guidance_scale": guidance,
                "num_inference_steps": num_steps,
                "num_images_per_prompt": num_images,
                "max_sequence_length": max_seq,
                "generator": generator
            }

            # Handle image-to-image if input_image is set
            if self.input_image:
                self.get_logger().info(f"Loading input image: {self.input_image}")
                input_img = load_image(self.input_image).convert("RGB")
                call_kwargs["image"] = input_img

            self.get_logger().info(f"Generating image with seed {current_seed} ...")
            gen_start = time.time()
            output = self.pipe(**call_kwargs)
            
            for i, image in enumerate(output.images):
                self.get_logger().info(f"Generation finished (Image {i+1}/{len(output.images)}) in {time.time() - gen_start:.2f}s")

                # Save image
                output_path = self._get_image_output_path()
                image.save(output_path)
                self.get_logger().info(f"Image saved to: {output_path}")

                if self.image_path.endswith('auto'):
                    self.image_counter += 1

                # Publish image as BGR8 (Standard for ROS 2 vision nodes)
                # We convert PIL RGB -> Numpy -> BGR
                image_rgb = np.array(image.convert("RGB"))
                image_bgr = image_rgb[:, :, ::-1].copy() # Swaps R and B channels and ensures memory is contiguous
                
                image_msg = self.bridge.cv2_to_imgmsg(image_bgr, encoding="bgr8")
                image_msg.header.stamp = self.get_clock().now().to_msg()
                image_msg.header.frame_id = self.get_parameter('frame_id').value
                self.image_pub.publish(image_msg)

        except Exception as e:
            self.get_logger().error(f"Error in prompt_callback: {e}")
        
        finally:
            keep_loaded = self.get_parameter('keep_loaded').value
            
            if not keep_loaded:
                self.get_logger().info("Cleaning up (keep_loaded=false)...")
                if self.pipe is not None:
                    del self.pipe
                    self.pipe = None
                gc.collect()
                torch.cuda.empty_cache()
            else:
                self.get_logger().debug("Keeping model loaded for next prompt.")

            if self.once:
                self.get_logger().info("'once' is True, exiting node.")
                sys.exit(0)
            
            total_duration = time.time() - callback_start
            self.get_logger().info(f"DONE: Total callback duration: {total_duration:.2f}s")

def main(args=None):
    rclpy.init(args=args)
    node = Flux2Knode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
