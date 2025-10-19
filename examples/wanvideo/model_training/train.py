import torch, os, json
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig, WanVideoUnit_PromptEmbedder
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser
from diffsynth.trainers.unified_dataset import UnifiedDataset, LoadVideo, ImageCropAndResize, ToAbsolutePath


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        prompt_path=None,
        prompt_emb_cache_path="prompt_emb_cache.pt",
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs)
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

        # --- プロンプトとキャッシュに関する処理 ---
        self.prompt_emb_cache_path = prompt_emb_cache_path
        self.prompt = None
        if prompt_path is not None:
            with open(prompt_path, "r", encoding="utf-8") as f:
                self.prompt = f.read().strip()

        self.cached_prompt_emb = None
        self.cached_prompt_emb_cpu = None
        if self.prompt is not None:
            if os.path.exists(self.prompt_emb_cache_path):
                print(f"Loading cached prompt embedding from {self.prompt_emb_cache_path}")
                # weights_only=Falseの警告を避けるため、try-exceptで対応
                try:
                    self.cached_prompt_emb_cpu = torch.load(self.prompt_emb_cache_path, weights_only=True)
                except:
                    self.cached_prompt_emb_cpu = torch.load(self.prompt_emb_cache_path, weights_only=False)
            else:
                print("Encoding and caching prompt embedding...")
                inference_device = "cuda"
                self.pipe.load_models_to_device(["text_encoder"])
                self.pipe.text_encoder.to(inference_device)

                prompt_emb = self.pipe.prompter.encode_prompt(self.prompt, positive=True, device=inference_device)

                prompt_emb_cpu = prompt_emb.to("cpu")
                torch.save(prompt_emb_cpu, self.prompt_emb_cache_path)
                self.cached_prompt_emb_cpu = prompt_emb_cpu

                self.pipe.text_encoder.to("cpu")
                self.pipe.load_models_to_device([]) # Offload after encoding
        # ------------------------------------

        
    def forward_preprocess(self, data):
        # Move cached embedding to the correct device on first forward pass
        if self.cached_prompt_emb is None and self.cached_prompt_emb_cpu is not None:
            self.cached_prompt_emb = self.cached_prompt_emb_cpu.to(self.pipe.device)

        # Use cached prompt if available
        if self.cached_prompt_emb is not None:
            inputs_posi = {"context": self.cached_prompt_emb}
        else: # Fallback to original behavior if no prompt is provided
            inputs_posi = {"prompt": data.get("prompt", "")}

        inputs_nega = {}
        
        video_key = "file_name" if "file_name" in data else "video"

        inputs_shared = {
            "input_video": data[video_key],
            "height": data[video_key][0].size[1],
            "width": data[video_key][0].size[0],
            "num_frames": len(data[video_key]),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        
        # Extra inputs
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data[video_key][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data[video_key][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            # If prompt embedding is already cached, skip the unit that generates it.
            if isinstance(unit, WanVideoUnit_PromptEmbedder) and self.cached_prompt_emb is not None:
                continue
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss


if __name__ == "__main__":
    parser = wan_parser()
    parser.add_argument("--prompt_path", type=str, default=None, help="Path to the fixed prompt text file.")
    parser.add_argument("--prompt_emb_cache_path", type=str, default="prompt_emb_cache.pt", help="Path to save/load the cached prompt embedding.")
    args = parser.parse_args()

    # --- データセットパスの組み立て ---
    video_data_path = os.path.join(args.dataset_base_path, "videos")
    metadata_path = os.path.join(video_data_path, "metadata.jsonl")

    # Fallback to original metadata path if new one doesn't exist
    if not os.path.exists(metadata_path):
        metadata_path = args.dataset_metadata_path
        video_data_path = args.dataset_base_path
    # -------------------------------

    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=video_data_path, # videosサブディレクトリを指定
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
        special_operator_map={
            "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16))
        }
    )
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        prompt_path=args.prompt_path,
        prompt_emb_cache_path=args.prompt_emb_cache_path,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )
    launch_training_task(dataset, model, model_logger, args=args)
