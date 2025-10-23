import torch, os, json, glob
import gc
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from diffsynth import load_state_dict, save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig, WanVideoUnit_PromptEmbedder
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser
from diffsynth.trainers.unified_dataset import UnifiedDataset, LoadVideo, ImageCropAndResize, ToAbsolutePath
from PIL import Image


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
        negative_prompt_path=None,
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
        self.nega_prompt_emb_cache_path = "nega_" + prompt_emb_cache_path

        self.prompt = ""
        if prompt_path is not None:
            with open(prompt_path, "r", encoding="utf-8") as f:
                self.prompt = f.read().strip()

        self.negative_prompt = ""
        if negative_prompt_path is not None:
            with open(negative_prompt_path, "r", encoding="utf-8") as f:
                self.negative_prompt = f.read().strip()

        self.cached_prompt_emb = None # GPU上のキャッシュ
        self.cached_nega_prompt_emb = None # GPU上のキャッシュ
        self.cached_prompt_emb_cpu = None # CPU上のキャッシュ
        self.cached_nega_prompt_emb_cpu = None # CPU上のキャッシュ

        text_encoder_needed = False
        # ポジティブプロンプトのキャッシュ処理
        if os.path.exists(self.prompt_emb_cache_path):
            print(f"Loading cached prompt embedding from {self.prompt_emb_cache_path}")
            try:
                self.cached_prompt_emb_cpu = torch.load(self.prompt_emb_cache_path, weights_only=True)
            except:
                self.cached_prompt_emb_cpu = torch.load(self.prompt_emb_cache_path, weights_only=False)
        else:
            text_encoder_needed = True

        # ネガティブプロンプトのキャッシュ処理
        if os.path.exists(self.nega_prompt_emb_cache_path):
            print(f"Loading cached negative prompt embedding from {self.nega_prompt_emb_cache_path}")
            try:
                self.cached_nega_prompt_emb_cpu = torch.load(self.nega_prompt_emb_cache_path, weights_only=True)
            except:
                self.cached_nega_prompt_emb_cpu = torch.load(self.nega_prompt_emb_cache_path, weights_only=False)
        else:
            text_encoder_needed = True

        # キャッシュが存在しない場合はエンコードして保存
        if text_encoder_needed:
            print("Encoding and caching prompt embeddings...")
            inference_device = "cuda" if torch.cuda.is_available() else "cpu"
            self.pipe.load_models_to_device(["text_encoder"])
            self.pipe.text_encoder.to(inference_device)

            if self.cached_prompt_emb_cpu is None:
                prompt_emb = self.pipe.prompter.encode_prompt(self.prompt, positive=True, device=inference_device)
                self.cached_prompt_emb_cpu = prompt_emb.to("cpu")
                torch.save(self.cached_prompt_emb_cpu, self.prompt_emb_cache_path)

            if self.cached_nega_prompt_emb_cpu is None:
                nega_prompt_emb = self.pipe.prompter.encode_prompt(self.negative_prompt, positive=False, device=inference_device)
                self.cached_nega_prompt_emb_cpu = nega_prompt_emb.to("cpu")
                torch.save(self.cached_nega_prompt_emb_cpu, self.nega_prompt_emb_cache_path)

            self.pipe.text_encoder.to("cpu")
            self.pipe.load_models_to_device([]) # Text EncoderをCPUに戻す (VAEやDiTが使うVRAMを確保するため)

        # Text Encoderを削除
        if self.cached_prompt_emb_cpu is not None:
            print("Prompt embedding is cached. Deleting text encoder to save memory.")
            del self.pipe.text_encoder
            self.pipe.text_encoder = None
            gc.collect()
            torch.cuda.empty_cache() # VRAMを確実に解放

    def forward_preprocess(self, data):
        # Move cached embedding to the correct device on first forward pass
        # ここで `self.pipe.device` は `accelerator.device` に設定されるため、
        # `accelerator.prepare` 後に初めて`forward`が呼ばれた際に実行される。
        if self.cached_prompt_emb is None and self.cached_prompt_emb_cpu is not None:
            self.cached_prompt_emb = self.cached_prompt_emb_cpu.to(self.pipe.device)
            del self.cached_prompt_emb_cpu # CPUメモリも解放
            gc.collect()
        if self.cached_nega_prompt_emb is None and self.cached_nega_prompt_emb_cpu is not None:
            self.cached_nega_prompt_emb = self.cached_nega_prompt_emb_cpu.to(self.pipe.device)
            del self.cached_nega_prompt_emb_cpu # CPUメモリも解放
            gc.collect()
        torch.cuda.empty_cache() # キャッシュ転送後のVRAMを整理

        # Use cached prompt.
        inputs_posi = {"context": self.cached_prompt_emb}
        inputs_nega = {"context": self.cached_nega_prompt_emb}
        
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
        
        # Pipeline units
        for unit in self.pipe.units:
            # PromptEmbedderは常にスキップ
            if isinstance(unit, WanVideoUnit_PromptEmbedder):
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

    # --- データセット準備 ---
    video_data_path = os.path.join(args.dataset_base_path, "videos")
    metadata_path = os.path.join(video_data_path, "metadata.jsonl")

    # Fallback to original metadata path if new one doesn't exist
    if not os.path.exists(metadata_path):
        metadata_path = args.dataset_metadata_path
        video_data_path = args.dataset_base_path

    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=video_data_path,
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

    # --- モデル初期化 ---
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
        negative_prompt_path=args.negative_prompt_path,
        prompt_emb_cache_path=args.prompt_emb_cache_path,
    )

    # --- 検証ロジックの定義 ---
    def run_validation(validation_state_dict, current_step, training_module, accelerator):
        # メインプロセスでのみログ出力
        if accelerator.is_main_process:
            print(f"  Starting validation for step {current_step}...")

        # 1. バックアップとチェックポイントのロード
        pipe_for_validation = training_module.pipe
        lora_base = getattr(pipe_for_validation, args.lora_base_model) if args.lora_base_model else pipe_for_validation.dit

        # すべてのプロセスで元のstate_dictを保持
        original_state_dict_cpu = {k: v.cpu().clone() for k, v in lora_base.state_dict().items()}

        if args.lora_base_model:
            pipe_for_validation.load_lora(lora_base, state_dict=validation_state_dict, alpha=1.0)
        else:
            lora_base.load_state_dict(validation_state_dict, strict=False)

        # pipe_for_validation.to(accelerator.device) # 修正: 削除。acceleratorが管理しているため不要
        lora_base.eval()

        # デバイス上でキャッシュを準備
        # ここでは training_module.cached_prompt_emb (GPU上のキャッシュ) を直接使用
        posi_emb = training_module.cached_prompt_emb
        nega_emb = training_module.cached_nega_prompt_emb

        # 検証用メタデータを読み込む (全プロセスで読み込んでも問題ない)
        for validation_path in args.validation_dataset_path:
            dataset_name = os.path.basename(validation_path)
            if accelerator.is_main_process:
                print(f"  Validating with dataset: {dataset_name}")

            validation_metadata_path = os.path.join(validation_path, "images", "metadata.jsonl")
            if not os.path.exists(validation_metadata_path):
                if accelerator.is_main_process:
                    print(f"    Warning: metadata.jsonl not found in {validation_path}/images. Skipping this dataset.")
                continue

            all_validation_data = []
            with open(validation_metadata_path, 'r') as f:
                for line in f:
                    all_validation_data.append(json.loads(line.strip()))

            # 各プロセスに検証データを分割して割り当てる
            num_processes = accelerator.num_processes
            process_index = accelerator.process_index
            data_per_process = len(all_validation_data) // num_processes
            start_index = process_index * data_per_process
            end_index = (process_index + 1) * data_per_process if process_index < num_processes - 1 else len(all_validation_data)

            # 現在のプロセスが担当するデータのサブセット
            validation_data_subset = all_validation_data[start_index:end_index]

            # メインプロセスでのみ出力ディレクトリを作成
            validation_output_path = os.path.join(args.output_path, "validations", f"step_{current_step}", dataset_name)
            if accelerator.is_main_process:
                os.makedirs(validation_output_path, exist_ok=True)
            accelerator.wait_for_everyone() # ディレクトリ作成を待つ

            # 3. 推論実行
            with torch.no_grad():
                # メタデータに基づいてループ (割り当てられたサブセットを使用)
                for item in validation_data_subset:
                    file_name = item["file_name"]
                    item_id = item["id"]
                    image_path = os.path.join(validation_path, "images", file_name)

                    if accelerator.is_main_process: # ログはメインプロセスのみ
                        print(f"    Validating with {file_name} (ID: {item_id})...")

                    if not os.path.exists(image_path):
                        if accelerator.is_main_process:
                            print(f"      Warning: Image file not found at {image_path}. Skipping.")
                        continue

                    input_image = Image.open(image_path)

                    # 各画像に対して複数の動画を生成
                    for i in range(args.num_validation_videos_per_image):
                        seed = 420*item_id + 1139*i
                        if accelerator.is_main_process:
                            print(f"      Generating video {i+1}/{args.num_validation_videos_per_image} with seed {seed}...")

                        # 全てのプロセスで推論を実行
                        video = pipe_for_validation(
                            prompt=training_module.prompt,
                            negative_prompt=training_module.negative_prompt,
                            context=posi_emb,
                            negative_context=nega_emb,
                            input_image=input_image,
                            seed=seed,
                            height=args.height,
                            width=args.width,
                            num_frames=args.num_frames,
                            tiled=True,
                            cfg_scale=5.0,
                            progress_bar_cmd=lambda x: x # サブプロセスのプログレスバーを無効化
                        )

                        # 全てのプロセスがそれぞれの結果を保存
                        # ファイル名がユニークなので競合は発生しない
                        output_filename = f"{item_id}_{i}.mp4"
                        save_video(video, os.path.join(validation_output_path, output_filename), fps=10)

        # 全てのプロセスが推論を完了するまで待機
        accelerator.wait_for_everyone()

        # モデルの状態を学習時に戻す
        lora_base.load_state_dict(original_state_dict_cpu)

        # 2. パイプライン全体を学習モードに戻す (スケジューラ設定とモデルモード切り替えを含む)
        # この関数は内部で scheduler.set_timesteps(1000, training=True) と
        # 各モデルの train()/eval() 切り替えをまとめて行います。
        training_module.switch_pipe_to_training_mode(
            pipe=pipe_for_validation,
            trainable_models=args.trainable_models,
            lora_base_model=args.lora_base_model,
            lora_target_modules=args.lora_target_modules,
            lora_rank=args.lora_rank
        )

        # --- クリーンアップ ---
        gc.collect()
        torch.cuda.empty_cache()

        # 全てのプロセスが状態復元とクリーンアップを完了するのを待つ
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            print("Validation finished.")

    # --- ModelLoggerの初期化 ---
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        validation_fn=run_validation if (args.validation_steps is not None and args.validation_dataset_path is not None) else None,
        max_checkpoints=10
    )

    # --- 学習の開始 ---
    launch_training_task(dataset, model, model_logger, args=args)