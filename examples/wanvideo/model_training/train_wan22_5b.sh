# bash
TIMESTAMP=$(date +%m%d%H%M)
OUTPUT_PATH=./models/train/Wan2.2-TI2V-5B_full_${TIMESTAMP}

accelerate launch --config_file examples/wanvideo/model_training/accelerate_config_5B.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path /data/phyworld/combinatorial_data/new10003/train_000 \
  --data_file_keys "file_name" \
  --prompt_path "examples/wanvideo/model_training/prompt.txt" \
  --negative_prompt_path "examples/wanvideo/model_training/nega_prompt.txt" \
  --height 512 \
  --width 512 \
  --num_frames 49 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$OUTPUT_PATH" \
  --trainable_models "dit" \
  --extra_inputs "input_image" \
  --save_steps $((495)) \
  --validation_steps $((495)) \
  --validation_dataset_path /data/phyworld/combinatorial_data/new10003/test_000 /data/phyworld/combinatorial_data/new10003/test_090 \
  --num_validation_videos_per_image 5