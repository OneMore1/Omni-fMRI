# --- Experiment Settings ---
experiment:
  name: "pretrain_mae"
  output_dir: null
  log_dir: null
  checkpoint_dir: null
  resume: null # Path to checkpoint to resume from, or null
  seed: 42

# --- Data Settings ---
data:
  data_root: null
  datasets: ["HCP", "ISYB", "CHCP", "ABCD", "PIOP1", "PIOP2", "PPMI"]
  train_split_suffixes: ["train_40"]
  val_split_suffixes: ["val_40"]
  input_seq_len: 40 # Temporal length to crop (T dimension)

  # DataLoader settings
  batch_size: 16 # Per GPU batch size
  num_workers: 4
  pin_memory: true
  prefetch_factor: 2

  augment_enable: false

# --- Model Settings ---
model:
  img_size: [96, 96, 96]
  patch_size: [4, 4, 4]
  in_chans: 40
  embed_dim: 768
  depth: 12
  num_heads: 12
  mlp_ratio: 4.0
  qkv_bias: true
  qk_norm: true
  drop_rate: 0.0
  attn_drop_rate: 0.0
  drop_path_rate: 0.1
  num_scales: 2
  thresholds: [0.23]
  method: "std"
  mean: [0.0]
  std: [1.0]
  downstream: false
  gate_attention: "elementwise"

  # MAE SETTINGS
  decoder_embed_dim: 512
  decoder_depth: 8
  decoder_num_heads: 16
  mask_ratio: 0.75
  enable_patch_norm: false

  # JEPA SETTINGS
  predictor_embed_dim: 192
  predictor_depth: 6
  predictor_num_heads: 8
  lamda: 0.05
  num_slices: 1024
  n_points: 17
  use_patch_loss: false
  use_flatten_tokens: false
  sample_ratio: 1.0
  proj_dim: 128

  model_chose: mae  # jepa / mae

# --- Training Settings ---
training:
  # Optimization
  optimizer: "adamw"
  learning_rate: 4.0e-4
  weight_decay: 0.05
  betas: [0.9, 0.95]

  # Learning rate schedule
  lr_scheduler: "cosine"
  warmup_epochs: 5
  min_lr: 1.0e-6 

  # Training duration
  epochs: 400

  # Gradient settings
  clip_grad: 1.0 # Gradient clipping value, null to disable
  accum_iter: 1 # Gradient accumulation steps

  # Mixed precision
  use_amp: true # Use automatic mixed precision

# --- Distributed Training Settings ---
distributed:
  backend: "nccl"
  init_method: "env://"
  world_size: -1 # Will be set automatically
  rank: -1 # Will be set automatically
  dist_url: "env://"

# --- Logging Settings ---
logging:
  print_freq: 20 # Print frequency (iterations)
  log_freq: 40 # Log frequency (iterations)
  save_freq: 4 # Checkpoint save frequency (epochs)

  # Weights & Biases (optional)
  use_wandb: false
  wandb_project: "mae_4d_fmri"
  wandb_entity: null # Your wandb username/team

# --- Validation Settings ---
validation:
  val_freq: 1 # Validation frequency (epochs)
  save_best: true # Save best model based on validation loss
