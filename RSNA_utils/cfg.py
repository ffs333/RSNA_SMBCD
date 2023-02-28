# ====================================================
# Config
# ====================================================
import torch


class CFG:
    ####################
    # MAIN
    ####################
    wandb = True
    wandb_project = 'WandB_project_name'
    competition = 'competition_name'
    wb_group = None
    exp_name = 'experiment_name'
    base_path = 'path_to_base_directory'

    seed = 333
    train = True
    debug = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ####################
    # DATA
    ####################

    use_external = True
    external_multiplier = 2  # 2
    dnt_use_ext_after = 10  # 9

    img_size = 2048
    image_size = (1536, 1024)  # for transformer
    dataset = 'stripes'  # ['v1', 'v2', 'transformer', 'stripes', 'transformer_stripes']
    use_meta = False  # False
    pos_multiplier = 10
    num_workers = 12
    train_bs = 5  # 8
    valid_bs = 16
    n_fold = 4
    trn_fold = [1]  # [0,1,2,3]

    interesing_cols = ['site_id', 'patient_id', 'image_id', 'laterality', 'view', 'age',
                       'cancer', 'implant', 'machine_id', 'difficult_negative_case', 'path', 'fold']

    ####################
    # AUGMENTATIONS
    #################### [CURRENT BEST]
    change_aug = False
    use_aug_prob = 0.85    # 0.85

    flip_prob_0 = 0.5      # 0.5
    flip_prob_1 = 0.5      # 0.5

    zoom_prob_0 = 0.35     # 0.35
    zoom_prob_1 = 0.4      # 0.4

    rotate_prob_0 = 0.35   # 0.35
    rotate_prob_1 = 0.4    # 0.4

    contr_prob_0 = 0.45    # 0.45
    contr_prob_1 = 0.45    # 0.45

    elastic_prob_0 = 0.3   # 0.3
    elastic_prob_1 = 0.35  # 0.35

    affine_prob_0 = 0.35   # 0.35
    affine_prob_1 = 0.4    # 0.4

    drop_prob_0 = 0.35     # 0.35
    drop_prob_1 = 0.35     # 0.35

    gibbs_prob_0 = 0.2
    gibbs_prob_1 = 0.25

    gaus_prob_0 = 0.2
    gaus_prob_1 = 0.25

    stripe_prob = 0.7

    # gap_dict = {1: 350,
    #            2: 300,
    #            3: 250,
    #            4: 200,
    #            5: 150,
    #            6: 150,
    #            7: 150} #100}

    gap_dict = {1: 350,
                2: 350,
                3: 300,
                4: 250,
                5: 250,
                6: 200,
                7: 200,
                8: 200,
                9: 150}  # 100}

    circle_aug_prob = 0.15

    ####################
    # MODEL
    ####################
    deep_supervision = True
    deep_supervision_out = True
    model = 'effv2_deepsuper'  # ['v0', 'v1', 'v2', 'v3', 'nextvit', 'effv2_deepsuper', 'v0_deepsuper']  # 'v0'
    backbone = "tf_efficientnetv2_s"
    pretrained = True
    use_act = False
    drop_rate = 0.12  # 0.2
    in_channels = 3

    ####################
    # TRAIN
    ####################
    FULL_TRAIN = False
    patient_wise = False
    apex = True

    eval_after = 0
    eval_every = 1
    eval_always_after = 1

    finetune = False
    finetune_path = 'path_to_checkpoint'
    finetune_fold = 1
    finetune_sched_opt = False
    finetune_epoch = 7
    finetune_change_seed = True

    ####################
    # LOSS
    ####################
    loss = 'LMF'  # ['bce', 'diff_bce', 'LMF']
    pos_wgt = 5  # 2
    dif_wgt = 0.7  # 0.7

    focal_alpha = 0.3
    focal_gamma = 2.
    ldam_max_m = 0.5
    ldam_s = 2
    w_ldam = 1
    w_focal = 7

    # Scheduler step 1

    scheduler = 'onecycle'  # 'onecycle'   #  ["linear", "cosine", "cosine_restart", "onecycle", "simple_cosine" ]
    onecycle_start = 0.06  # 0.1337
    onecycle_m = 1.2
    num_cycles = 0.485  # 1
    num_warmup_steps = 100

    # Loop step 1

    epochs = 16
    use_restart = True
    rest_thr_ = 0.05
    rest_epoch = 8
    iter4eval = 5245

    save_for_future = True
    save_future_epoch = epochs - 5

    # LR, optimizer step 1

    lr = 3.3e-4
    min_lr = 1e-6
    eps = 1e-8
    betas = (0.9, 0.999)
    weight_decay = 0.001
    gradient_accumulation_steps = 1
    optimizer = "AdamW" 
