_base_ = './default.py'
OptimizationParams = dict(
    hash_init_lr = 0.0002,
    hash_final_lr = 0.000002,
    hashmap_size = 16,  # 15
    lambda_dssim = 0.2,
    dynamic_thro = 5.0,  # 7
    activation = "ReLU",
    weight_prune_from_iter = 3000,
    weight_prune_interval = 3000,
    weight_prune_threshold = 0.02,

)