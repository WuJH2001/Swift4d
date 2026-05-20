ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [64, 64, 64, 150]
    },
    sh_degree = 3,
    multires = [1,2],
    defor_depth = 0,
    net_width = 128,
    plane_tv_weight = 0.0002,
    time_smoothness_weight = 0.001,
    l1_time_planes =  0.0001,
    no_do=False,
    no_dshs=False,
    no_ds=False,
    empty_voxel=False,
    render_process=False,
    static_mlp=False

)

OptimizationParams = dict(
    dataloader=True,
    iterations = 13001,
    batch_size=2,
    coarse_iterations = 5000,
    densify_until_iter = 10_000,
    opacity_reset_interval = 60000,
    opacity_threshold_coarse = 0.005,
    opacity_threshold_fine_init = 0.005,
    opacity_threshold_fine_after = 0.005,
    hash_init_lr = 0.0002,
    hash_final_lr = 0.000002,
    hashmap_size = 15,
    lambda_dssim = 0.2,
    dynamic_thro = 7.0,
    activation = "ReLU",
    weight_prune_from_iter = 3000,
    weight_prune_interval = 3000,
    weight_prune_threshold = 0.02,
    n_levels = 16,
    n_features_per_level = 4,
    base_resolution = 16,
    n_neurons = 128,
    # pruning_interval = 2000
)
