def get_config():
    return {
        # ---------------------------------------------------------------------
        # Shared output / plotting
        # ---------------------------------------------------------------------
        'output_dir': 'comparison_sample',
        'dpi': 600,
        # ---------------------------------------------------------------------
        # Dataset and sample control
        # ---------------------------------------------------------------------
        'datasets': ['bil', 'exp', 'grf'],
        # Sample index can be set independently for each dataset.
        'sample_index': {
            'bil': 0,
            'exp': 0,
            'grf': 0,
        },
        # Noise levels in percent.
        'noise_levels': [0, 1, 2, 3],
        # ---------------------------------------------------------------------
        # UNet model selection
        # ---------------------------------------------------------------------
        'unet_config_type': 'mix',
        'unet_load_type': 'force_load',
        'unet_use_batch_norm': True,
        'unet_architecture': '[2, 32, 64, 128]',
        'unet_methods': ['MSE', 'GloResloss', 'LocResloss'],
        # Optional Mix-model gamma selectors used by run_sample_unet.py,
        # run_sample_warm_start.py and plot_sample.py.
        # - mix_gamma: one gamma for both LocMixloss/GloMixloss
        # - mix_gamma_by_method: per-method override (higher priority)
        # - strict_mix_gamma: require exact gamma match when loading Mix outputs
        'mix_gamma': None,
        'mix_gamma_by_method': None,
        'strict_mix_gamma': True,
        # ---------------------------------------------------------------------
        # Mesh info used by ASM wrapper in pipeline/common.py
        # ---------------------------------------------------------------------
        'nodesx': 40,
        'nodesy': 40,
        # ---------------------------------------------------------------------
        # ASM inversion controls
        # ---------------------------------------------------------------------
        'asm_dof_order': 'C',  # options: 'C', 'F'
        'asm_gamma': None,     # None means read default gamma from asm_log/config.py
        'asm_max_iter': 600,
        'asm_ftol': 1e-12,
        'asm_gtol': 1e-8,
        # Optional UNet-based warm-start for ASM:
        # if enabled, ASM will load the saved UNet prediction for the same
        # dataset / sample / noise level and use it as E_init.
        # By default, warm-start cases also reuse the corresponding cold-start
        # gamma_opt, so gamma does not need to be set manually.
        # Warm-start ASM files are saved and loaded with a method-specific
        # suffix based on warm_start_method.
        'use_warm_start': False,
        'warm_start_method': 'GloResloss',
        'warm_start_output_dir': 'comparison_sample',
        # Reuse gamma selected from the corresponding cold-start ASM case.
        # If enabled, the current warm-start run skips its own L-curve scan
        # and directly uses the saved cold-start gamma_opt at each noise level.
        'use_cold_start_gamma': True,
        'cold_start_gamma_source_dir': 'comparison_sample',
        # L-curve controls
        'enable_lcurve': True,
        'lcurve_points': 25,
        'lcurve_gamma_min': None,
        'lcurve_gamma_max': None,
    }
