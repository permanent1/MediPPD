def test_public_model_module_has_no_ablation_matrix():
    import medippd_gvlm.task_routed_model as module

    assert not hasattr(module, "task_routed_ablation_matrix")


def test_public_fusion_module_has_no_ablation_variants():
    import medippd_gvlm.red_mask_fusion as module

    assert not hasattr(module, "FUSION_VARIANTS")


def test_dataset_cache_exposes_no_legacy_training_or_ablation_api():
    import medippd_gvlm.dataset_cache as module

    assert not hasattr(module, "ablation_matrix")
    assert not hasattr(module, "train_adapter")
    assert not hasattr(module, "evaluate_yolo_baseline")


def test_main_config_loads_public_yaml():
    from medippd_gvlm.config import load_main_config

    config = load_main_config()
    assert config["seed"] == 42
    assert config["cap_diameter_mm"] == 30.0
    assert config["data"]["dataset_root"] == "datasets/ppd553_seg"


def test_metrics_module_has_no_method_comparison_bootstrap():
    import medippd_gvlm.metrics as module

    assert not hasattr(module, "paired_bootstrap_difference")
