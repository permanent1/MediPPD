"""Ultralytics HUB callbacks (disabled in this fork).

在本工程中，我们不使用 Ultralytics HUB，因此这里提供一组安全的空实现，
即使 SETTINGS["hub"] 为 True，也不会向 trainer 注册任何 HUB 相关回调。
"""

from core.utils.events import events


def on_pretrain_routine_start(trainer):  # noqa: D401, ARG001
    """No-op: HUB integration disabled."""


def on_pretrain_routine_end(trainer):  # noqa: D401, ARG001
    """No-op: HUB integration disabled."""


def on_fit_epoch_end(trainer):  # noqa: D401, ARG001
    """No-op: HUB integration disabled."""


def on_model_save(trainer):  # noqa: D401, ARG001
    """No-op: HUB integration disabled."""


def on_train_end(trainer):  # noqa: D401, ARG001
    """No-op: HUB integration disabled."""


def on_train_start(trainer):  # noqa: D401, ARG001
    """No-op: HUB integration disabled."""


def on_val_start(validator):  # noqa: D401, ARG001
    """No-op: HUB integration disabled."""


def on_predict_start(predictor):  # noqa: D401, ARG001
    """Run generic events on predict start (not HUB-specific)."""
    events(predictor.args, predictor.device)


def on_export_start(exporter):  # noqa: D401, ARG001
    """Run generic events on export start (not HUB-specific)."""
    events(exporter.args, exporter.device)


# 在本 fork 中彻底关闭 HUB 集成：不向外暴露任何 HUB 回调
callbacks = {}
