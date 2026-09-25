"""DualGraphFC configuration."""

VISION_MODELS = {
    "poolformer": "sail/poolformer_s12",
    "dinov2": "facebook/dinov2-base",
    "dinov3": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "convnextv2": "facebook/convnextv2-tiny-22k-224",
}


def resolve_vision_model(name):
    try:
        return VISION_MODELS[name]
    except KeyError as exc:
        choices = ", ".join(VISION_MODELS)
        raise ValueError(
            f"Unknown vision model {name!r}; choose one of: {choices}"
        ) from exc


class Config:
    # Directory containing train/, val/, and test/ MOCHEG directories.
    data_root = "dataset/mocheg"

    checkpoint_dir = "outputs/checkpoints"
    log_dir = "outputs/logs"
    prediction_dir = "outputs/predictions"

    text_model = "microsoft/deberta-v3-base"
    # long_text_model = "allenai/longformer-base-4096"
    long_text_model = "microsoft/deberta-v3-base"

    vision_model = VISION_MODELS["dinov2"]
    vision_feature_cache_dir = "dataset/mocheg/vision_features"
    retrieved_text_dir = "dataset/mocheg/retrieved_text"
    max_text_length = 512

    image_size = 224
    hidden_dim = 512
    text_finetune_layers = 0

    text_gnn_layers = 2
    text_gnn_heads = 4
    text_graph_k = 2

    vision_gnn_layers = 2

    cross_heads = 4

    batch_size = 8
    epochs = 30
    early_stopping_patience = 5
    num_workers = 0
    seed = 42

    transformer_lr = 2e-4
    graph_lr = 5e-4
    weight_decay = 0.01
    max_grad_norm = 1.0
    # Retained so configurations stored by older checkpoints remain loadable.
    scheduler_factor = 0.5
    scheduler_patience = 2
    min_lr = 1e-5

    focal_gamma = 2.0
    # Label order: supported, refuted, not enough information.
    class_weights = None

    alignment_weight = 0.1
    temperature = 0.07

    dropout = 0.3
    num_classes = 3

    def __init__(self, **overrides):
        for name, value in overrides.items():
            if name not in self.field_names():
                raise ValueError(f"Unknown configuration option: {name}")
            setattr(self, name, value)

    @classmethod
    def field_names(cls):
        return tuple(
            name
            for name in vars(cls)
            if not name.startswith("_") and not callable(getattr(cls, name))
        )

    def to_dict(self):
        return {name: getattr(self, name) for name in self.field_names()}

    @classmethod
    def from_dict(cls, values):
        return cls(**values)
