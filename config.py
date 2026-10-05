"""DualGraphFC configuration."""

class Config:
    architecture_version = 2
    # Directory containing train/, val/, and test/ MOCHEG directories.
    data_root = "dataset/mocheg"
    run_dir = "outputs/runs"

    text_model = "microsoft/deberta-v3-base"
    text_model_revision = None
    vision_model = "openai/clip-vit-base-patch32"
    vision_model_revision = None
    retrieved_text_dir = "dataset/mocheg/retrieved_text"
    max_text_length = 512

    image_size = 224
    hidden_dim = 256
    text_finetune_layers = 2
    vision_finetune_layers = 2

    text_gnn_layers = 2
    text_gnn_heads = 4
    text_graph_k = 2

    vision_gnn_layers = 2

    cross_heads = 4

    batch_size = 8
    epochs = 20
    early_stopping_patience = 7
    num_workers = 0
    seed = 42

    transformer_lr = 2e-5
    vision_lr = 1e-5
    graph_lr = 5e-5
    weight_decay = 0.02
    max_grad_norm = 0.5
    min_lr = 1e-6
    warmup_epochs = 3

    alignment_weight = 0

    dropout = 0.3
    num_classes = 3

    def __init__(self, **overrides):
        for name, value in overrides.items():
            if name not in self.field_names():
                raise ValueError(f"Unknown configuration option: {name}")
            setattr(self, name, value)
        if self.architecture_version != 2:
            raise ValueError("Only DualGraphFC architecture_version=2 is supported")
        if self.alignment_weight != 0:
            raise ValueError("Architecture version 2 uses classification only")

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
        if values.get("architecture_version") != 2:
            raise ValueError(
                "Checkpoint is not DualGraphFC architecture version 2; "
                "use its original implementation to load it"
            )
        return cls(**values)
