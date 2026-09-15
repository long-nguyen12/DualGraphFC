class Config:
    # Directory containing train/, val/, and test/ MOCHEG directories.
    data_root = "dataset/mocheg"

    checkpoint_dir = "outputs/checkpoints"
    log_dir = "outputs/logs"
    prediction_dir = "outputs/predictions"

    text_model = "microsoft/deberta-v3-base"
    vision_model = "sail/poolformer_s12"
    max_text_length = 256

    image_size = 224
    hidden_dim = 256

    text_gnn_layers = 2
    text_gnn_heads = 4
    text_graph_k = 3

    vision_knn = 9
    vision_gnn_layers = 4

    cross_heads = 4

    batch_size = 16
    epochs = 30
    num_workers = 0
    seed = 42

    transformer_lr = 2e-5
    graph_lr = 1e-4
    weight_decay = 0.01
    max_grad_norm = 1.0

    alignment_weight = 0.1
    temperature = 0.07

    dropout = 0.1
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
