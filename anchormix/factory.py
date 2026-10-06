"""Model factory using only independent neural components in this package."""
from .config import AnchorMixConfig, DEFAULTS
from .model import AnchorMix
from .siren import ModulatedSiren


def create_inr_instance(cfg, input_dim=1, output_dim=1, device="cuda"):
    settings = cfg.inr
    kwargs = dict(dim_in=input_dim, dim_hidden=settings.hidden_dim, dim_out=output_dim,
                  num_layers=settings.depth, w0=settings.w0, w0_initial=settings.w0,
                  use_bias=True, modulate_scale=settings.modulate_scale,
                  modulate_shift=settings.modulate_shift, use_latent=settings.use_latent,
                  latent_dim=settings.latent_dim, modulation_net_dim_hidden=settings.hypernet_width,
                  modulation_net_num_layers=settings.hypernet_depth, last_activation=settings.last_activation)
    kind = settings.model_type
    if kind == "anchormix":
        grid_base = settings.get("grid_base", 64)
        if isinstance(grid_base, bool) or not isinstance(grid_base, int) or grid_base < 1:
            raise ValueError("grid_base must be a positive integer")
        options = dict(settings.get("anchormix", {}))
        unknown = set(options) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown AnchorMix options: {sorted(unknown)}")
        options = vars(AnchorMixConfig(**options).validate())
        model = AnchorMix(**kwargs, grid_base=grid_base, anchor_config=options)
    elif kind == "siren":
        model = ModulatedSiren(**kwargs)
    elif kind in ("siren_GridMix", "gridmix"):
        if input_dim != 2:
            raise ValueError("Legacy GridMix supports 2D grids; AnchorMix infers any coordinate dimension")
        from .legacy_gridmix import ModulatedSirenGridMix
        model = ModulatedSirenGridMix(**kwargs, **{
            key: settings.get(key, default) for key, default in {
                "use_norm": False, "grid_size": 64, "grid_size_2": 0,
                "grid_base": 64, "grid_sum": True, "share_grid": False, "siren_init": True,
            }.items()})
        model.modulated_forward = model.forward
    else:
        raise ValueError(f"Unsupported independent INR type: {kind}")
    model.network_package = "anchormix"
    return model.to(device)


def initialize_models(models, training_coordinates):
    for model in models:
        if isinstance(model, AnchorMix):
            model.initialize_positions(training_coordinates)


def parameter_groups(model, lr):
    return model.parameter_groups(lr) if isinstance(model, AnchorMix) else [{"params": model.parameters()}]


def project_positions(model):
    if isinstance(model, AnchorMix):
        model.project_positions()


def representation_metadata(models, cfg):
    return {"network_package": "anchormix", "schema_version": 1,
            "coordinate_source": ("shape2coordinates_[0,1]" if cfg.data.dataset_name == "airfoil"
                                  else "mean_training_geometry_in_loader_units"),
            "normalization_ntrain": int(cfg.data.ntrain),
            "branches": {kind: model.anchor_metadata() if isinstance(model, AnchorMix)
                         else {"model_type": cfg[f"inr_{kind}"].model_type}
                         for kind, model in models.items()}}
