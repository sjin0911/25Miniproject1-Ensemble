import torch.optim as optim

def get_optimizer(config, parameters):
    # 안전 기본값들
    opt_name = str(getattr(config.optim, "optimizer", "Adam")).strip().lower()
    lr       = getattr(config.optim, "lr", 2e-4)
    wd       = getattr(config.optim, "weight_decay", 0.0)
    eps      = getattr(config.optim, "eps", 1e-8)
    amsgrad  = getattr(config.optim, "amsgrad", False)
    betas    = getattr(config.optim, "betas", (0.9, 0.999))

    if opt_name in ("adamw", "adam_w"):
        return optim.AdamW(parameters, lr=lr, betas=betas, eps=eps,
                           weight_decay=wd, amsgrad=amsgrad)

    elif opt_name == "adam":
        return optim.Adam(parameters, lr=lr, betas=betas, eps=eps,
                          weight_decay=wd, amsgrad=amsgrad)

    elif opt_name in ("rmsprop", "rms_prop"):
        momentum = getattr(config.optim, "momentum", 0.0)
        alpha    = getattr(config.optim, "alpha", 0.99)
        return optim.RMSprop(parameters, lr=lr, alpha=alpha,
                             momentum=momentum, weight_decay=wd)

    elif opt_name == "sgd":
        momentum = getattr(config.optim, "momentum", 0.9)
        nesterov = getattr(config.optim, "nesterov", False)
        return optim.SGD(parameters, lr=lr, momentum=momentum,
                         weight_decay=wd, nesterov=nesterov)

    else:
        raise NotImplementedError(f"Optimizer {getattr(config.optim, 'optimizer', opt_name)} not understood.")
