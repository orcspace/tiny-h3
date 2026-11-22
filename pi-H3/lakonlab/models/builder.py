class Registry(dict):

    def register_module(self, module=None):
        def register(cls):
            self[cls.__name__] = cls
            return cls
        return register(module) if module is not None else register


MODULES = Registry()


def build_module(cfg, default_args=None):
    cfg = cfg.copy()
    module_type = cfg.pop('type')
    return MODULES[module_type](**dict(default_args or {}, **cfg))
