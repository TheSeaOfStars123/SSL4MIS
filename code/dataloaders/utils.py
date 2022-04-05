import importlib
def get_class(class_name, modules):
    for module in modules:
        m = importlib.import_module(module)
        clazz = getattr(m, class_name, None)
        if clazz is not None:
            return clazz
    raise RuntimeError(f'Unsupported dataset class: {class_name}')

def _loader_classes(class_name):
    modules = [
        'code.dataloaders.hdf5',
        'pytorch3dunet.datasets.dsb',
        'pytorch3dunet.datasets.utils.py'
    ]
    return get_class(class_name, modules)