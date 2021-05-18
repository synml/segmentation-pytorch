def get_feature_maps(feature_maps: list):
    def hook(module, input, output):
        feature_maps.append(output.detach())
    return hook


def get_feature_maps_with_name(feature_maps: dict, name: str):
    def hook(module, input, output):
        feature_maps[name] = output.detach()
    return hook
