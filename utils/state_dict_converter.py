def convert_ddp_state_dict(state_dict: dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key.removeprefix('module.')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def remove_items_in_state_dict(state_dict: dict, keys_to_remove: list):
    for key in keys_to_remove:
        state_dict.pop(key)
    return state_dict
