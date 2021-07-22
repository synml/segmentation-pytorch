def convert_ddp_state_dict(state_dict: dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key.replace('module.', '')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict
