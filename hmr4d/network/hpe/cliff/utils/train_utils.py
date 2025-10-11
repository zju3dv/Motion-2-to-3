import torch
import numpy as np
from collections import OrderedDict

def load_pretrained_model(model, state_dict, strict=False, overwrite_shape_mismatch=True, remove_lightning=False):
    if remove_lightning:
        print(f'Removing "model." keyword from state_dict keys..')
        pretrained_keys = state_dict.keys()
        new_state_dict = OrderedDict()
        for pk in pretrained_keys:
            if pk.startswith('model.'):
                new_state_dict[pk.replace('model.', '')] = state_dict[pk]
            else:
                new_state_dict[pk] = state_dict[pk]

        model.load_state_dict(new_state_dict, strict=strict)
    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError:
        if overwrite_shape_mismatch:
            model_state_dict = model.state_dict()
            pretrained_keys = state_dict.keys()
            model_keys = model_state_dict.keys()

            updated_pretrained_state_dict = state_dict.copy()

            for pk in pretrained_keys:
                if pk in model_keys:
                    if model_state_dict[pk].shape != state_dict[pk].shape:
                        print(f'size mismatch for \"{pk}\": copying a param with shape {state_dict[pk].shape} '
                                       f'from checkpoint, the shape in current model is {model_state_dict[pk].shape}')

                        if pk == 'model.head.fc1.weight':
                            updated_pretrained_state_dict[pk] = torch.cat(
                                [state_dict[pk], state_dict[pk][:,-7:]], dim=-1
                            )
                            print(f'Updated \"{pk}\" param to {updated_pretrained_state_dict[pk].shape} ')
                            continue
                        else:
                            del updated_pretrained_state_dict[pk]

            model.load_state_dict(updated_pretrained_state_dict, strict=False)
        else:
            raise RuntimeError('there are shape inconsistencies between pretrained ckpt and current ckpt')
    return model