import torch

def load_operator(operator, saved_pth):
    print('loading saved operator', saved_pth)
    checkpoint = torch.load(saved_pth)
    checkpoint = {key.replace('module.','') : val for key, val in checkpoint['state_dict'].items()}
    try:
        operator.module.load_state_dict(checkpoint)
    except:
        operator.load_state_dict(checkpoint)
    return operator
