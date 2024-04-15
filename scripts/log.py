import os
import json

def create_folder_path(args):
    if args.l96:
        args.prefix += f'l96'
    if args.metric_epochs > 0:
        prefix_for_CL  = f'xl_{args.x_len}_embd_{args.embed_dim}'
        prefix_for_operator = 'lmCL_{}'.format(args.lambda_contra)
    else:
        prefix_for_CL = 'noCL'
        prefix_for_operator = f'xl_{args.x_len}'

    if args.with_geomloss > 0:
        prefix_for_operator += f'_lmOT_{args.lambda_geomloss}'
        if args.with_geomloss_kd > 0:
            print('We are using OT loss *without* the perfect knowledge of the dynamics.')
            print('The detailed configuration is controlled by the command value of args.with_geomloss_kd')
            prefix_for_operator += '_OT_{}'.format(args.with_geomloss_kd)
    args.prefix = f'{args.prefix}_{prefix_for_CL}_{prefix_for_operator}'

    print('\n', args.prefix)
    contra_models_path = 'saved_checkpoints/CL'
    operators_path = 'saved_checkpoints/operators'
    output_path = f'output_folder/{args.prefix}'
    if args.is_master:
        os.makedirs(f'{contra_models_path}/{prefix_for_CL}', exist_ok = True)
        os.makedirs(f'{operators_path}/{args.prefix}', exist_ok = True)
        os.makedirs(output_path, exist_ok = True)
        img_folder = f'output_folder/{args.prefix}'
        with open('{}/configuration.txt'.format(img_folder), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    return prefix_for_CL, contra_models_path, operators_path, output_path
