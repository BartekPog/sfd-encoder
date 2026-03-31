import yaml
import copy

base_file = 'configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001_noisy_enc_curriculum_cfg_1p5.yaml'

with open(base_file, 'r') as f:
    base_config = yaml.safe_load(f)

# Ensure common settings
base_config['model']['hidden_reuse_noise_pass2'] = False
base_config['model']['hidden_reuse_noise_pass3'] = False
base_config['model']['hidden_guidance_scale'] = 1.5

def save_config(name, mse_w, cos_w, old_tags, new_tags):
    c = copy.deepcopy(base_config)
    c['model']['hidden_weight'] = mse_w
    c['model']['hidden_cos_weight'] = cos_w
    c['train']['exp_name'] = name
    
    tags = c['wandb']['tags']
    for t in old_tags:
        if t in tags: tags.remove(t)
    for t in new_tags:
        if t not in tags: tags.append(t)
    c['wandb']['tags'] = tags
    
    path = f'configs/sfd/hidden_b_h200_from_ft/{name}.yaml'
    with open(path, 'w') as f:
        yaml.dump(c, f, sort_keys=False, default_flow_style=False)
    print(f"Generated {path}")

# 1. v4_mse0001_noisy_enc_curriculum_repg_1p5
save_config('v4_mse0001_noisy_enc_curriculum_repg_1p5', 
            mse_w=0.001, cos_w=0.0, 
            old_tags=['mse01', 'cos001'], 
            new_tags=['mse0001_only'])

# 2. v4_noisy_enc_curriculum_repg_1p5_no_hloss
save_config('v4_noisy_enc_curriculum_repg_1p5_no_hloss', 
            mse_w=0.0, cos_w=0.0, 
            old_tags=['mse01', 'cos001'], 
            new_tags=['no_hloss'])

# 4. v4_mse001_cos001_noisy_enc_curriculum_repg_1p5
save_config('v4_mse001_cos001_noisy_enc_curriculum_repg_1p5', 
            mse_w=0.01, cos_w=0.01, 
            old_tags=['mse01'], 
            new_tags=['mse001'])

