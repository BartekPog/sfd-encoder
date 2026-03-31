import yaml

def update_config(base_file, out_file, new_exp_name):
    with open(base_file, 'r') as f:
        config = yaml.safe_load(f)
    
    config['model']['noisy_img_encode'] = False
    config['train']['exp_name'] = new_exp_name
    
    if 'noisy_img_encode' in config['wandb']['tags']:
        config['wandb']['tags'].remove('noisy_img_encode')
    elif 'noisy_img' in config['wandb']['tags']:
        config['wandb']['tags'].remove('noisy_img_encode')
        
    if 'clean_img_encode' not in config['wandb']['tags']:
        config['wandb']['tags'].append('clean_img_encode')

    with open(out_file, 'w') as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=False)

# For E4 (dropout=0.3)
update_config('configs/sfd/hidden_b_h200_from_ft/e4_noisy_enc_drop03_p3_sep_enc.yaml',
              'configs/sfd/hidden_b_h200_from_ft/e4_clean_enc_drop03_p3_sep_enc.yaml',
              'e4_clean_enc_drop03_p3_sep_enc')
              
# For V4 Separate Encoder (NO dropout)
update_config('configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001_noisy_enc_curriculum_cfg_1p5_sep_enc.yaml',
              'configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001_clean_enc_curriculum_cfg_1p5_sep_enc.yaml',
              'v4_mse01_cos001_clean_enc_curriculum_cfg_1p5_sep_enc')

