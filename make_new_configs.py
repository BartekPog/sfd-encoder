import yaml
import copy

base_file = 'configs/sfd/hidden_b_h200_from_ft/v4_mse001_noisy_enc_curriculum_repg_1p5_r3.yaml'

with open(base_file, 'r') as f:
    base_config = yaml.safe_load(f)

# Config 1: v4_mse001_noisy_enc_curriculum_repg_1p5
c1 = copy.deepcopy(base_config)
c1['model']['hidden_reuse_noise_pass2'] = False
c1['model']['hidden_reuse_noise_pass3'] = False
c1['train']['exp_name'] = 'v4_mse001_noisy_enc_curriculum_repg_1p5'

tags1 = c1['wandb']['tags']
if 'v4_r3' in tags1: tags1.remove('v4_r3')
if 'hidden_reuse_pass3' in tags1: tags1.remove('hidden_reuse_pass3')
if 'v4' not in tags1: tags1.append('v4')
c1['wandb']['tags'] = tags1

with open('configs/sfd/hidden_b_h200_from_ft/v4_mse001_noisy_enc_curriculum_repg_1p5.yaml', 'w') as f:
    yaml.dump(c1, f, sort_keys=False, default_flow_style=False)

# Config 2: v4_mse001_noisy_enc_curriculum_hgd_scale_4_repg_1p5
c2 = copy.deepcopy(c1)
c2['model']['hidden_grad_dyn_scale'] = 4.0
c2['train']['exp_name'] = 'v4_mse001_noisy_enc_curriculum_hgd_scale_4_repg_1p5'

tags2 = c2['wandb']['tags']
if 'hgd_scale_4' not in tags2: tags2.append('hgd_scale_4')
c2['wandb']['tags'] = tags2

with open('configs/sfd/hidden_b_h200_from_ft/v4_mse001_noisy_enc_curriculum_hgd_scale_4_repg_1p5.yaml', 'w') as f:
    yaml.dump(c2, f, sort_keys=False, default_flow_style=False)

print("Configs generated successfully.")
