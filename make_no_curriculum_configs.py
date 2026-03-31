import yaml
import copy
import os

base_file = "configs/sfd/hidden_b_h200_from_ft/v4_mse001_noisy_enc_curriculum_repg_1p5.yaml"

with open(base_file, 'r') as f:
    base_data = yaml.safe_load(f)

# Common modifications for all "no curriculum" configs
base_data['model']['hidden_loss_warmup_start'] = 0
base_data['model']['hidden_loss_warmup_end'] = 0
base_data['model']['hidden_t_shift_warmup'] = 0

tags_list = base_data.get('wandb', {}).get('tags', [])
if 'hidden_curriculum' in tags_list:
    tags_list.remove('hidden_curriculum')
if 'no_curriculum' not in tags_list:
    tags_list.append('no_curriculum')

# Variant 1: MSE 0.01, Shift 1.0
v1_data = copy.deepcopy(base_data)
v1_data['model']['hidden_weight'] = 0.01
v1_data['model']['hidden_t_shift_init'] = 1.0
v1_data['model']['hidden_t_shift_final'] = 1.0
v1_data['train']['exp_name'] = "v4_mse001_noisy_enc_nocurr_shift1_repg_1p5"
v1_tags = [t for t in v1_data['wandb']['tags']]
v1_tags.append('shift_1')
v1_data['wandb']['tags'] = v1_tags

# Variant 2: MSE 0.01, Shift 1.5
v2_data = copy.deepcopy(base_data)
v2_data['model']['hidden_weight'] = 0.01
v2_data['model']['hidden_t_shift_init'] = 1.5
v2_data['model']['hidden_t_shift_final'] = 1.5
v2_data['train']['exp_name'] = "v4_mse001_noisy_enc_nocurr_shift1p5_repg_1p5"
v2_tags = [t for t in v2_data['wandb']['tags']]
v2_tags.append('shift_1p5')
v2_data['wandb']['tags'] = v2_tags

# Variant 3: MSE 0.001, Shift 1.0
v3_data = copy.deepcopy(base_data)
v3_data['model']['hidden_weight'] = 0.001
v3_data['model']['hidden_t_shift_init'] = 1.0
v3_data['model']['hidden_t_shift_final'] = 1.0
v3_data['train']['exp_name'] = "v4_mse0001_noisy_enc_nocurr_shift1_repg_1p5"
v3_tags = [t if t != 'mse001_only' else 'mse0001_only' for t in v3_data['wandb']['tags']]
if 'shift_1' not in v3_tags:
    v3_tags.append('shift_1')
v3_data['wandb']['tags'] = list(set(v3_tags))

# Write out files
for data in [v1_data, v2_data, v3_data]:
    out_path = os.path.join(os.path.dirname(base_file), f"{data['train']['exp_name']}.yaml")
    with open(out_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    print(f"Generated {out_path}")

