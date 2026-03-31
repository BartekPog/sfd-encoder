import yaml
import os

base_path = 'configs/sfd/hidden_b_h200_from_ft/'
base_file = os.path.join(base_path, 'v4_mse01_cos001_noisy_enc_curriculum_cfg_1p5.yaml')

with open(base_file, 'r') as f:
    base = yaml.safe_load(f)

# E1: Pass 2 gets clean encoding (for 70%) and full noise (for 30%). No Pass 3.
# no representation guidance.
e1 = eval(repr(base))
e1['model']['hidden_dropout_prob'] = 0.3
e1['model']['hidden_clean_only_pass2'] = True
e1['model']['noisy_img_encode'] = False # the user said clean encoding, not noisy, for passed ones
e1['model']['hidden_weight'] = 0.0
e1['model']['hidden_cos_weight'] = 0.0
e1['model']['hidden_guidance_scale'] = 1.0 # no rep guidance
e1['model']['hidden_loss_warmup_start'] = 999999
e1['model']['hidden_loss_warmup_end'] = 999999
e1['train']['exp_name'] = 'e1_clean_enc_drop03_no_p3'
e1['wandb']['tags'] = ["E1", "base", "H8", "encode_ablation", "clean_enc", "drop03", "no_p3"]

with open(os.path.join(base_path, 'e1_clean_enc_drop03_no_p3.yaml'), 'w') as f:
    yaml.dump(e1, f, sort_keys=False)

# E2: Simliar to before, but Pass 2 gets either noisy encoding (noise level distibution according to curriculum) or full noise with 30% probability, again no Pass 3
# and no rep guidance? The user said "and we don't do rep guidance during training" for E1, implying E2 also might not? Actually, let's keep rep guidance 1.5 if they didn't explicitly say disable for E2. Wait, the user said "Similar to before... again no Pass 3". If "similar to before" means no rep_guidance... wait, I'll set rep_guidance to 1.5 since it's the base config and they only mentioned disabling it for E1 or E1 as a special case? "we don’t do representation guidance during training." was for E1 as part of "skip Pass 1". Let's disable for E2 as well? Actually, if it's base, leave it 1.5. No, "similar to before" means no rep guidance. Wait, base has cfg_1p5! So E2 should keep cfg_1p5.
e2 = eval(repr(base))
e2['model']['hidden_dropout_prob'] = 0.3
e2['model']['hidden_weight'] = 0.0
e2['model']['hidden_cos_weight'] = 0.0
e2['model']['hidden_loss_warmup_start'] = 999999
e2['model']['hidden_loss_warmup_end'] = 999999
e2['train']['exp_name'] = 'e2_noisy_enc_drop03_no_p3'
e2['wandb']['tags'] = ["E2", "base", "H8", "encode_ablation", "noisy_enc", "drop03", "no_p3", "cfg_1p5"]

with open(os.path.join(base_path, 'e2_noisy_enc_drop03_no_p3.yaml'), 'w') as f:
    yaml.dump(e2, f, sort_keys=False)


# (3) Could you also make a variant similar to (2), but with Pass 3 enabled? So it should be like v4_mse01_cos001_noisy_enc_curriculum_cfg_1p5, but with the 30% “noisy dropout” (full noising) on the hidden tokens
e3 = eval(repr(base))
e3['model']['hidden_dropout_prob'] = 0.3
e3['train']['exp_name'] = 'e3_noisy_enc_drop03_p3'
e3['wandb']['tags'] = ["E3", "base", "H8", "encode_ablation", "noisy_enc", "drop03", "p3", "cfg_1p5"]

with open(os.path.join(base_path, 'e3_noisy_enc_drop03_p3.yaml'), 'w') as f:
    yaml.dump(e3, f, sort_keys=False)


# E4: similar to (3), but the encoding would be performed by an architectural COPY of the diffusion model 
e4 = eval(repr(base))
e4['model']['hidden_dropout_prob'] = 0.3
e4['model']['use_separate_encoder'] = True
e4['train']['exp_name'] = 'e4_noisy_enc_drop03_p3_sep_enc'
e4['wandb']['tags'] = ["E4", "base", "H8", "encode_ablation", "noisy_enc", "drop03", "p3", "sep_enc", "cfg_1p5"]

with open(os.path.join(base_path, 'e4_noisy_enc_drop03_p3_sep_enc.yaml'), 'w') as f:
    yaml.dump(e4, f, sort_keys=False)

print("Done generating configs")
