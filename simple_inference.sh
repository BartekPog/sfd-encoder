PRECISION=bf16 bash run_fast_inference.sh $INFERENCE_CONFIG
# take XL model (675M) as an example. 
CFG_SCALE="1.5" \
AUTOGUIDANCE_MODEL_SIZE="b" \
AUTOGUIDANCE_CKPT_ITER="70" \
PRECISION=bf16 bash run_fast_inference.sh configs/sfd/lightningdit_xl/inference_4m_autoguidance_demo.yaml