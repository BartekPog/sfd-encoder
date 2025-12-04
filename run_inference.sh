CONFIG_PATH=$1
CFG_SCALE=${CFG_SCALE:-}
CFG_INTERVAL_START=${CFG_INTERVAL_START:-}
TIMESTEP_SHIFT=${TIMESTEP_SHIFT:-}
AUTOGUIDANCE_MODEL_SIZE=${AUTOGUIDANCE_MODEL_SIZE:-}
AUTOGUIDANCE_CKPT_ITER=${AUTOGUIDANCE_CKPT_ITER:-}
CFG_SCALE_SEM=${CFG_SCALE_SEM:-}
CFG_SCALE_TEX=${CFG_SCALE_TEX:-}
FID_NUM=${FID_NUM:-}
NUM_SAMPLING_STEPS=${NUM_SAMPLING_STEPS:-}

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-1235}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
PRECISION=${PRECISION:-bf16}

# Build arguments
EXTRA_ARGS=""
if [ -n "$CFG_SCALE" ]; then
    # Note: CFG_SCALE is not quoted here to allow expansion into multiple arguments
    EXTRA_ARGS="$EXTRA_ARGS --cfg_scale $CFG_SCALE"
fi
if [ -n "$CFG_INTERVAL_START" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --cfg_interval_start $CFG_INTERVAL_START"
fi
if [ -n "$TIMESTEP_SHIFT" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --timestep_shift $TIMESTEP_SHIFT"
fi
if [ -n "$AUTOGUIDANCE_MODEL_SIZE" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --autoguidance_model_size $AUTOGUIDANCE_MODEL_SIZE"
fi
if [ -n "$AUTOGUIDANCE_CKPT_ITER" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --autoguidance_ckpt_iter $AUTOGUIDANCE_CKPT_ITER"
fi
if [ -n "$CFG_SCALE_SEM" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --cfg_scale_sem $CFG_SCALE_SEM"
fi
if [ -n "$CFG_SCALE_TEX" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --cfg_scale_tex $CFG_SCALE_TEX"
fi
if [ -n "$FID_NUM" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --fid_num $FID_NUM"
fi
if [ -n "$NUM_SAMPLING_STEPS" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --num_sampling_steps $NUM_SAMPLING_STEPS"
fi

accelerate launch \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $NODE_RANK \
    --num_processes  $(($GPUS_PER_NODE*$NNODES)) \
    --num_machines $NNODES \
    --mixed_precision $PRECISION \
    inference.py \
    --config $CONFIG_PATH --calculate-fid \
    $EXTRA_ARGS