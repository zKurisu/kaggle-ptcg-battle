# 05 — RL Training Commands

## Start Training
```bash
cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git
CUDA_VISIBLE_DEVICES=3 python3 -u train.py \
    --iterations 500 --games 32 --device cuda:0 \
    --mcts --mcts-sims 32
```

## Monitor
```bash
tail -f train.log                    # training output
tail -1 checkpoints/ppo_iter*.pt    # latest checkpoint  
nvidia-smi                           # GPU usage
```

## Evaluate
```bash
python3 -c "from ptcg_rl.trainer import export_numpy; ..."
python3 main.py                      # local smoke test
```

## Package for Kaggle
```bash
# After training produces policy.npz:
cp main.py deck.csv policy.npz /tmp/submit/
cp -r cg ptcg_rl /tmp/submit/
cd /tmp/submit && tar czf submission.tar.gz *
```
