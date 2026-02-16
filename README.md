# Revisiting MoE Load Balancing
[See presentation](https://docs.google.com/presentation/d/1VzsWgv3v_GQbCMZxmbv_FylsRurCjEWM1-WbVysDjbU/edit?usp=sharing)
## Why does this exist?
- In the [original project](https://github.com/peterqlin/moe-load-balance), we falsely assumed that expert load balance was primarily caused by "real" (i.e. non-padding) tokens
- This assumption turned out to be false upon further scrutiny, which is motivates a new, simpler approach to the problem of expert load imbalance
  - Experiment with pre-sorting batches by length

## What the code does
- Visualize (with heatmap) load imbalance caused by pad tokens during MoE model batch inference
- Implement an optimized forward function that redistributes pad tokens across experts to mitigate expert oversaturation
  - Based on `Qwen3MoeSparseMoeBlock::foward` found [here](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py)
- Plot coefficient of variance, execution time, and MMLU accuracy metrics

## How to run your own evaluations
- Update the list of `BatchRun`s passed into the `Evaluation` class with your own parameters
