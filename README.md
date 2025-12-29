# llm_baseline
大模型项目自用


## SST-2 Baseline (DistilBERT)

### Install

```bash
pip install -r requirements.txt

Train
python scripts/train_sst2_baseline.py --epochs 1

Result (1 epoch)
# eval_accuracy: 0.8956422018348624
# eval_f1: 0.8987764182424917
# saved_to: outputs/sst2_distilbert_baseline
