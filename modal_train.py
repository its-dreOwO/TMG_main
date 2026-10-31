"""Fine-tune a transformer risk classifier on Modal GPUs (an RTX PRO 6000 by default, see GPU below).

Modal downloads the public CUAD/LEDGAR datasets from Kaggle itself and builds the
splits on volume `legal-clause-data`; no contract text ever leaves the local machine.

  modal secret create kaggle --from-dotenv <file with KAGGLE_API_TOKEN=...>
  modal run modal_train.py::prepare_data
  modal run modal_train.py::train --model-name xlm-roberta-base --run-name smoke --epochs 1 --max-samples 500
  modal run --detach modal_train.py::train --model-name nlpaueb/legal-bert-base-uncased --run-name legal-bert
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("legal-clause-train")
data_vol = modal.Volume.from_name("legal-clause-data", create_if_missing=True)
model_vol = modal.Volume.from_name("legal-clause-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.12.0", "transformers==4.57.6", "accelerate", "sentencepiece", "protobuf",
        "scikit-learn", "pandas", "pyarrow", "numpy",
    )
    .env({"HF_HOME": "/models/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("lca")
)

data_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("kaggle==2.1.2", "pandas", "pyarrow", "scikit-learn")
    .add_local_python_source("lca")
)

KAGGLE_DATASETS = {
    "cuad": "konradb/atticus-open-contract-dataset-aok-beta",
    "ledgar": "chipthui1306/ledgar",
}
RAW = Path("/data/raw")
DATA = Path("/data/processed")
MODELS = Path("/models")
SEED = 42
GPU = "RTX-PRO-6000"  # Modal GPU type string, see https://modal.com/docs/guide/gpu


@app.function(image=data_image, timeout=3600, cpu=4, memory=8192,
              secrets=[modal.Secret.from_name("kaggle", required_keys=["KAGGLE_API_TOKEN"])],
              volumes={"/data": data_vol})
def prepare_data(force_download: bool = False) -> dict:
    """Download the datasets from Kaggle into the volume, then build data/processed there."""
    import subprocess

    from lca.data.prepare import prepare

    for name, ref in KAGGLE_DATASETS.items():
        target = RAW / name
        if target.exists() and any(target.iterdir()) and not force_download:
            print(f"{name}: already downloaded")
            continue
        target.mkdir(parents=True, exist_ok=True)
        print(f"{name}: downloading {ref}")
        subprocess.run(["kaggle", "datasets", "download", ref, "-p", str(target), "--unzip"], check=True)
    data_vol.commit()
    stats = prepare(RAW, DATA)
    data_vol.commit()
    print(stats)
    return stats


def _limit(df, n: int, n_classes: int):
    """At most n rows, spread across labels so rare classes survive a smoke run."""
    if not n or len(df) <= n:
        return df
    per = max(1, n // n_classes)
    import pandas as pd
    return pd.concat([g.sample(min(len(g), per), random_state=SEED) for _, g in df.groupby("label")])


@app.function(image=image, gpu=GPU, timeout=4 * 3600, retries=2,
              volumes={"/data": data_vol, "/models": model_vol})
def train(model_name: str, run_name: str, epochs: int = 3, lr: float = 2e-5, batch_size: int = 64,
          max_length: int = 256, max_samples: int = 0) -> dict:
    """max_samples=0 means use all rows."""
    import numpy as np
    import pandas as pd
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              TrainerCallback, TrainingArguments, set_seed)
    from transformers.trainer_utils import get_last_checkpoint

    from lca.eval.metrics import predict_labels, report, tune_thresholds
    from lca.train.core import class_weights, make_weighted_trainer, softmax

    set_seed(SEED)
    data_vol.reload()
    classes = ["confidentiality", "indemnity", "non_compete", "none", "penalty", "termination"]
    label2id = {c: i for i, c in enumerate(classes)}
    splits = {s: _limit(pd.read_parquet(DATA / f"{s}.parquet"), max_samples, len(classes))
              for s in ("train", "val", "test")}
    print({s: len(df) for s, df in splits.items()}, "on", torch.cuda.get_device_name(0))

    tok = AutoTokenizer.from_pretrained(model_name)

    class Encoded(torch.utils.data.Dataset):
        def __init__(self, df):
            self.enc = tok(df.text.tolist(), truncation=True, max_length=max_length)
            self.labels = [label2id[l] for l in df.label]

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            item = {k: v[i] for k, v in self.enc.items()}
            item["labels"] = self.labels[i]
            return item

    ds = {s: Encoded(df) for s, df in splits.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(classes), id2label=dict(enumerate(classes)), label2id=label2id)

    def compute_metrics(pred):
        y_pred = [classes[i] for i in pred.predictions.argmax(axis=1)]
        y_true = [classes[i] for i in pred.label_ids]
        return {"macro_f1": report(y_true, y_pred, classes)["macro_risk"]["f1"]}

    class CommitOnSave(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            model_vol.commit()

    run_dir = MODELS / run_name
    ckpt_dir = run_dir / "ckpt"
    bf16 = torch.cuda.is_bf16_supported()
    print("mixed precision:", "bf16" if bf16 else "fp16", "batch_size:", batch_size)
    args = TrainingArguments(
        output_dir=str(ckpt_dir), num_train_epochs=epochs, learning_rate=lr,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size * 2,
        warmup_ratio=0.06, weight_decay=0.01, bf16=bf16, fp16=not bf16, seed=SEED,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_macro_f1", greater_is_better=True,
        logging_steps=100, report_to=[], dataloader_num_workers=2,
    )
    Trainer = make_weighted_trainer(class_weights(splits["train"].label.tolist(), classes))
    trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["val"],
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics,
                      callbacks=[CommitOnSave()])
    resume = get_last_checkpoint(str(ckpt_dir)) if ckpt_dir.exists() else None
    print("resuming from", resume) if resume else None
    trainer.train(resume_from_checkpoint=resume)

    val_proba = softmax(trainer.predict(ds["val"]).predictions)
    thresholds = tune_thresholds(splits["val"].label.tolist(), val_proba, classes)
    test_proba = softmax(trainer.predict(ds["test"]).predictions)
    metrics = report(splits["test"].label.tolist(), predict_labels(test_proba, classes, thresholds), classes)
    metrics.update(model_name=model_name, epochs=epochs, lr=lr, max_length=max_length,
                   n_train=len(splits["train"]))

    final = run_dir / "final"
    trainer.save_model(str(final))
    tok.save_pretrained(str(final))
    (final / "thresholds.json").write_text(json.dumps({"classes": classes, "thresholds": thresholds}, indent=2))
    (final / "metrics.json").write_text(json.dumps(metrics, indent=2))
    model_vol.commit()
    summary = {"run": run_name, "macro_risk_f1": metrics["macro_risk"]["f1"],
               "per_label_f1": {l: m["f1"] for l, m in metrics["per_label"].items()}, "thresholds": thresholds}
    print(json.dumps(summary, indent=2))
    return summary
