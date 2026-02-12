import os
import re
import pandas as pd
import matplotlib.pyplot as plt

ROOT = "checkpoint"
OUT = "log_analysis"
os.makedirs(OUT, exist_ok=True)

epoch_re = re.compile(
    r"Epoch #(\d+): Train loss \[(.*?)\]; Val loss: MSE \[(.*?)\], L1 \[(.*?)\], G-Mean \[(.*?)\]"
)
test_re = re.compile(
    r"Test loss: MSE \[(.*?)\], L1 \[(.*?)\], G-Mean \[(.*?)\]"
)

summary_rows = []

for run in sorted(os.listdir(ROOT)):
    run_dir = os.path.join(ROOT, run)
    log_file = os.path.join(run_dir, "training.log")

    if not os.path.isfile(log_file):
        continue

    ep, tr_l1, va_l1, va_mse, va_gm = [], [], [], [], []
    test_metrics = {}

    with open(log_file) as f:
        for line in f:
            m = epoch_re.search(line)
            if m:
                ep.append(int(m.group(1)))
                tr_l1.append(float(m.group(2)))
                va_mse.append(float(m.group(3)))
                va_l1.append(float(m.group(4)))
                va_gm.append(float(m.group(5)))

            t = test_re.search(line)
            if t:
                test_metrics = {
                    "test_mse": float(t.group(1)),
                    "test_l1": float(t.group(2)),
                    "test_gm": float(t.group(3)),
                }

    if not ep:
        continue

    df = pd.DataFrame({
        "epoch": ep,
        "train_l1": tr_l1,
        "val_l1": va_l1,
        "val_mse": va_mse,
        "val_gmean": va_gm,
    })

    # ---- plots ----
    plt.figure()
    plt.plot(df.epoch, df.train_l1, label="Train L1")
    plt.plot(df.epoch, df.val_l1, label="Val L1")
    plt.xlabel("Epoch")
    plt.ylabel("L1")
    plt.title(run)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/{run}_l1.png")
    plt.close()

    plt.figure()
    plt.plot(df.epoch, df.val_mse, label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(run)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/{run}_mse.png")
    plt.close()

    # ---- summary ----
    best_idx = df.val_l1.idxmin()
    summary = {
        "run": run,
        "best_epoch": int(df.loc[best_idx, "epoch"]),
        "best_val_l1": df.loc[best_idx, "val_l1"],
        "best_val_mse": df.loc[best_idx, "val_mse"],
        "best_val_gmean": df.loc[best_idx, "val_gmean"],
    }
    summary.update(test_metrics)
    summary_rows.append(summary)

# ---- save summary ----
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f"{OUT}/summary.csv", index=False)

print("✔ Analysis complete")
print(f"✔ Plots + summary saved in `{OUT}/`")
