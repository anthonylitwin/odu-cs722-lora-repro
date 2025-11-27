import os
import re
import csv

LOG_DIR = "logs"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

MNLI_ROWS = []
E2E_ROWS = []

def extract_trainable_params(text):
    m = re.search(r"Trainable params:\s*(\d+)\s*\|\s*Total params:\s*(\d+)", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

def extract_last_kv(text, key):
    # Finds last occurrence of e.g. 'train_loss': 0.45
    matches = re.findall(rf"'{key}':\s*([\d\.]+)", text)
    return float(matches[-1]) if matches else None

def detect_method(name):
    if "full" in name:
        return "full"
    if "bitfit" in name:
        return "bitfit"
    if "lora_r" in name:
        m = re.search(r"lora_r(\d+)", name)
        return f"lora_r{m.group(1)}"
    return "unknown"

def process_file(path, filename):
    with open(path, "r", errors="ignore") as f:
        text = f.read()

    trainable, total_params = extract_trainable_params(text)
    train_loss = extract_last_kv(text, "train_loss")
    eval_loss = extract_last_kv(text, "eval_loss")
    train_runtime = extract_last_kv(text, "train_runtime")

    # MNLI also has eval_accuracy
    eval_accuracy = extract_last_kv(text, "eval_accuracy")

    # Identify MNLI vs GPT2-E2E
    if filename.startswith("mnli"):
        method = detect_method(filename)
        MNLI_ROWS.append({
            "filename": filename,
            "method": method,
            "trainable_params": trainable,
            "total_params": total_params,
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "eval_accuracy": eval_accuracy,
            "train_runtime": train_runtime,
        })
    elif filename.startswith("e2e"):
        method = detect_method(filename)
        E2E_ROWS.append({
            "filename": filename,
            "method": method,
            "trainable_params": trainable,
            "total_params": total_params,
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "train_runtime": train_runtime,
        })

def main():
    for filename in os.listdir(LOG_DIR):
        if not filename.endswith(".out"):
            continue
        path = os.path.join(LOG_DIR, filename)
        process_file(path, filename)

    # Save MNLI results
    mnli_csv = os.path.join(RESULTS_DIR, "mnli_results.csv")
    with open(mnli_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename", "method",
                "trainable_params", "total_params",
                "train_loss", "eval_loss", "eval_accuracy",
                "train_runtime"
            ]
        )
        writer.writeheader()
        writer.writerows(MNLI_ROWS)

    # Save E2E results
    e2e_csv = os.path.join(RESULTS_DIR, "e2e_results.csv")
    with open(e2e_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename", "method",
                "trainable_params", "total_params",
                "train_loss", "eval_loss",
                "train_runtime"
            ]
        )
        writer.writeheader()
        writer.writerows(E2E_ROWS)

    print("Saved:", mnli_csv)
    print("Saved:", e2e_csv)

if __name__ == "__main__":
    main()
