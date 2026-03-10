import argparse
import csv
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description="Export SST-2 to TSV")
    parser.add_argument("--train_out", type=str, default="sst2_train.tsv",
                        help="Output path for training TSV (default: sst2_train.tsv)")
    parser.add_argument("--eval_out", type=str, default="sst2_eval.tsv",
                        help="Output path for evaluation TSV (default: sst2_eval.tsv)")
    parser.add_argument("--n_train", type=int, default=1000,
                        help="Max number of training samples (default: 1000)")
    parser.add_argument("--n_eval", type=int, default=1000,
                        help="Max number of evaluation samples (default: 1000)")
    args = parser.parse_args()

    print("Loading SST-2 dataset from Hugging Face...")
    raw = load_dataset("glue", "sst2")

    train_data = raw["train"]
    n_train = min(args.n_train, len(train_data))
    print(f"Exporting {n_train} training samples to {args.train_out}")
    with open(args.train_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["sentence", "label"])
        for i in range(n_train):
            sample = train_data[i]
            sentence = sample["sentence"].strip()
            label = sample["label"]
            writer.writerow([sentence, label])

    eval_data = raw["validation"]
    n_eval = min(args.n_eval, len(eval_data))
    print(f"Exporting {n_eval} evaluation samples to {args.eval_out}")
    with open(args.eval_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["sentence", "label"])
        for i in range(n_eval):
            sample = eval_data[i]
            sentence = sample["sentence"].strip()
            label = sample["label"]
            writer.writerow([sentence, label])

    print("Done!")
    print(f"  Train: {args.train_out} ({n_train} samples)")
    print(f"  Eval:  {args.eval_out} ({n_eval} samples)")

if __name__ == "__main__":
    main()
