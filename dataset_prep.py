import os, json, glob, argparse

def main(args):
    rows = []   # This will store all examples (each line for train.jsonl)

    # Iterate over subfolders inside args.root (e.g., data/raw/0001, 0002, ...)
    for triplet in sorted(glob.glob(os.path.join(args.root, "*"))):

        # Paths inside each subfolder
        instr = os.path.join(triplet, "instruction.txt")
        tgt = os.path.join(triplet, "target.jpg")
        inp = os.path.join(triplet, "input.jpg")

        # We MUST have instruction.txt and target.jpg
        # If they don't exist, skip this folder.
        if not (os.path.isfile(instr) and os.path.isfile(tgt)):
            continue

        # Read instruction from text file
        with open(instr, "r", encoding="utf-8") as f:
            instruction = f.read().strip()

        # Create the JSON entry for this example
        row = {
            "instruction": instruction,   # text instruction
            "target_image": tgt           # ground truth output image
        }

        # If input.jpg exists, add it (optional for some tasks)
        if os.path.isfile(inp):
            row["input_image"] = inp

        # Add to the list
        rows.append(row)

    # Ensure output folder exists
    # Example: if args.out = "data/train.jsonl", this creates folder "data/"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Write each example as one JSON line in train.jsonl
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[OK] wrote {len(rows)} examples -> {args.out}")


if __name__ == "__main__":
    # Parse command-line arguments
    ap = argparse.ArgumentParser()

    # REQUIRED: Path to data/raw where numbered subfolders exist
    ap.add_argument("--root", required=True,
                    help="data/raw folder containing numbered subfolders")

    # Optional: output file (default = data/train.jsonl)
    ap.add_argument("--out", default="data/train.jsonl")

    # Run main() with parsed args
    main(ap.parse_args())
