"""Render the study's Markdown tables from its JSON outputs."""

import argparse
import json
from pathlib import Path

from dequant import PROJECTIONS

STORAGES = ("bf16", "bfp8_b_N", "bfp8_b_K", "bfp4_b_N", "bfp4_b_K")
STORAGE_LABELS = ("BF16", "BFP8 N", "BFP8 K", "BFP4 N", "BFP4 K")


def table(headers, rows):
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print("| " + " | ".join(str(cell) for cell in row) + " |")
    print()


def weight_cell(stats):
    return f"{100 * stats['relative_l2']:.3f}% / {stats['max_abs']:.6g} / {100 * stats['changed_fraction']:.3f}%"


def print_weights(layers):
    for projection in PROJECTIONS:
        print(f"### `{projection}`\n")
        table(
            ("Layer", "Native", *STORAGE_LABELS),
            (
                (
                    layer["layer"],
                    layer["checkpoint_format"],
                    *(weight_cell(layer["weights"][projection]["storage"][name]) for name in STORAGES),
                )
                for layer in layers
            ),
        )


def print_loss(layers, inventory):
    for layer in layers:
        for projection in PROJECTIONS:
            if inventory:
                actual = layer["weights"][projection]["losses"]
                for key in ("elements", "native_bf16_loss"):
                    assert actual[key] == inventory[str(layer["layer"])][projection][key]

    if inventory:
        all_rows = []
        for label, group in (("NVFP4, layers 0–19", range(20)), ("MXFP8, layers 20–23", range(20, 24))):
            for projection in PROJECTIONS:
                count = sum(inventory[str(i)][projection]["elements"] for i in group)
                loss = sum(inventory[str(i)][projection]["native_bf16_loss"] for i in group)
                all_rows.append((label, projection, f"{count:,}", f"{loss:,}", f"{100 * loss / count:.4f}%"))
        table(("All physical layers", "Projection", "Elements", "BF16-inexact", "Fraction"), all_rows)
    rows = []
    for layer in layers:
        for projection in PROJECTIONS:
            result = layer["weights"][projection]
            loss = result["losses"]
            rows.append(
                (
                    layer["layer"],
                    projection,
                    f"{loss['native_bf16_loss']:,} / {loss['elements']:,}",
                    f"{100 * loss['native_bf16_loss'] / loss['nonzero']:.3f}%",
                    f"{loss['hf_fp32_vs_reference']:,}",
                    f"{loss['hf_bf16_vs_rounded_fp32_bf16']:,}",
                    "Yes" if result["hf_bitwise_equal"] else "—",
                )
            )
    table(
        (
            "Layer",
            "Projection",
            "BF16-inexact / all",
            "BF16-inexact of nonzeros",
            "Converter FP32 differs",
            "Converter BF16 differs from directly rounded",
            "Converter bitwise check",
        ),
        rows,
    )


def pct(value):
    return f"{100 * value:.3f}%"


def print_moe(layers):
    table(
        ("Layer", "Experts hit", "Route changes", *STORAGE_LABELS),
        (
            (
                layer["layer"],
                layer["moe"]["experts_hit"],
                layer["moe"]["router_decisions_changed"],
                *(pct(layer["moe"]["relative_l2"][name]) for name in STORAGES),
            )
            for layer in layers
        ),
    )
    checks = []
    for layer in layers:
        if "eager_check" in layer["moe"]:
            check = layer["moe"]["eager_check"]
            checks.append(
                (
                    layer["layer"],
                    pct(check["relative_l2"]),
                    f"{check['different_elements']:,} / {check['total_elements']:,}",
                )
            )
    table(("Layer", "BF16 CPU simulation vs HF eager relative L2", "Different output elements"), checks)


def print_loader(layers):
    table(
        ("Layer", "Gate/up storage error", "Down storage error", "MoE relative L2", "Generic loader bitwise check"),
        (
            (
                layer["layer"],
                *(weight_cell(layer["weights"][name]["storage"]["bf16"]) for name in PROJECTIONS),
                pct(layer["moe"]["relative_l2"]["bf16"]),
                "Yes" if all(p["hf_bitwise_equal"] for p in layer["weights"].values()) else "—",
            )
            for layer in layers
        ),
    )
    checked = [layer["moe"]["eager_check"] for layer in layers if "eager_check" in layer["moe"]]
    if checked:
        error = max(item["relative_l2"] for item in checked)
        print(
            f"Maximum BF16 CPU simulation vs HF eager difference under generic loading was {pct(error)} relative L2.\n"
        )


def print_fidelity(layers):
    for fmt in ("bfp8_b", "bfp4_b"):
        print(f"### `{fmt}`\n")
        rows = []
        for layer in layers:
            err = layer["moe"]["relative_l2"]
            row = [layer["layer"]]
            for path, axis in (("in1", "N"), ("transB", "K"), ("in0", "K")):
                name = f"{fmt}_{axis}"
                terms = (err[f"{name}_{path}_LoFi"], err[f"{name}_{path}_HiFi2"], err[name])
                row.append(" / ".join(map(pct, terms)))
            rows.append(row)
        table(("Layer", "N · in1", "K · transposed B", "K · in0"), rows)


def print_budget(budget):
    base_name = "bfp8_b"
    baseline = budget["options"][base_name]["expert_decode_bytes_per_token"]
    rows = []
    for name, item in budget["options"].items():
        gb = 1_000_000_000
        traffic = item["expert_decode_bytes_per_token"]
        rows.append(
            (
                name,
                f"{item['expert_bytes'] / gb:.3f}",
                f"{item['text_bytes'] / gb:.3f}",
                f"{item['target_bytes'] / gb:.3f}",
                f"{item['target_with_draft_bytes'] / gb:.3f}",
                f"{item['target_one_card_headroom_bytes'] / gb:+.3f}",
                f"{item['target_two_card_even_bytes_per_card'] / gb:.3f}",
                f"{item['target_two_card_ep_bytes_per_card'] / gb:.3f}",
                f"{traffic / gb:.3f}",
                f"{100 * (traffic / baseline - 1):+.1f}%",
                f"{item['text_weight_decode_bytes_per_token'] / gb:.3f}",
            )
        )
    table(
        (
            "Experts 0–19 / 20–23",
            "Experts GB",
            "Text GB",
            "Target GB",
            "Target + draft GB",
            "1-card spare GB",
            "2-card even GB/card",
            "2-card EP GB/card",
            "Expert GB/token",
            "vs all BFP8",
            "Est. text weight GB/token",
        ),
        rows,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("section", choices=("loss", "weights", "moe", "fidelity", "budget", "loader"))
    parser.add_argument("--results", type=Path, nargs="+", help="One or more study result files")
    parser.add_argument("--inventory", type=Path, default=Path("inventory.json"))
    parser.add_argument("--budget", type=Path, default=Path("budget.json"))
    args = parser.parse_args()
    if args.section == "budget":
        print_budget(json.loads(args.budget.read_text()))
    else:
        paths = args.results or [Path("hf-loader.json" if args.section == "loader" else "results.json")]
        layers = sorted(
            (layer for path in paths for layer in json.loads(path.read_text())["layers"]),
            key=lambda layer: layer["layer"],
        )
        if args.section == "loss":
            inventory = json.loads(args.inventory.read_text()) if args.inventory.exists() else None
            print_loss(layers, inventory)
        else:
            {"weights": print_weights, "moe": print_moe, "fidelity": print_fidelity, "loader": print_loader}[
                args.section
            ](layers)
