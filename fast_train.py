#!/usr/bin/env python3
"""
Fast training test: entrena unos pocos pasos para comprobar que el entrenamiento
arranca y la loss baja. NO mide rendimiento real (es un subset).

Uso:
    python fast_train.py --data-root <ruta-dataset> --samples 500 --epochs 3
"""
import os
os.environ.setdefault("TORCH_HOME", r"C:\torch_cache")

import argparse
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast training smoke test")
    parser.add_argument("--data-root", type=str, default="./data",
                        help="Ruta al dataset (raíz de COCO si Config.DATASET='COCO')")
    parser.add_argument("--samples", type=int, default=500, help="Nº de imágenes (subset)")
    parser.add_argument("--epochs", type=int, default=3, help="Nº de epochs del test")
    parser.add_argument("--wandb-offline", action="store_true", help="Wandb en modo offline")
    args = parser.parse_args()

    print("=" * 70)
    print(f"FAST TRAINING TEST — {args.samples} imágenes, {args.epochs} epochs")
    print(f"  device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print("=" * 70)

    from main import principal

    # Reutilizamos `principal`: le pasamos un namespace con overfit=N (subset) y sin wandb.
    args.overfit  = args.samples
    args.no_wandb = True
    principal(args)

    print("\n" + "=" * 70)
    print("FAST TEST COMPLETADO. Si el mIoU sube y la loss baja epoch a epoch, el "
          "pipeline funciona.\nPara entrenar de verdad: python main.py --data-root "
          f"{args.data_root} --epochs <N>")
    print("=" * 70)
