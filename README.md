# GNP for CG.

## IChol-Guided Sparse Tensor Architecture

This branch includes `ICholSparseTensorNet`, a modular architecture that uses
a frozen IChol sparse-tensor core with lightweight trainable calibration
scalars. It can be run with the same entrypoint used for existing models.

Example:

```bash
python scripts/run_exp.py \
	--mode both \
	--problem Boeing/msc01050 \
	--network_override ICholSparseTensorNet
```

You can also set `NETWORK_OVERRIDE = 'ICholSparseTensorNet'` in
`GNP/config.py` and run your existing job scripts unchanged.