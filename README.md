# nle-interface-wrapper

Interface Wrapper for [Nethack Learning Environment (NLE)](https://github.com/facebookresearch/nle)

### Installation

```
MINOR=$(python3 -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
pip install "https://github.com/BartekCupial/nle/releases/download/v1.2.1/nle-1.2.0-${MINOR}-${MINOR}-manylinux_2_17_$(uname -m).manylinux2014_$(uname -m).whl"
pip install -e .[dev]
```

### post-installation setup
```bash
pre-commit install
```

### How to play nethack with this wrapper?

```bash
python -m nle_interface_wrapper.scripts.play --env NetHackScore-v0 --seed 42 --play-mode nle
```