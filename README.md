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

### Potential additions for tracking state in nethack:
- intrinsics - https://nethackwiki.com/wiki/Property#Intrinsic_properties
some of them can be easily tracked with messages https://nethackwiki.com/wiki/You_feel 
    - Disintegration resistance
    - Warning
    - Poison resistance
    - Fire resistance
    - Cold resistance
    - Sleep resistance
    - Shock resistance
    - Invisible
    - See invisible
    - Protection
    - Searching
    - Speed
    - Teleportitis
    - Teleport control
    - Telepathy

- extrinsics - https://nethackwiki.com/wiki/Property#Extrinsic_properties notable
    - reflection
    - magic resistance
    - life saving
    - waterwalking

- prayer - factors that affect prayer in NetHack
    - prayer timeout (starts at 300, then is a random number~[200,1000])
    - alignment record (negative don't pray)
    - luck (negarive don't pray)
    - gods anger (if angry don't pray)
    - polymorphy (don't pray when undead or a demon)
    - gehennom (don't pray in hell)