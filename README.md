# nle-interface-wrapper

Interface Wrapper for [Nethack Learning Environment (NLE)](https://github.com/facebookresearch/nle)

### Installation

```
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

### How to use?

```bash
import gymnasium as gym
import nle

from nle_interface_wrapper.wrappers import Properties, AddTextMap, AddTextOverview, AddTextInventory, AddTextSpells, AddTextSkills, AutoMore

env = gym.make("NetHackChallenge-v0")
env = AutoMore(env)

env = Properties(env)
env = AddTextOverview(env)
env = AddTextMap(env)
env = AddTextInventory(env)
env = AddTextSpells(env)
env = AddTextSkills(env)

obs, info = env.reset()
print(obs["text_overview"])
print(obs["text_map"])
print(obs["text_inventory"])
print(obs["text_spells"])
print(obs["text_skills"])
```

example output
```
>>> print(obs["text_overview"])
The Dungeons of Doom:
Level 1: <- You are here.
A sink.

>>> print(obs["text_map"])
Explored room with 2 exits: <- You are here.
    Objects: a sink.
>>> print(obs["text_inventory"])
weapons:
    a) a +1 long sword (weapon in hand)
    b) a +1 lance (alternate weapon; not wielded)
armor:
    c) an uncursed +1 ring mail (being worn)
    d) an uncursed +0 helmet (being worn)
    e) an uncursed +0 small shield (being worn)
    f) an uncursed +0 pair of leather gloves (being worn)
comestibles:
    g) 11 uncursed apples
    h) 11 uncursed carrots
>>> print(obs["text_spells"])

>>> print(obs["text_skills"])
dagger: Basic, knife: Basic, axe: Skilled, pick-axe: Basic, short sword: Skilled, broadsword: Skilled, long sword: Expert, two-handed sword: Skilled, scimitar: Basic, saber: Skilled, club: Basic, mace: Skilled, morning star: Skilled, flail: Basic, hammer: Basic, polearms: Skilled, spear: Skilled, trident: Basic, lance: Expert, bow: Basic, crossbow: Skilled, attack spells: Skilled, healing spells: Skilled, clerical spells: Skilled, martial arts: Expert, two weapon combat: Skilled, riding: Expert
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
