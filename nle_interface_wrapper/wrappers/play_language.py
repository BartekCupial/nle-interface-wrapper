import os
import readline
from functools import partial

import gymnasium as gym
from nle.language_wrapper.wrappers import nle_language_wrapper


def completer(text, state, commands=[]):
    options = [cmd for cmd in commands if cmd.startswith(text)]
    return options[state] if state < len(options) else None


def setup_autocomplete(completer_fn):
    os.system("stty sane")  # forces the terminal back to “cooked” mode
    readline.parse_and_bind("tab: complete")
    print("Type commands and use TAB to autocomplete.")
    print("To see available actions use command: `help`")
    readline.set_completer(completer_fn)


class PlayLanguage(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.action_str_enum_map = {}
        self.action_enum_index_map = {}
        self.action_str_desc_map = {}

        all_action_strs = [
            action_str
            for action_strs in nle_language_wrapper.NLELanguageWrapper.all_nle_action_map.values()
            for action_str in action_strs
        ]
        assert all(key in all_action_strs for key in NLE_ACTIONS_TO_DESCR), ", ".join(
            [key for key in NLE_ACTIONS_TO_DESCR if key not in all_action_strs]
        )

        for action_enum in self.env.unwrapped.actions:
            for action_str in nle_language_wrapper.NLELanguageWrapper.all_nle_action_map[action_enum]:
                if action_str not in NLE_ACTIONS_TO_DESCR:
                    continue

                self.action_str_enum_map[action_str] = action_enum
                self.action_enum_index_map[action_enum] = self.env.unwrapped.actions.index(action_enum)
                self.action_str_desc_map[action_str] = NLE_ACTIONS_TO_DESCR[action_str]

    def get_action(self, obs=None):
        names = list(self.action_str_enum_map.keys())
        setup_autocomplete(partial(completer, commands=names))

        while True:
            command = input("> ")

            if command == "help":
                for name in names:
                    print(name)
                continue
            else:
                try:
                    action = self.env.unwrapped.actions.index(self.action_str_enum_map[command])
                    break
                except ValueError:
                    print(f"Selected action '{command}' is not in action list. Please try again.")
                    continue

        return action


NLE_ACTIONS_TO_DESCR = {
    "north": "move north",
    "east": "move east",
    "south": "move south",
    "west": "move west",
    "northeast": "move northeast",
    "southeast": "move southeast",
    "southwest": "move southwest",
    "northwest": "move northwest",
    "far north": "move far north",
    "far east": "move far east",
    "far south": "move far south",
    "far west": "move far west",
    "far northeast": "move far northeast",
    "far southeast": "move far southeast",
    "far southwest": "move far southwest",
    "far northwest": "move far northwest",
    "up": "go up a staircase",
    "down": "go down a staircase (tip: you can only go down if you are standing on the stairs)",
    "wait": "rest one move while doing nothing",
    "more": "display more of the message (tip: ONLY ever use when current message ends with --More--)",
    "annotate": "leave a note about the level",
    "apply": "apply (use) a tool",
    "call": "name a monster or object, or add an annotation",
    "cast": "cast a spell",
    "close": "close an adjacent door",
    "open": "open an adjacent door",
    "dip": "dip an object into something",
    "drop": "drop an item",
    "droptype": "drop specific item types (specify in the next prompt)",
    "eat": "eat something (tip: replenish food when hungry)",
    "esc": "exit menu or message",
    "engrave": "engrave writing on the floor (tip: Elbereth)",
    "enhance": "advance or check weapons skills",
    "fire": "fire ammunition from quiver",
    "fight": "fight a monster (even if you only guess one is there)",
    "force": "force a lock",
    "inventory": "show your inventory",
    "invoke": "invoke ",
    "jump": "jump to a location",
    "kick": "kick an enemy or a locked door or chest",
    "look": "look at what is under you",
    "loot": "loot a box on the floor",
    "monster": "use a monster's special ability (when polymorphed)",
    "offer": "offer a sacrifice to the gods (tip: on an aligned altar)",
    # "overview": "display an overview of the dungeon",
    "pay": "pay your shopping bill",
    "pickup": "pick up things at the current location",
    "pray": "pray to the gods for help",
    "puton": "put on an accessory",
    "quaff": "quaff (drink) something",
    "quiver": "select ammunition for quiver",
    "read": "read a scroll or spellbook",
    "remove": "remove an accessory",
    "rub": "rub a lamp or a stone",
    "search": "search for hidden doors and passages",
    "swap": "swap wielded and secondary weapons",
    "takeoff": "take off one piece of armor",
    "takeoffall": "take off all armor",
    "teleport": "teleport to another level (if you have the ability)",
    "throw": "throw something (e.g. a dagger or dart)",
    "travel": "travel to a specific location on the map (tip: in the next action, specify > or < for stairs, { for fountain, and _ for altar)",
    "twoweapon": "toggle two-weapon combat",
    "untrap": "untrap something",
    "wear": "wear a piece of armor",
    "wield": "wield a weapon",
    "wipe": "wipe off your face",
    "zap": "zap a wand",
    "minus": "-",
    "space": " ",
    "apos": "'",
    "0": "0",
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
}
