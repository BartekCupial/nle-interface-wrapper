from collections import defaultdict

from nle import nethack as nh

from nle_interface_wrapper.wrappers.inventory.properties import ArmorClass, ItemCategory

# collect all objects by their class
class_to_objects = defaultdict(list)
for glyph in range(nh.GLYPH_OBJ_OFF, nh.NUM_OBJECTS + nh.GLYPH_OBJ_OFF):
    obj = nh.objclass(nh.glyph_to_obj(glyph))
    obj_class = obj.oc_class
    if nh.OBJ_NAME(obj):
        class_to_objects[obj_class].append((obj, glyph))


NAME_TO_OBJECTS = defaultdict(list)


def add_obj(key, obj, glyph, prefixes=[""], suffixes=[""]):
    assert isinstance(prefixes, list)
    assert isinstance(suffixes, list)
    if key:
        for prefix in prefixes:
            for suffix in suffixes:
                NAME_TO_OBJECTS[prefix + key + suffix].append((obj, glyph))


for glyph in range(nh.GLYPH_OBJ_OFF, nh.NUM_OBJECTS + nh.GLYPH_OBJ_OFF):
    obj_name = ""
    obj_description = ""

    obj = nh.objclass(nh.glyph_to_obj(glyph))
    obj_class = obj.oc_class

    if nh.OBJ_NAME(obj) is not None:
        obj_name = nh.OBJ_NAME(obj)
    if nh.OBJ_DESCR(obj) is not None:
        obj_description = nh.OBJ_DESCR(obj)

    match ItemCategory(ord(obj_class)):
        case ItemCategory.ILLOBJ:
            add_obj(obj_name, obj, glyph)
            add_obj(obj_description, obj, glyph)
        case ItemCategory.WEAPON:
            add_obj(obj_name, obj, glyph)
            add_obj(obj_description, obj, glyph)
        case ItemCategory.ARMOR:
            add_obj(obj_name, obj, glyph)
            add_obj(obj_description, obj, glyph)
        case ItemCategory.RING:
            add_obj(obj_name, obj, glyph, ["ring of "])

            for obj, glyph in class_to_objects[obj_class]:
                add_obj(obj_description, obj, glyph, suffixes=[" ring"])
        case ItemCategory.AMULET:
            if obj_name in ["Amulet of Yendor"]:
                add_obj(obj_name, obj, glyph)
            elif obj_description in ["Amulet of Yendor"]:
                add_obj(obj_name, obj, glyph)
                add_obj(obj_description, obj, glyph)
            else:
                add_obj(obj_name, obj, glyph)
                if obj_name not in ["Amulet of Yendor", "cheap plastic imitation of the Amulet of Yendor"]:
                    for obj, glyph in class_to_objects[obj_class]:
                        if nh.OBJ_NAME(obj) not in [
                            "Amulet of Yendor",
                            "cheap plastic imitation of the Amulet of Yendor",
                        ]:
                            add_obj(obj_description, obj, glyph, suffixes=[" amulet"])
                else:
                    add_obj(obj_description, obj, glyph, suffixes=[" amulet"])
        case ItemCategory.TOOL:
            if obj_name in ["lenses"]:
                add_obj(obj_name, obj, glyph, ["pair of "])
            else:
                add_obj(obj_name, obj, glyph)

            add_obj(obj_description, obj, glyph)
        case ItemCategory.COMESTIBLES:
            add_obj(obj_name, obj, glyph)
            add_obj(obj_description, obj, glyph)
        case ItemCategory.POTION:
            add_obj(obj_name, obj, glyph, ["potion of "])
            if obj_name not in ["water"]:
                for obj, glyph in class_to_objects[obj_class]:
                    if nh.OBJ_NAME(obj) not in ["water"]:
                        add_obj(obj_description, obj, glyph, suffixes=[" potion"])
            else:
                add_obj(obj_description, obj, glyph, suffixes=[" potion"])
        case ItemCategory.SCROLL:
            add_obj(obj_name, obj, glyph, ["scroll of "])
            if obj_name not in ["mail", "blank paper"]:
                for obj, glyph in class_to_objects[obj_class]:
                    if nh.OBJ_NAME(obj) not in ["mail", "blank paper"]:
                        add_obj(obj_description, obj, glyph, ["scroll labeled "])
            else:
                if obj_description in ["unlabeled"]:
                    add_obj(obj_description, obj, glyph, suffixes=[" scroll"])
                else:
                    add_obj(obj_description, obj, glyph, ["scroll labeled "])
        case ItemCategory.SPELLBOOK:
            if obj_name in ["Book of the Dead", "novel"]:
                add_obj(obj_name, obj, glyph)
            else:
                add_obj(obj_name, obj, glyph, ["spellbook of "])

            if obj_description in ["paperback"]:
                add_obj(obj_description, obj, glyph, suffixes=[" book"])
            else:
                if obj_name not in ["novel", "blank paper", "Book of the Dead"]:
                    for obj, glyph in class_to_objects[obj_class]:
                        if nh.OBJ_NAME(obj) not in ["novel", "blank paper", "Book of the Dead"]:
                            add_obj(obj_description, obj, glyph, suffixes=[" spellbook"])
                else:
                    add_obj(obj_description, obj, glyph, suffixes=[" spellbook"])
        case ItemCategory.WAND:
            add_obj(obj_name, obj, glyph, ["wand of "])
            for obj, glyph in class_to_objects[obj_class]:
                add_obj(obj_description, obj, glyph, suffixes=[" wand"])
        case ItemCategory.COIN:
            add_obj(obj_name, obj, glyph)
            add_obj(obj_description, obj, glyph)
        case ItemCategory.GEM:
            if obj_name in [
                "luckstone",
                "loadstone",
                "touchstone",
                "rock",
            ]:
                add_obj(obj_name, obj, glyph)
                add_obj(obj_description, obj, glyph, suffixes=[" stone"])
            elif obj_name in ["flint"]:
                add_obj(obj_name, obj, glyph, suffixes=[" stone"])
                add_obj(obj_description, obj, glyph, suffixes=[" stone"])
            else:
                add_obj(obj_name, obj, glyph)
                add_obj(obj_description, obj, glyph, suffixes=[" gem"])
        case ItemCategory.ROCK:
            # only boulder and statue
            add_obj(obj_name, obj, glyph)
            add_obj(obj_description, obj, glyph, suffixes=[" stone"])
        case ItemCategory.BALL:
            add_obj(obj_name, obj, glyph)  # heavy iron ball
            add_obj(obj_name, obj, glyph, ["very "])  # very heavy iron ball
        case ItemCategory.CHAIN:
            add_obj(obj_name, obj, glyph)  # only iron chain
        case ItemCategory.VENOM:
            # only blinding venom, acid venom
            add_obj(obj_name, obj, glyph)
            add_obj(obj_description, obj, glyph)
        case _:
            add_obj(obj_name, obj, glyph)
            add_obj(obj_description, obj, glyph)


artifacts = [
    ("Excalibur", "long sword"),
    ("Stormbringer", "runesword"),
    ("Mjollnir", "war hammer"),
    ("Cleaver", "battle-axe"),
    ("Grimtooth", "orcish dagger"),
    ("Orcrist", "elven broadsword"),
    ("Sting", "elven dagger"),
    ("Magicbane", "athame"),
    ("Frost Brand", "long sword"),
    ("Fire Brand", "long sword"),
    ("Dragonbane", "broadsword"),
    ("Demonbane", "long sword"),
    ("Werebane", "silver saber"),
    ("Grayswandir", "silver saber"),
    ("Giantslayer", "long sword"),
    ("Ogresmasher", "war hammer"),
    ("Trollsbane", "morning star"),
    ("Vorpal Blade", "long sword"),
    ("Snickersnee", "katana"),
    ("Sunsword", "long sword"),
    ("The Orb of Detection", "crystal ball"),
    ("The Heart of Ahriman", "luckstone"),
    ("The Sceptre of Might", "mace"),
    ("The Palantir of Westernesse", "crystal ball"),
    ("The Staff of Aesculapius", "quarterstaff"),
    ("The Magic Mirror of Merlin", "mirror"),
    ("The Eyes of the Overworld", "pair of lenses"),
    ("The Mitre of Holiness", "helm of brilliance"),
    ("The Longbow of Diana", "bow"),
    ("The Master Key of Thievery", "skeleton key"),
    ("The Tsurugi of Muramasa", "tsurugi"),
    ("The Platinum Yendorian Express Card", "credit card"),
    ("The Orb of Fate", "crystal ball"),
    ("The Eye of the Aethiopica", "amulet of ESP"),
]

for artifact, appearance in artifacts:
    assert len(NAME_TO_OBJECTS[appearance]) == 1
    for obj, glyph in NAME_TO_OBJECTS[appearance]:
        add_obj(artifact, obj, glyph, prefixes=["", f"{appearance} named "])

        obj_description = nh.OBJ_DESCR(obj)
        if obj_description:
            match ItemCategory(ord(obj.oc_class)):
                case ItemCategory.ARMOR:
                    add_obj(artifact, obj, glyph, prefixes=["helmet named "])
                case ItemCategory.AMULET:
                    add_obj(artifact, obj, glyph, prefixes=["amulet named "])
                case ItemCategory.GEM:
                    add_obj(artifact, obj, glyph, prefixes=[f"{obj_description} stone named "])
                case _:
                    add_obj(artifact, obj, glyph, prefixes=[f"{obj_description} named "])


novels = [
    "The Colour of Magic",
    "The Light Fantastic",
    "Equal Rites",
    "Mort",
    "Sourcery",
    "Wyrd Sisters",
    "Pyramids",
    "Guards! Guards!",
    "Eric",
    "Moving Pictures",
    "Reaper Man",
    "Witches Abroad",
    "Small Gods",
    "Lords and Ladies",
    "Men at Arms",
    "Soul Music",
    "Interesting Times",
    "Maskerade",
    "Feet of Clay",
    "Hogfather",
    "Jingo",
    "The Last Continent",
    "Carpe Jugulum",
    "The Fifth Elephant",
    "The Truth",
    "Thief of Time",
    "The Last Hero",
    "The Amazing Maurice and His Educated Rodents",
    "Night Watch",
    "The Wee Free Men",
    "Monstrous Regiment",
    "A Hat Full of Sky",
    "Going Postal",
    "Thud!",
    "Wintersmith",
    "Making Money",
    "Unseen Academicals",
    "I Shall Wear Midnight",
    "Snuff",
    "Raising Steam",
    "The Shepherd's Crown",
]
appearance = "paperback book"
assert len(NAME_TO_OBJECTS[appearance]) == 1
for novel in novels:
    for obj, glyph in NAME_TO_OBJECTS[appearance]:
        add_obj(novel, obj, glyph, prefixes=["", f"{appearance} named "])


scrolls = [name.removeprefix("scroll labeled ") for name in NAME_TO_OBJECTS.keys() if "scroll labeled" in name]


name_to_monsters = {}
for glyph in range(nh.GLYPH_MON_OFF, nh.GLYPH_MON_OFF + nh.NUMMONS):
    permonst = nh.permonst(glyph)
    name_to_monsters[permonst.mname] = permonst


NAME_TO_GLYPHS = {key: [glyph for obj, glyph in inner] for key, inner in NAME_TO_OBJECTS.items()}

NAME_TO_OBJECTS = {key: [obj for obj, glyph in inner] for key, inner in NAME_TO_OBJECTS.items()}


if __name__ == "__main__":

    def not_none(s):
        return s if s is not None else "None"

    for key, objs in NAME_TO_OBJECTS.items():
        # if len(objs) > 1:
        print(key)
        print("\n".join([str((not_none(nh.OBJ_NAME(obj)), not_none(nh.OBJ_DESCR(obj)))) for obj in objs]))
        print("")
