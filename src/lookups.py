import json
from pathlib import Path

from poke_env.battle.effect import Effect
from poke_env.battle.status import Status

# janky way to fetch the 36 pokemon in the game
# TODO: update this to have a more flexible pokemon description fetch
# maybe use the pokemon base species, ability, item and move set to
# create a hash that has a description for it
# should allow to add newer sets nicely

POKEMON = {
    "liquidation,lastrespects,aquajet,protect": 1,
    "superfang,feint,followme,protect": 2,
    "scaleshot,haze,tailwind,protect": 3,
    "flareblitz,knockoff,fakeout,partingshot": 4,
    "vacuumwave,earthpower,hypervoice,bloodmoon": 5,
    "makeitrain,shadowball,powergem,trick": 6,
    "expandingforce,dazzlinggleam,terablast,trickroom": 7,
    "psychic,followme,helpinghand,trickroom": 8,
    "eruption,heatwave,earthpower,weatherball": 9,
    "facade,earthquake,headlongrush,protect": 10,
    "sacredsword,psychocut,wideguard,trickroom": 11,
    "populationbomb,followme,taunt,protect": 12,
    "scaleshot,tailwind,haze,protect": 13,
    "makeitrain,shadowball,nastyplot,protect": 14,
    "bloodmoon,earthpower,hypervoice,vacuumwave": 15,
    "drainpunch,ragefist,bulkup,protect": 16,
    "superfang,beatup,followme,protect": 17,
    "woodhammer,grassyglide,highhorsepower,fakeout": 18,
    "tailwind,moonblast,encore,faketears": 19,
    "makeitrain,shadowball,thunderbolt,powergem": 20,
    "icywind,weatherball,muddywater,hurricane": 21,
    "fakeout,grassyglide,woodhammer,taunt": 22,
    "scaleshot,lowkick,protect,tailwind": 23,
    "electroshot,dracometeor,flashcannon,protect": 24,
    "nastyplot,makeitrain,shadowball,protect": 25,
    "closecombat,direclaw,coaching,protect": 26,
    "fakeout,grassyglide,drumbeating,highhorsepower": 27,
    "scaleshot,stompingtantrum,tailwind,protect": 28,
    "blizzard,icywind,encore,auroraveil": 29,
    "rockslide,flareblitz,extremespeed,protect": 30,
    "kowtowcleave,suckerpunch,swordsdance,protect": 31,
    "woodhammer,grassyglide,uturn,fakeout": 32,
    "followme,helpinghand,moonblast,protect": 33,
    "heatwave,gigadrain,quiverdance,protect": 34,
    "protect,fakeout,closecombat,direclaw": 35,
    "liquidation,fissure,yawn,curse": 36,
}

POKEMON_DESCRIPTION = {
    1: "Basculegion with Adaptability, which boosts STAB moves from 1.5x to 2x. Max Attack and Speed with an Adamant nature maximizes its offensive presence. Tera Ghost further boosts Last Respects.",
    2: "Maushold-Four with Friend Guard, which reduces damage to allies by 25%. Max HP and significant investment in Defense and Special Defense with a Jolly nature make it a bulky supporter.",
    3: "Dragonite with Multiscale, which halves damage taken when at full HP. Max Speed and high Attack with a Jolly nature make it a fast and powerful attacker.",
    4: "Incineroar with Intimidate, which lowers the opponent's Attack by one stage. Significant investment in HP and Special Defense with a Jolly nature makes it a bulky pivot that can switch in and weaken physical attackers.",
    5: "Ursaluna-Bloodmoon with Mind's Eye, which allows it to hit Ghost-types with Normal- and Fighting-type moves and ignore accuracy checks. Investment in HP, Special Attack, and Special Defense with a Modest nature makes it a bulky special attacker.",
    6: "Gholdengo with Good as Gold, which makes it immune to status moves. Investment in HP, Defense, Special Attack, and Special Defense with a Modest nature makes it a versatile and bulky special attacker.",
    7: "Hatterene with Magic Bounce, which reflects status moves back to the user. Max HP and Special Attack with a Quiet nature and 0 Speed IVs make it a powerful Trick Room attacker.",
    8: "Indeedee-F with Psychic Surge, which sets up Psychic Terrain when it enters the battle. Max Defense and significant investment in HP with a Relaxed nature and 0 Speed IVs make it a bulky Trick Room setter and supporter.",
    9: "Torkoal with Drought, which sets up sun when it enters the battle. Max HP and Special Attack with a Quiet nature and 0 Speed IVs make it a powerful Trick Room attacker that can abuse sun-boosted Fire-type moves.",
    10: "Ursaluna with Guts, which boosts its Attack by 1.5x when it has a status condition. Max HP and Attack with a Brave nature and 0 Speed IVs make it a powerful Trick Room attacker, especially when combined with the Flame Orb.",
    11: "Ga    print(len(ITEM_DESCRIPTIONS))llade with Sharpness, which boosts the power of slicing moves by 1.5x. Max Attack and significant investment in HP with a Brave nature and 0 Speed IVs make it a powerful Trick Room attacker.",
    12: "Maushold with Technician, which boosts the power of moves with 60 or less base power by 1.5x. Max Attack and Speed with a Jolly nature make it a fast attacker that can abuse Population Bomb.",
    13: "Dragonite with Multiscale, which halves damage taken when at full HP. Max Speed and high Attack with an Adamant nature make it a fast and powerful attacker.",
    14: "Gholdengo with Good as Gold, which makes it immune to status moves. Max Speed and significant investment in Special Attack and HP with a Timid nature make it a fast special attacker.",
    15: "Ursaluna-Bloodmoon with Mind's Eye, which allows it to hit Ghost-types with Normal- and Fighting-type moves and ignore accuracy checks. Investment in HP, Special Attack, and Special Defense with a Modest nature makes it a bulky special attacker.",
    16: "Annihilape with Defiant, which boosts its Attack by two stages when its stats are lowered. Max Speed and significant investment in HP and Attack with a Jolly nature make it a fast and powerful attacker that can punish Intimidate.",
    17: "Maushold-Four with Friend Guard, which reduces damage to allies by 25%. Max HP and significant investment in Defense and Speed with a Jolly nature make it a bulky supporter.",
    18: "Rillaboom with Grassy Surge, which sets up Grassy Terrain when it enters the battle. Significant investment in HP and Attack with an Adamant nature makes it a bulky attacker that can abuse Grassy Glide.",
    19: "Whimsicott with Prankster, which gives its status moves +1 priority. Max Special Attack and Speed with a Timid nature make it a fast supporter that can set up Tailwind or disrupt the opponent with Encore.",
    20: "Gholdengo with Good as Gold, which makes it immune to status moves. Max Speed and Special Attack with a Modest nature make it a fast and powerful special attacker.",
    21: "Pelipper with Drizzle, which sets up rain when it enters the battle. Max Special Attack and Speed with a Timid nature make it a fast special attacker that can abuse rain-boosted Water-type moves.",
    22: "Rillaboom with Grassy Surge, which sets up Grassy Terrain when it enters the battle. Significant investment in HP and Attack with an Adamant nature makes it a bulky attacker that can abuse Grassy Glide.",
    23: "Dragonite with Multiscale, which halves damage taken when at full HP. Max Attack and Speed with an Adamant nature make it a powerful and fast attacker.",
    24: "Archaludon with Sturdy, which allows it to survive a hit that would otherwise KO it with 1 HP if it has full HP. Max Special Attack and Speed with a Timid nature make it a fast special attacker.",
    25: "Gholdengo with Good as Gold, which makes it immune to status moves. Significant investment in HP and Speed with a Modest nature make it a fast and bulky special attacker.",
    26: "Sneasler with Unburden, which doubles its Speed when it loses its held item. Investment in HP, Attack, and Speed with an Adamant nature makes it a fast physical attacker, especially after its White Herb is consumed.",
    27: "Rillaboom with Grassy Surge, which sets up Grassy Terrain when it enters the battle. Max Speed and significant investment in Attack and HP with a Jolly nature make it a fast and bulky attacker.",
    28: "Dragonite with Multiscale, which halves damage taken when at full HP. Max Attack and Speed with an Adamant nature make it a powerful and fast attacker.",
    29: "Ninetales-Alola with Snow Warning, which sets up snow when it enters the battle. Max Special Attack and Speed with a Timid nature make it a fast special attacker that can set up Aurora Veil.",
    30: "Arcanine-Hisui with Intimidate, which lowers the opponent's Attack by one stage. Max Attack and Speed with a Jolly nature make it a fast physical attacker that can weaken opposing physical threats.",
    31: "Kingambit with Defiant, which boosts its Attack by two stages when its stats are lowered. Max HP and Attack with an Adamant nature make it a bulky and powerful attacker that can punish Intimidate.",
    32: "Rillaboom with Grassy Surge, which sets up Grassy Terrain when it enters the battle. Max HP and significant investment in Attack and Special Defense with an Adamant nature make it a bulky attacker that can abuse Grassy Glide.",
    33: "Clefable with Unaware, which ignores the opponent's stat changes when attacking. Max HP and Defense with a Bold nature make it a very bulky supporter that can't be easily broken by setup sweepers.",
    34: "Volcarona with Flame Body, which has a 30% chance to burn an attacker that makes contact. Significant investment in HP, Defense, and Speed with a Modest nature make it a bulky setup sweeper.",
    35: "Sneasler with Unburden, which doubles its Speed when it loses its held item. Max Speed and significant investment in HP and Attack with an Adamant nature makes it a fast physical attacker, especially after its Grassy Seed is consumed.",
    36: "Dondozo with Unaware, which ignores the opponent's stat changes when attacking. Max HP and Special Defense with a Careful nature make it a very bulky wall that can't be easily broken by setup sweepers.",
}

ITEM_PATH = Path(__file__).parent.parent / "data" / "items.json"

with ITEM_PATH.open() as f:
    ITEM_DESCRIPTION = json.load(f)

STATUS_DESCRIPTION = {
    Status.BRN: "This pokemon is burned. Burn reduces the Pokemon's Attack stat by 50% and causes them to lose 6.25% of their maximum HP at the end of each turn. Fire-type Pokemon cannot be burned",
    Status.FNT: "The Pokemon has 0 HP and cannot battle. Fainted Pokemon must be switched out and cannot use moves or take actions until revived.",
    Status.FRZ: "This pokemon is frozen. Freeze causes the pokemon to not move. Has a 20% chance to thaw each turn, and will immediately thaw if hit by a Fire-type move. Ice-type Pokemon cannot be frozen.",
    Status.PAR: "This pokemon is paralysed. Paralysis reduces the Pokemon's Speed stat by 50% and gives them a 25% chance to be fully paralyzed (unable to move) each turn. Electric-type Pokemon cannot be paralyzed.",
    Status.PSN: "This pokemon is poisoned. Poison causes the Pokemon to lose 12.5% of their maximum HP at the end of each turn. Poison and Steel-type Pokemon cannot be poisoned.",
    Status.SLP: "This pokemon is asleep. Sleep prevents the Pokemon from using moves for 1-3 turns (duration chosen randomly). The Pokemon will wake up automatically when the sleep duration ends or are cured by a move.",
    Status.TOX: "This poison is badly poisoned. It causes the pokemon to take increasing damage each turn: 6.25% max HP on turn 1, then 12.5%, then 18.75%, and so on. The damage counter resets if the Pokemon switches out.",
}

EFFECT_DESCRIPTION = {
    Effect.CONFUSION: "Pokemon is confused and has a 33% chance to hurt itself instead of using moves. Lasts 2-5 turns.",
    Effect.TAUNT: "Pokemon has been taunted. It can only use damaging moves for 3 turns.",
    Effect.ENCORE: "Pokemon is forced to repeat the last used move for 3 turns.",
    Effect.YAWN: "Pokemon will fall asleep after 1 more turn unless switched out.",
}

MOVE_PATH = Path(__file__).parent.parent / "data" / "moves.json"

with MOVE_PATH.open() as f:
    MOVES = json.load(f)


if __name__ == "__main__":
    print(f"Number of moves: {len(MOVES)}")
    print(f"Number of items: {len(ITEM_DESCRIPTION)}")
