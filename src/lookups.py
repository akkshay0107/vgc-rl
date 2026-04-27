import json
from pathlib import Path

from poke_env.battle.effect import Effect
from poke_env.battle.status import Status

TINYBERT_SZ = 624
EXTRA_SZ = 104
# 1 field row + 1 info row + (for each pokemon (textA, textB, num) so 12 * 3) + 1 field num = 39
# + 3 static history embeddings = 42
OBS_DIM = (42, TINYBERT_SZ)

# Action space parameters
NUM_SWITCHES = 6
NUM_MOVES = 4
NUM_TARGETS = 5
NUM_GIMMICKS = 1
ACT_SIZE = 1 + NUM_SWITCHES + NUM_MOVES * NUM_TARGETS * (NUM_GIMMICKS + 1)

# janky way to fetch the 36 pre selected pokemon
# TODO: update this to have a more flexible pokemon description fetch
# maybe use the pokemon base species, ability, item and move set to
# create a hash that has a description for it
# should allow to add newer sets nicely

# two gholdengos get mapped to the same movelist
# but have similar spreads and evs aren't information available
# from battle anyways, so just had them share the same description
POKEMON = {
    "expandingforce,dazzlinggleam,terablast,trickroom": 1,
    "psychic,followme,helpinghand,trickroom": 2,
    "eruption,heatwave,earthpower,weatherball": 3,
    "facade,earthquake,headlongrush,protect": 4,
    "sacredsword,psychocut,wideguard,trickroom": 5,
    "populationbomb,followme,taunt,protect": 6,
    "scaleshot,tailwind,haze,protect": 7,
    "makeitrain,shadowball,nastyplot,protect": 8,
    "bloodmoon,earthpower,hypervoice,vacuumwave": 9,
    "drainpunch,ragefist,bulkup,protect": 10,
    "superfang,beatup,followme,protect": 11,
    "woodhammer,grassyglide,highhorsepower,fakeout": 12,
    "weatherball,hurricane,muddywater,icywind": 13,
    "electroshot,dracometeor,thunderbolt,protect": 14,
    "liquidation,lastrespects,aquajet,protect": 15,
    "moonblast,taunt,faketears,tailwind": 16,
    "makeitrain,shadowball,terablast,protect": 17,
    "knockoff,flareblitz,fakeout,uturn": 18,
    "nastyplot,makeitrain,shadowball,protect": 19,
    "closecombat,direclaw,coaching,protect": 20,
    "fakeout,grassyglide,drumbeating,highhorsepower": 21,
    "scaleshot,stompingtantrum,tailwind,protect": 22,
    "blizzard,icywind,encore,auroraveil": 23,
    "rockslide,flareblitz,extremespeed,protect": 24,
    "eruption,shadowball,heatwave,overheat": 25,
    "moonblast,tailwind,sunnyday,encore": 26,
    "flareblitz,knockoff,fakeout,partingshot": 27,
    "bloodmoon,earthpower,hypervoice,protect": 28,
    "psychic,nightshade,helpinghand,trickroom": 29,
    "closecombat,feint,wideguard,detect": 30,
    "thunderbolt,airslash,tailwind,protect": 31,
    "facade,headlongrush,earthquake,protect": 32,
    "terablast,icebeam,recover,trickroom": 33,
    "flareblitz,knockoff,uturn,fakeout": 34,
    "pollenpuff,spore,ragepowder,protect": 35,
}

# synthetic data (just there to give each pokemon a different embedding)
POKEMON_DESCRIPTION = {
    1: "Hatterene with Magic Bounce, which reflects status moves back at the user. Max HP and Special Attack with a Quiet nature and 0 Speed IVs make it a dedicated Trick Room attacker that threatens huge Expanding Force damage under Psychic Terrain.",
    2: "Indeedee-F with Psychic Surge, which sets Psychic Terrain on entry. Heavy HP and Defense investment with a Relaxed nature and 0 Speed IVs make it a bulky redirection support that helps set Trick Room safely.",
    3: "Torkoal with Drought, which sets sun on entry. Max HP and Special Attack with a Quiet nature and 0 Speed IVs make it a classic Trick Room wallbreaker that pressures teams with sun-boosted Eruption.",
    4: "Ursaluna with Guts, which boosts its Attack by 1.5x while statused. Max HP and Attack with a Brave nature and 0 Speed IVs make it a slow Trick Room breaker that abuses Flame Orb-boosted Facade and Ground coverage.",
    5: "Gallade with Sharpness, which boosts slicing moves. Heavy HP and max Attack with a Brave nature and 0 Speed IVs make it a slow attacker that also gives the team Wide Guard and a secondary Trick Room option.",
    6: "Maushold with Technician, which boosts low-base-power moves. Max Attack and Speed with a Jolly nature make it a fast support attacker that threatens Population Bomb while also offering Follow Me and Taunt.",
    7: "Dragonite with Multiscale, which halves damage taken at full HP. Max Speed with strong Attack investment and an Adamant nature make it an offensive Tailwind setter that can also use Haze to reset setup.",
    8: "Gholdengo with Good as Gold, which blocks status moves. A Timid nature with max Speed and strong Special Attack investment makes it a fast offensive setup sweeper built around Nasty Plot and Make It Rain.",
    9: "Ursaluna-Bloodmoon with Mind's Eye, which lets it hit Ghost-types with Normal- and Fighting-type moves and ignore accuracy checks. Its HP, Special Attack, and special bulk investment with a Modest nature make it a sturdy special attacker.",
    10: "Annihilape with Defiant, which raises Attack by two stages when its stats are lowered. Max Speed with useful HP and Attack investment and a Jolly nature make it a fast win-condition that punishes Intimidate and snowballs with Rage Fist.",
    11: "Maushold-Four with Friend Guard, which reduces damage taken by its ally. High HP with Defense and Speed investment and a Jolly nature make it a disruptive support that enables partners with Follow Me and Beat Up.",
    12: "Rillaboom with Grassy Surge, which sets Grassy Terrain on entry. Its HP and Attack investment with an Adamant nature make it a bulky terrain attacker that offers Fake Out pressure and strong priority with Grassy Glide.",
    13: "Pelipper with Drizzle, which sets rain on entry. Max Special Attack and Speed with a Timid nature make it a fast rain attacker that also provides speed control with Icy Wind.",
    14: "Archaludon with Sturdy, which lets it survive from full HP. Max Special Attack and Speed with a Timid nature make it an aggressive rain abuser that can fire immediate Power Herb Electro Shot pressure.",
    15: "Basculegion with Adaptability, which boosts STAB attacks from 1.5x to 2x. High Attack and max Speed with an Adamant nature make it a fast physical cleaner that hits especially hard with Last Respects.",
    16: "Whimsicott with Prankster, which gives priority to its status moves. Its bulk-focused spread with enough Speed and a Timid nature makes it a utility support that can disrupt with Taunt, amplify damage with Fake Tears, and set Tailwind.",
    17: "Gholdengo with Good as Gold, which blocks status moves. Max Special Attack and Speed with a Modest nature make it an immediate offensive threat that trades setup for direct coverage with Tera Blast.",
    18: "Incineroar with Intimidate, which lowers opposing Attack on entry. Its Assault Vest set with HP, Attack, and high Speed investment makes it an offensive pivot that still provides Fake Out and U-turn utility.",
    19: "Gholdengo with Good as Gold, which blocks status moves. Its bulky Grassy Seed set with high Speed and a Modest nature is a setup-oriented special attacker that becomes harder to remove after terrain activation.",
    20: "Sneasler with Unburden, which doubles its Speed after its item is consumed. Its HP, Attack, Defense, and Speed investment with an Adamant nature make it a flexible physical attacker that can also support with Coaching.",
    21: "Rillaboom with Grassy Surge, which sets Grassy Terrain on entry. Max Speed with solid HP and Attack investment and a Jolly nature make it a faster utility attacker that combines Fake Out with Drum Beating speed control.",
    22: "Dragonite with Multiscale, which halves damage taken at full HP. Max Attack and Speed with an Adamant nature make it a straightforward attacker that gains Ground coverage through Stomping Tantrum.",
    23: "Ninetales-Alola with Snow Warning, which sets snow on entry. Max Special Attack and Speed with a Timid nature make it a fast support attacker that enables Aurora Veil and disrupts with Encore.",
    24: "Arcanine-Hisui with Intimidate, which lowers opposing Attack on entry. Max Attack and Speed with a Jolly nature make it an aggressive physical attacker that adds fast Rock Slide pressure and Extreme Speed utility.",
    25: "Typhlosion-Hisui with Blaze, which powers up Fire-type moves at low HP. Max Special Attack and Speed with a Timid nature make it a fast sun attacker that pressures teams heavily with Choice Specs Eruption.",
    26: "Whimsicott with Prankster, which gives priority to its status moves. Its bulky Calm spread makes it a durable support that can set Tailwind, change weather with Sunny Day, and punish passive turns with Encore.",
    27: "Incineroar with Intimidate, which lowers opposing Attack on entry. Its physically bulky Safety Goggles set provides standard utility with Fake Out and Parting Shot while staying strong into powder moves.",
    28: "Ursaluna-Bloodmoon with Mind's Eye, which lets it hit Ghost-types with Normal- and Fighting-type moves and ignore accuracy checks. Its bulky Modest spread with Speed investment makes it a strong mid-speed special breaker under Tailwind or Trick Room support.",
    29: "Farigiraf with Armor Tail, which blocks opposing priority moves aimed at its side. Its mixed bulk with a small Speed creep and a Bold nature make it a reliable Trick Room setter and support piece.",
    30: "Flamigo with Scrappy, which lets it hit Ghost-types with Fighting-type moves. Max Attack and Speed with a Jolly nature make it a fast utility attacker that brings both Feint and Wide Guard support.",
    31: "Kilowattrel with Competitive, which raises Special Attack by two stages when one of its stats is lowered. Max Special Attack and Speed with a Timid nature make it a fragile but fast Tailwind attacker that can punish Intimidate cycling.",
    32: "Ursaluna with Guts, which boosts its Attack by 1.5x while statused. Its Adamant spread with substantial Speed investment makes it a faster Flame Orb attacker that functions better outside full Trick Room.",
    33: "Porygon2 with Download, which can raise Attack or Special Attack depending on opposing defenses. Its Eviolite-boosted bulk and mixed defensive spread make it a durable Trick Room setter that still threatens damage with Tera Blast and Ice Beam.",
    34: "Incineroar with Intimidate, which lowers opposing Attack on entry. Its Assault Vest spread with HP, Attack, and Speed investment makes it a more offensive pivot that still compresses Fake Out and U-turn.",
    35: "Amoonguss with Regenerator, which restores HP when it switches out. Its physically and specially bulky spread makes it a durable redirection support that can heal allies with Pollen Puff and control games with Spore.",
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
}

EFFECT_DESCRIPTION = {
    Effect.CONFUSION: "Pokemon is confused and has a 33% chance to hurt itself instead of using moves. Lasts 2-5 turns.",
    Effect.TAUNT: "Pokemon has been taunted. It can only use damaging moves for 3 turns.",
    Effect.ENCORE: "Pokemon is forced to repeat the last used move for 3 turns.",
}

MOVE_PATH = Path(__file__).parent.parent / "data" / "moves.json"

with MOVE_PATH.open() as f:
    _MOVES_RAW = json.load(f)
    MOVES = {k: json.dumps(v, separators=(",", ":")) for k, v in _MOVES_RAW.items()}


if __name__ == "__main__":
    print(f"Number of moves: {len(MOVES)}")
    print(f"Number of items: {len(ITEM_DESCRIPTION)}")
