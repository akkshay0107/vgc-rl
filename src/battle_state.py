import torch
import poke_env.data.gen_data as gen_data

DEX = gen_data.GenData(9)
NON_VOLATILE_STATUS = ['none', 'par', 'brn', 'slp', 'frz', 'psn']
STAT_STAGES = [0.25, 0.285, 0.33, 0.4, 0.5, 0.66, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
POKEMON = {
    0: """Basculegion @ Focus Sash
Ability: Adaptability
Level: 50
Tera Type: Ghost
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Liquidation
- Last Respects
- Aqua Jet
- Protect
""",
    1: """Maushold-Four @ Rocky Helmet
Ability: Friend Guard
Level: 50
Tera Type: Poison
EVs: 252 HP / 4 Atk / 180 Def / 20 SpD / 52 Spe
Jolly Nature
- Super Fang
- Feint
- Follow Me
- Protect
""",
    2: """Dragonite @ Loaded Dice
Ability: Multiscale
Level: 50
Tera Type: Fairy
EVs: 44 HP / 204 Atk / 4 Def / 4 SpD / 252 Spe
Jolly Nature
- Scale Shot
- Tailwind
- Haze
- Protect
""",
    3: """Incineroar @ Safety Goggles
Ability: Intimidate
Level: 50
Tera Type: Grass
EVs: 196 HP / 4 Atk / 4 Def / 68 SpD / 236 Spe
Jolly Nature
- Flare Blitz
- Knock Off
- Fake Out
- Parting Shot
""",
    4: """Ursaluna-Bloodmoon @ Assault Vest
Ability: Mind's Eye
Level: 50
Tera Type: Fire
EVs: 156 HP / 4 Def / 116 SpA / 100 SpD / 132 Spe
Modest Nature
IVs: 0 Atk
- Blood Moon
- Earth Power
- Hyper Voice
- Vacuum Wave
""",
    5: """Gholdengo @ Choice Specs
Ability: Good as Gold
Level: 50
Tera Type: Steel
EVs: 228 HP / 84 Def / 52 SpA / 60 SpD / 84 Spe
Modest Nature
- Make It Rain
- Shadow Ball
- Power Gem
- Trick
""",
    6: """Hatterene @ Covert Cloak
Ability: Magic Bounce
Tera Type: Fire
EVs: 252 HP / 4 Def / 252 SpA
Quiet Nature
IVs: 0 Atk / 0 Spe
- Expanding Force
- Dazzling Gleam
- Tera Blast
- Trick Room
""",
    7: """Indeedee-F @ Psychic Seed
Ability: Psychic Surge
Tera Type: Water
EVs: 236 HP / 252 Def / 20 SpD
Relaxed Nature
IVs: 0 Atk / 0 Spe
- Psychic
- Follow Me
- Helping Hand
- Trick Room
""",
    8: """Torkoal @ Choice Specs
Ability: Drought
Level: 50
Tera Type: Fire
EVs: 252 HP / 252 SpA / 4 SpD
Quiet Nature
IVs: 0 Atk / 0 Spe
- Eruption
- Heat Wave
- Earth Power
- Weather Ball
""",
    9: """Ursaluna @ Flame Orb
Ability: Guts
Level: 50
Tera Type: Ghost
EVs: 252 HP / 252 Atk / 4 Def
Brave Nature
IVs: 0 Spe
- Facade
- Earthquake
- Headlong Rush
- Protect
""",
    10: """Gallade @ Clear Amulet
Ability: Sharpness
Level: 50
Tera Type: Grass
EVs: 220 HP / 252 Atk / 36 SpD
Brave Nature
IVs: 0 Spe
- Sacred Sword
- Psycho Cut
- Wide Guard
- Trick Room
""",
    11: """Maushold @ Wide Lens
Ability: Technician
Level: 50
Tera Type: Poison
EVs: 4 HP / 252 Atk / 252 Spe
Jolly Nature
- Population Bomb
- Follow Me
- Taunt
- Protect
""",
    12: """Dragonite @ Loaded Dice
Ability: Multiscale
Level: 50
Tera Type: Fairy
EVs: 36 HP / 212 Atk / 4 Def / 4 SpD / 252 Spe
Adamant Nature
- Scale Shot
- Tailwind
- Haze
- Protect
""",
    13: """Gholdengo @ Life Orb
Ability: Good as Gold
Level: 50
Tera Type: Water
EVs: 116 HP / 4 Def / 132 SpA / 4 SpD / 252 Spe
Timid Nature
- Make It Rain
- Shadow Ball
- Nasty Plot
- Protect
""",
    14: """Ursaluna-Bloodmoon @ Assault Vest
Ability: Mind's Eye
Level: 50
Tera Type: Water
EVs: 148 HP / 12 Def / 196 SpA / 100 SpD / 52 Spe
Modest Nature
- Blood Moon
- Earth Power
- Hyper Voice
- Vacuum Wave
""",
    15: """Annihilape @ Sitrus Berry
Ability: Defiant
Level: 50
Tera Type: Fire
EVs: 180 HP / 68 Atk / 4 Def / 4 SpD / 252 Spe
Jolly Nature
- Drain Punch
- Rage Fist
- Bulk Up
- Protect
""",
    16: """Maushold-Four @ Focus Sash
Ability: Friend Guard
Level: 50
Tera Type: Ghost
EVs: 236 HP / 60 Def / 212 Spe
Jolly Nature
- Super Fang
- Beat Up
- Follow Me
- Protect
""",
    17: """Rillaboom @ Miracle Seed
Ability: Grassy Surge
Level: 50
Tera Type: Fire
EVs: 188 HP / 196 Atk / 4 Def / 20 SpD / 100 Spe
Adamant Nature
- Wood Hammer
- Grassy Glide
- High Horsepower
- Fake Out
""",
    18: """Whimsicott @ Focus Sash
Ability: Prankster
Level: 50
Tera Type: Ghost
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
IVs: 0 Atk
- Tailwind
- Moonblast
- Encore
- Fake Tears
""",
    19: """Gholdengo @ Choice Specs
Ability: Good as Gold
Level: 50
Tera Type: Steel
EVs: 4 HP / 4 Def / 244 SpA / 4 SpD / 252 Spe
Modest Nature
IVs: 0 Atk
- Make It Rain
- Shadow Ball
- Thunderbolt
- Power Gem
""",
    20: """Pelipper @ Choice Scarf
Ability: Drizzle
Level: 50
Tera Type: Ghost
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
IVs: 0 Atk
- Icy Wind
- Weather Ball
- Muddy Water
- Hurricane
""",
    21: """Rillaboom @ Miracle Seed
Ability: Grassy Surge
Level: 50
Tera Type: Grass
EVs: 244 HP / 196 Atk / 4 Def / 12 SpD / 52 Spe
Adamant Nature
- Fake Out
- Grassy Glide
- Wood Hammer
- Taunt
""",
    22: """Dragonite @ Loaded Dice
Ability: Multiscale
Level: 50
Tera Type: Steel
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Scale Shot
- Low Kick
- Protect
- Tailwind
""",
    23: """Archaludon @ Power Herb
Ability: Sturdy
Level: 50
Tera Type: Electric
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
IVs: 0 Atk
- Electro Shot
- Draco Meteor
- Flash Cannon
- Protect
""",
    24: """Gholdengo @ Grassy Seed
Ability: Good as Gold
Level: 50
Tera Type: Water
EVs: 236 HP / 4 Def / 4 SpA / 28 SpD / 236 Spe
Modest Nature
IVs: 0 Atk
- Nasty Plot
- Make It Rain
- Shadow Ball
- Protect
""",
    25: """Sneasler @ White Herb
Ability: Unburden
Level: 50
Tera Type: Ghost
EVs: 164 HP / 92 Atk / 108 Def / 4 SpD / 140 Spe
Adamant Nature
- Close Combat
- Dire Claw
- Coaching
- Protect
""",
    26: """Rillaboom @ Assault Vest
Ability: Grassy Surge
Level: 50
Tera Type: Fire
EVs: 108 HP / 140 Atk / 4 Def / 4 SpD / 252 Spe
Jolly Nature
- Fake Out
- Grassy Glide
- Drum Beating
- High Horsepower
""",
    27: """Dragonite @ Loaded Dice
Ability: Multiscale
Level: 50
Tera Type: Ground
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
IVs: 23 SpA
- Scale Shot
- Stomping Tantrum
- Tailwind
- Protect
""",
    28: """Ninetales-Alola @ Focus Sash
Ability: Snow Warning
Level: 50
Tera Type: Ghost
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
IVs: 0 Atk
- Blizzard
- Icy Wind
- Encore
- Aurora Veil
""",
    29: """Arcanine-Hisui @ Clear Amulet
Ability: Intimidate
Level: 50
Tera Type: Water
EVs: 4 HP / 252 Atk / 252 Spe
Jolly Nature
- Rock Slide
- Flare Blitz
- Extreme Speed
- Protect
""",
    30: """Garchomp @ Life Orb
Ability: Rough Skin
Level: 50
Tera Type: Fairy
EVs: 4 HP / 252 Atk / 252 Spe
Jolly Nature
- Dragon Claw
- Stomping Tantrum
- Earthquake
- Protect
""",
    31: """Jumpluff @ Covert Cloak
Ability: Chlorophyll
Level: 50
Tera Type: Poison
EVs: 228 HP / 28 Def / 252 Spe
Timid Nature
- Leaf Storm
- Sleep Powder
- Encore
- Tailwind
""",
    32: """Torkoal @ Eject Pack
Ability: Drought
Level: 50
Tera Type: Flying
EVs: 212 HP / 196 SpA / 100 Spe
Modest Nature
- Overheat
- Burning Jealousy
- Helping Hand
- Protect
""",
    33: """Dragonite @ Loaded Dice
Ability: Multiscale
Level: 50
Tera Type: Fairy
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Scale Shot
- Extreme Speed
- Ice Spinner
- Protect
""",
    34: """Kingambit @ Assault Vest
Ability: Defiant
Level: 50
Tera Type: Flying
EVs: 140 HP / 172 Atk / 196 Spe
Adamant Nature
- Iron Head
- Sucker Punch
- Assurance
- Low Kick
""",
    35: """Typhlosion-Hisui @ Choice Specs
Ability: Blaze
Tera Type: Fire
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Eruption
- Shadow Ball
- Heat Wave
- Overheat
""",
}

class BattleState:
    '''
    BattleState is a wrapper over a 2x5x22 tensor storing the state of the pokemons
    of the two players.

    2 channels - one for each player

    5 rows - 1st row for field conditions on respective players side of the field
                Next 4 rows for the state of each pokemon selected to play

    22 cols for pokemons
        Col 0 - pokemonID
        Col 1 - primary typing
        Col 2 - secondary typing
        Col 3 - tera burnt or not
        Col 4 - item held / consumed or knocked off
        Col 5 - non volatile status condition
        Col [6-8] - one hot encoding of taunt, encore, confusion status respectively (1 if active, 0 otherwise)
        Col 9 - current HP stat
        Col [10-14] - base stats (excluding HP)
        Col [15-21] - stat stages (all 6 base stars excluding HP + accuracy and evasion)

    cols for field effects (first 5 are global, last 2 are local, value of 0 means inactive)
        Col 0 - trick room turns remaining
        Col 1 - grassy terrain turns remaining
        Col 2 - psy terrain turns remaining
        Col 3 - sun turns remaining
        Col 4 - rain turns remaining
        Col 5 - tailwind turns remaining
        Col 6 - aurora veil turns remaining
        Col [7-21] - padding using 0 (future space to expand ??)

    Additional considerations for the future
        Store history of moves for moves that are not independent (Protect, Stomping Tantrum)
        Store Rage Fist stacks for annihilape ??
    '''

    def __init__(self):
        # defaults to wrapping over a bunch of zeros
        return torch.zeros(2,5,22)
