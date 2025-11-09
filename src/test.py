import extended_battle
from observing_random_player import ObservingRandomPlayer
import encoder
import asyncio
from poke_env import AccountConfiguration, Player
from poke_env.teambuilder import ConstantTeambuilder
from poke_env.player import SimpleHeuristicsPlayer, RandomPlayer
from poke_env.battle import DoubleBattle
from encoder import Encoder
import torch
#random test team
team1 = ConstantTeambuilder("""
Basculegion @ Focus Sash
Ability: Adaptability
Level: 50
Tera Type: Ghost
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Liquidation
- Last Respects
- Aqua Jet
- Protect

Maushold-Four @ Rocky Helmet
Ability: Friend Guard
Level: 50
Tera Type: Poison
EVs: 252 HP / 4 Atk / 180 Def / 20 SpD / 52 Spe
Jolly Nature
- Super Fang
- Feint
- Follow Me
- Protect

Dragonite @ Loaded Dice
Ability: Multiscale
Level: 50
Tera Type: Fairy
EVs: 44 HP / 204 Atk / 4 Def / 4 SpD / 252 Spe
Jolly Nature
- Scale Shot
- Haze
- Tailwind
- Protect

Incineroar @ Safety Goggles
Ability: Intimidate
Level: 50
Tera Type: Grass
EVs: 196 HP / 4 Atk / 4 Def / 68 SpD / 236 Spe
Jolly Nature
- Flare Blitz
- Knock Off
- Fake Out
- Parting Shot

Ursaluna-Bloodmoon @ Assault Vest
Ability: Mind's Eye
Level: 50
Tera Type: Fire
EVs: 156 HP / 4 Def / 116 SpA / 100 SpD / 132 Spe
Modest Nature
IVs: 0 Atk
- Vacuum Wave
- Earth Power
- Hyper Voice
- Blood Moon

Gholdengo @ Choice Specs
Ability: Good as Gold
Level: 50
Tera Type: Steel
EVs: 228 HP / 84 Def / 52 SpA / 60 SpD / 84 Spe
Modest Nature
- Make It Rain
- Shadow Ball
- Power Gem
- Trick
""")
team2 = ConstantTeambuilder("""
Basculegion @ Focus Sash
Ability: Adaptability
Level: 50
Tera Type: Ghost
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Liquidation
- Last Respects
- Aqua Jet
- Protect

Maushold-Four @ Rocky Helmet
Ability: Friend Guard
Level: 50
Tera Type: Poison
EVs: 252 HP / 4 Atk / 180 Def / 20 SpD / 52 Spe
Jolly Nature
- Super Fang
- Feint
- Follow Me
- Protect

Dragonite @ Loaded Dice
Ability: Multiscale
Level: 50
Tera Type: Fairy
EVs: 44 HP / 204 Atk / 4 Def / 4 SpD / 252 Spe
Jolly Nature
- Scale Shot
- Haze
- Tailwind
- Protect

Incineroar @ Safety Goggles
Ability: Intimidate
Level: 50
Tera Type: Grass
EVs: 196 HP / 4 Atk / 4 Def / 68 SpD / 236 Spe
Jolly Nature
- Flare Blitz
- Knock Off
- Fake Out
- Parting Shot

Ursaluna-Bloodmoon @ Assault Vest
Ability: Mind's Eye
Level: 50
Tera Type: Fire
EVs: 156 HP / 4 Def / 116 SpA / 100 SpD / 132 Spe
Modest Nature
IVs: 0 Atk
- Vacuum Wave
- Earth Power
- Hyper Voice
- Blood Moon

Gholdengo @ Choice Specs
Ability: Good as Gold
Level: 50
Tera Type: Steel
EVs: 228 HP / 84 Def / 52 SpA / 60 SpD / 84 Spe
Modest Nature
- Make It Rain
- Shadow Ball
- Power Gem
- Trick
""")

player1_config = AccountConfiguration("udhuePljdhuyer1", None)  #for some reason showdown sometimes doesnt work for me without this
player2_config = AccountConfiguration("edhwiduwihdw", None) #for some reason showdown sometimes doesnt work for me without this
async def main():
    player_1 = ObservingRandomPlayer(
        account_configuration= player1_config, battle_format="gen9vgc2025regh", team=team1, max_concurrent_battles=10
    )
    player_2 = ObservingRandomPlayer(
        account_configuration= player2_config, battle_format="gen9vgc2025regh", team=team2, max_concurrent_battles=10
    )
    await player_1.battle_against(player_2, n_battles=1)

asyncio.run(main())
