# Adding src to path
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from poke_env.battle.pokemon import Pokemon
from poke_env.teambuilder import TeambuilderPokemon

from encoder import Encoder

if __name__ == "__main__":
    # Example of pokemon string exceeding 512 tokens
    pokepaste_string = """
    Rillaboom @ Miracle Seed
    Ability: Grassy Surge
    Level: 50
    Tera Type: Grass
    EVs: 244 HP / 196 Atk / 4 Def / 12 SpD / 52 Spe
    Adamant Nature
    - Fake Out
    - Grassy Glide
    - Wood Hammer
    - Taunt
    """

    teambuilder_mon = TeambuilderPokemon.from_showdown(pokepaste_string)
    pokemon = Pokemon(gen=9, teambuilder=teambuilder_mon)

    pokemon_str = Encoder._get_pokemon_as_text(pokemon, cond=2)
    print(len(pokemon_str[0]), len(pokemon_str[1]))
    tokens = tuple(Encoder.tokenizer.encode(s, add_special_tokens=True) for s in pokemon_str)
    print(len(tokens[0]), len(tokens[1]))
    print("-" * 100)
    print(pokemon_str)
