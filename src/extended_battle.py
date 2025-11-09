from poke_env.battle.pokemon import Pokemon
from poke_env.battle.status import Status
from poke_env.battle.effect import Effect
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.move import SPECIAL_MOVES, Move
from poke_env.battle import DoubleBattle

from lookups import POKEMON

class ExtendedBattle(DoubleBattle):  
    def __init__(self, *args, **kwargs):  
        super().__init__(*args, **kwargs)  
        self.our_move_failed = {}  # Maps pokemon identifier to failed status 
        self.opponent_move_failed = {}  # Maps opponent pokemon identifier to failed status 
      
    def parse_message(self, split_message):  
        if len(split_message) > 1 and split_message[1] == "move":  
            failed = False  
            for suffix in ["[miss]", "[still]", "[notarget]"]:  
                if suffix in split_message:  
                    failed = True  
                    break  
              
            if len(split_message) > 2:  
                pokemon_id = split_message[2] 
                player_identifier = pokemon_id[:2]

                if pokemon_id[3] != " ":  
                    normalized_id = pokemon_id[:2] + pokemon_id[3:]  
                else:  
                    normalized_id = pokemon_id
                
                if player_identifier == self.player_role:  
                    for identity, mon in self.team.items():  
                        if identity == normalized_id:  
                            move_ids = [move.id for move in mon.moves.values()]
                            move_list_str = ",".join(move_ids)
                            self.our_move_failed[POKEMON.get(move_list_str, -1)] = failed
                            break
                elif player_identifier == self.opponent_role:
                    for identity, mon in self.opponent_team.items():  
                        if identity == normalized_id:  
                            move_ids = [move.id for move in mon.moves.values()]
                            move_list_str = ",".join(move_ids)
                            self.opponent_move_failed[POKEMON.get(move_list_str, -1)] = failed
          
        super().parse_message(split_message)  
      
    def did_last_move_fail(self, pokemon, opponent: bool = False) -> bool:  
        """Check if the given pokemon's last move failed"""  
        # Get pokemon identifier  
        for ident, mon in (self.opponent_team if opponent else self.team).items():  
            if mon is pokemon:  
                if opponent:  
                    return self.opponent_move_failed.get(ident, False)  
                else:  
                    return self.our_move_failed.get(ident, False) 
        return False