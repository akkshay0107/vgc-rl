from poke_env.battle.pokemon import Pokemon
from poke_env.battle.status import Status
from poke_env.battle.effect import Effect
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.move import SPECIAL_MOVES, Move
from poke_env.battle import DoubleBattle
import sys
from lookups import POKEMON

class ExtendedBattle(DoubleBattle):  
    def __init__(self, *args, **kwargs):  
        super().__init__(*args, **kwargs)  
        self.our_move_failed = {}  # Maps pokemon identifier to failed status 
        self.opponent_move_failed = {}  # Maps opponent pokemon identifier to failed status 
      
    def parse_message(self, split_message):  
        super().parse_message(split_message)
        if len(split_message) > 1 and (split_message[1] == "move" or split_message[1] == "cant"):  
            stomping_tantrum_fail = False
            for suffix in ["[miss]", "[still]", "[notarget]"]:  
                if suffix in split_message:  
                    stomping_tantrum_fail = True
                    break  
            if split_message[1] == "cant":  
                stomping_tantrum_fail = True
                print("parsing cant message", split_message, file=sys.stderr)
            
              
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
                            self.our_move_failed[normalized_id] = stomping_tantrum_fail
                            break
                elif player_identifier == self.opponent_role:
                    for identity, mon in self.opponent_team.items():  
                        if identity == normalized_id:  
                            self.opponent_move_failed[normalized_id] = stomping_tantrum_fail
        elif len(split_message) > 1 and split_message[1] == "-immune":  
            stomping_tantrum_fail = False
            if len(split_message) >= 3:  
                pokemon_id = split_message[2]  
                player_identifier = pokemon_id[:2]  
                
                if len(pokemon_id) > 3 and pokemon_id[3] != " ":  
                    normalized_id = pokemon_id[:2] + pokemon_id[3:]  
                else:  
                    normalized_id = pokemon_id  
                
                is_ability_immunity = any("[from] ability:" in part or "[from]ability:" in part   
                                        for part in split_message[3:])  
                if not is_ability_immunity:  
                    current_events = self._current_observation.events  
                    prev_message = current_events[-2] 
                    print("previous message:", prev_message, file=sys.stderr)
                    if len(prev_message) > 1 and prev_message[1] == "move":  
                        attacker_id = prev_message[2]  
                        player_identifier = attacker_id[:2]  

                        if len(attacker_id) > 3 and attacker_id[3] != " ":  
                            normalized_id = attacker_id[:2] + attacker_id[3:]  
                        else:  
                            normalized_id = attacker_id 
                            print(f"Recorded move failure for {normalized_id} due to -immune", file=sys.stderr) 
                        
                        if player_identifier == self.player_role:  
                            self.our_move_failed[normalized_id] = True  
                        elif player_identifier == self.opponent_role:  
                            self.opponent_move_failed[normalized_id] = True   
      
    def did_last_move_fail(self, pokemon, opponent: bool = False) -> bool:  
        """Check if the given pokemon's last move failed"""  
        for ident, mon in (self.opponent_team if opponent else self.team).items():  

            if mon is pokemon:  
                if opponent:  
                    return self.opponent_move_failed.get(ident, False)  
                else:  
                    return self.our_move_failed.get(ident, False) 
        return False