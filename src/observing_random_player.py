from poke_env.player import RandomPlayer 
from encoder import Encoder
import torch
import asyncio
from poke_env.battle import Battle
from extended_battle import ExtendedBattle
from poke_env.teambuilder import ConstantTeambuilder
class ObservingRandomPlayer(RandomPlayer):
    def __init__(self, *args, encoder: Encoder | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = encoder or Encoder()
        self.collected_states = []

    
    def _make_state_tensor(self):
        return torch.zeros((2, 5, 24), dtype=torch.float32)

    def choose_move(self, battle):
        state_tensor = self._make_state_tensor()
        if isinstance(battle, ExtendedBattle):
            self.encoder.encode_battle_state(battle, state_tensor)

        self.collected_states.append((battle.turn, state_tensor.clone()))
        return super().choose_move(battle)
    
    async def _create_battle(self, split_message):
        if split_message[1] == self._format and len(split_message) >= 2:  
            battle_tag = "-".join(split_message)[1:]  
              
            if battle_tag in self._battles:  
                return self._battles[battle_tag]  
            else:  
                from poke_env.data import GenData  
                gen = GenData.from_format(self._format).gen  
                  
                if self.format_is_doubles:  
                    battle = ExtendedBattle(  
                        battle_tag=battle_tag,  
                        username=self.username,  
                        logger=self.logger,  
                        gen=gen,  
                        save_replays=self._save_replays,  
                    )   
                else:  
                    battle = Battle(
                        battle_tag=battle_tag,
                        username=self.username,
                        logger=self.logger,
                        gen=gen,
                        save_replays=self._save_replays,
                    )
                    
                  
                if isinstance(self._team, ConstantTeambuilder):  
                    from poke_env.battle import Pokemon  
                    battle.teampreview_team = set([  
                        Pokemon(gen=gen, teambuilder=tb_mon)  
                        for tb_mon in self._team.team  
                    ])  
                  
                await self._battle_count_queue.put(None)  
                if battle_tag in self._battles:  
                    await self._battle_count_queue.get()  
                    return self._battles[battle_tag]  
                async with self._battle_start_condition:  
                    self._battle_semaphore.release()  
                    self._battle_start_condition.notify_all()  
                    self._battles[battle_tag] = battle  
                  
                if self._start_timer_on_battle_start:  
                    await self.ps_client.send_message("/timer on", battle.battle_tag)  
                  
                return battle  
        else:  
            from poke_env.exceptions import ShowdownException  
            self.logger.critical(  
                "Unmanaged battle initialisation message received: %s", split_message  
            )  
            raise ShowdownException()  