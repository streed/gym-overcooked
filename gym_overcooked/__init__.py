from gym.envs.registration import register

register(id='overcooked-v0',
         entry_point='gym_overcooked.envs:OvercookedEnv')
