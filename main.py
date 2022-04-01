from bw4t.BW4TWorld import BW4TWorld
from bw4t.statistics import Statistics

# from agents1.BW4TBaselineAgent import BaseLineAgent
# from agents1.BW4TStrongAgent import StrongAgent
# from agents1.BW4TLyingAgent import LyingAgent
# from agents1.BW4TColorblindAgent import ColorBlindAgent
# from agents1.BetterAgent import BetterAgent
# from agents1.BW4THuman import Human
# # from agents1.LyingAgent import LyingAgent
# from agents1.Agent import Agent

from agents1.Agents import NormalAgent, StrongAgent, LyingAgent, ColorblindAgent, LazyAgent



"""
This runs a single session. You have to log in on localhost:3000 and 
press the start button in god mode to start the session.
"""

if __name__ == "__main__":
    agents = [
        # {'name': 'lyingagent', 'botclass': LyingAgent, 'settings': {}},
        # {'name': 'betteragent2', 'botclass': BetterAgent, 'settings': {}},
        # {'name': 'betteragent3', 'botclass': BetterAgent, 'settings': {}},
        # {'name':'agent1', 'botclass':BaseLineAgent, 'settings':{'slowdown':10}},
        # {'name':'agent2', 'botclass':BaseLineAgent, 'settings':{'slowdown':10}},

        {'name': 'strongAgent', 'botclass': StrongAgent, 'settings': {}},
        # {'name': 'lazyAgent', 'botclass': LazyAgent, 'settings': {}},
        # {'name': 'colorblindagent', 'botclass': ColorblindAgent, 'settings': {}},
        # {'name': 'normalAgent', 'botclass': NormalAgent, 'settings': {}},
        # {'name': 'lyingAgent1', 'botclass': LyingAgent, 'settings': {}},
        #
        # {'name':'human', 'botclass':Human, 'settings':{}}
        ]

    print("Started world...")
    world=BW4TWorld(agents).run()
    print("DONE!")
    print(Statistics(world.getLogger().getFileName()))
