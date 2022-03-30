from bw4t.BW4TWorld import BW4TWorld
from bw4t.statistics import Statistics
from agents1.BW4TBaselineAgent import BaseLineAgent
from agents1.BW4TStrongAgent import StrongAgent
from agents1.BW4TLyingAgent import LyingAgent
from agents1.BW4THuman import Human
# from agents1.LyingAgent import LyingAgent
from agents1.Agent import Agent


"""
This runs a single session. You have to log in on localhost:3000 and 
press the start button in god mode to start the session.
"""

if __name__ == "__main__":
    agents = [
        # {'name': 'strongagent', 'botclass': StrongAgent, 'settings': {}},
        {'name': 'lyingagent', 'botclass': LyingAgent, 'settings': {}},
        # {'name':'agent1', 'botclass':BaseLineAgent, 'settings':{'slowdown':10}},
        # {'name':'agent2', 'botclass':BaseLineAgent, 'settings':{'slowdown':10}},
        # {'name': 'lyingAgent1', 'botclass': LyingAgent, 'settings': {'slowdown': 3}},
        # {'name': 'ourAgent', 'botclass': Agent, 'settings': {}},
        #
        # {'name':'human', 'botclass':Human, 'settings':{}}
        ]

    print("Started world...")
    world=BW4TWorld(agents).run()
    print("DONE!")
    print(Statistics(world.getLogger().getFileName()))
