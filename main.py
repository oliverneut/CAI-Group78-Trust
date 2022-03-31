from bw4t.BW4TWorld import BW4TWorld
from bw4t.statistics import Statistics
from agents1.BW4TBaselineAgent import BaseLineAgent
<<<<<<< Updated upstream
=======
from agents1.BW4TStrongAgent import StrongAgent
from agents1.BW4TLyingAgent import LyingAgent
from agents1.BetterAgent import BetterAgent
from agents1.ColorblindAgent import ColorblindAgent
>>>>>>> Stashed changes
from agents1.BW4THuman import Human


"""
This runs a single session. You have to log in on localhost:3000 and 
press the start button in god mode to start the session.
"""

if __name__ == "__main__":
    agents = [
<<<<<<< Updated upstream
        {'name':'agent1', 'botclass':BaseLineAgent, 'settings':{'slowdown':10}},
        {'name':'agent2', 'botclass':BaseLineAgent, 'settings':{}},
        {'name':'human', 'botclass':Human, 'settings':{}}
=======
        # {'name': 'strongagent', 'botclass': StrongAgent, 'settings': {}},
        # {'name': 'lyingagent', 'botclass': LyingAgent, 'settings': {}},
        # {'name': 'colorblindagent', 'botclass': ColorBlindAgent, 'settings': {}},
        {'name': 'betteragent', 'botclass': BetterAgent, 'settings': {}},
        {'name': 'colorblindagent', 'botclass': ColorblindAgent, 'settings': {}},

        # {'name':'agent1', 'botclass':BaseLineAgent, 'settings':{'slowdown':10}},
        # {'name':'agent2', 'botclass':BaseLineAgent, 'settings':{'slowdown':10}},
        # {'name': 'lyingAgent1', 'botclass': LyingAgent, 'settings': {}},
        # {'name': 'ourAgent', 'botclass': Agent, 'settings': {}},
        #
        # {'name':'human', 'botclass':Human, 'settings':{}}
>>>>>>> Stashed changes
        ]

    print("Started world...")
    world=BW4TWorld(agents).run()
    print("DONE!")
    print(Statistics(world.getLogger().getFileName()))
