from typing import final, List, Dict, Final
import enum, random
from bw4t.BW4TBrain import BW4TBrain
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.actions.door_actions import OpenDoorAction
from matrx.actions.object_actions import GrabObject, DropObject
from matrx.messages.message import Message
import random


class Phase(enum.Enum):
    PLAN_PATH_TO_CLOSED_DOOR = 1,
    FOLLOW_PATH_TO_CLOSED_DOOR = 2,
    OPEN_DOOR = 3


class LyingAgent(BW4TBrain):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
        self._teamMembers = []

    def initialize(self):
        super().initialize()
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,
                                    action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)

        self.collect_blocks = []



    def filter_observations(self, state):
        # find the Collect Blocks if not done yet
        if len(self.collect_blocks) is 0:
            self.collect_blocks = [item for item in state.values() if ('name' in item.keys() and item['name'] == 'Collect Block')]

        # Check for blocks nearby that can be seen
        self.blocksNearby = [item for item in state.values() if 'class_inheritance' in item and 'CollectableBlock' in item['class_inheritance']]
        for block in self.blocksNearby:
            if self.isTargetBlock(block):
                # pick up block
                print("Found one of the Target Blocks!!")
                pass
            self._sendMessage('Block in room_x:', block['visualization']['colour'], state[self.agent_id]['obj_id'])


        # print("blocks:", [item for item in state.values() if 'class_inheritance' in item and 'CollectableBlock' in item['class_inheritance']])
        # print()
        return state

    def decide_on_bw4t_action(self, state: State):
        agent_name = state[self.agent_id]['obj_id']
        # Add team members
        for member in state['World']['team_members']:
            if member != agent_name and member not in self._teamMembers:
                self._teamMembers.append(member)
                # Process messages from team members
        receivedMessages = self._processMessages(self._teamMembers)
        # Update trust beliefs for team members
        self._trustBlief(self._teamMembers, receivedMessages)

        while True:
            if Phase.PLAN_PATH_TO_CLOSED_DOOR == self._phase:
                self._navigator.reset_full()
                closedDoors = [door for door in state.values()
                               if 'class_inheritance' in door and 'Door' in door['class_inheritance'] and not door[
                        'is_open']]
                if len(closedDoors) == 0:
                    return None, {}
                # Randomly pick a closed door
                self._door = random.choice(closedDoors)
                doorLoc = self._door['location']
                # Location in front of door is south from door
                doorLoc = doorLoc[0], doorLoc[1] + 1
                # Send message of current action
                if random.random() > 0.8:   # Tell the truth
                    self._sendMessage('[TRUTH] Moving to door of ' + self._door['room_name'], agent_name)
                else: # Lie
                    if len(closedDoors) > 1:
                        closedDoors.remove(self._door)
                        self._sendMessage('[LIE] Moving to door of ' + random.choice(closedDoors)['room_name'], agent_name) # Choose random other closed door
                    else:
                        allDoors = [door for door in state.values() if 'class_inheritance' in door and 'Door' in door['class_inheritance']]
                        distractionDoor = random.choice(allDoors)         # Choose random other open door
                        while(len(state.values()) > 1 and distractionDoor is self._door):
                            distractionDoor = random.choice(allDoors)
                        self._sendMessage('[LIE] Moving to door of ' + distractionDoor['room_name'], agent_name)

                self._navigator.add_waypoints([doorLoc])
                self._phase = Phase.FOLLOW_PATH_TO_CLOSED_DOOR

            if Phase.FOLLOW_PATH_TO_CLOSED_DOOR == self._phase:
                self._state_tracker.update(state)
                # Follow path to door
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._phase = Phase.OPEN_DOOR

            if Phase.OPEN_DOOR == self._phase:
                self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
                # Open door
                return OpenDoorAction.__name__, {'object_id': self._door['obj_id']}


    def isTargetBlock(self, block):
        for i in range(len(self.collect_blocks)):
            if block['visualization']['shape'] == self.collect_blocks[i]['visualization']['shape']\
                    and block['visualization']['colour'] == self.collect_blocks[i]['visualization']['colour']\
                    and block['visualization']['size'] == self.collect_blocks[i]['visualization']['size']:
                return True
        return False

    def _sendMessage(self, mssg, sender):
        '''
        Enable sending messages in one line of code
        '''
        msg = Message(content=mssg, from_id=sender)
        if msg.content not in self.received_messages:
            self.send_message(msg)

    def _processMessages(self, teamMembers):
        '''
        Process incoming messages and create a dictionary with received messages from each team member.
        '''
        receivedMessages = {}
        for member in teamMembers:
            receivedMessages[member] = []
        for mssg in self.received_messages:
            for member in teamMembers:
                if mssg.from_id == member:
                    receivedMessages[member].append(mssg.content)
        return receivedMessages

    def _trustBlief(self, member, received):
        '''
        Baseline implementation of a trust belief. Creates a dictionary with trust belief scores for each team member, for example based on the received messages.
        '''
        # You can change the default value to your preference
        default = 0.5
        trustBeliefs = {}
        for member in received.keys():
            trustBeliefs[member] = default
        for member in received.keys():
            for message in received[member]:
                if 'Found' in message and 'colour' not in message:
                    trustBeliefs[member] -= 0.1
                    break
        return trustBeliefs
