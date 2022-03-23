from typing import final, List, Dict, Final
import enum, random
from bw4t.BW4TBrain import BW4TBrain
from agents1.BW4TBaselineAgent import BaseLineAgent
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.actions.door_actions import OpenDoorAction
from matrx.actions.move_actions import MoveNorth, MoveEast, MoveSouth, MoveWest
from matrx.actions.object_actions import GrabObject, DropObject
from matrx.messages.message import Message


class Phase(enum.Enum):
    PLAN_PATH_TO_CLOSED_DOOR = 1,
    FOLLOW_PATH_TO_CLOSED_DOOR = 2,
    OPEN_DOOR = 3,
    SEARCH_ROOM = 4
    DELIVER_BLOCK = 5


class Agent(BaseLineAgent):

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
        self.grabBlock = None

    def filter_observations(self, state):
        # find the Collect Blocks if not done yet
        if len(self.collect_blocks) is 0:
            self.collect_blocks = [item for item in state.values() if
                                   ('name' in item.keys() and item['name'] == 'Collect Block')]

        # Check for blocks nearby that can be seen
        self.blocksNearby = [item for item in state.values() if
                             'class_inheritance' in item and 'CollectableBlock' in item['class_inheritance']]
        for block in self.blocksNearby:
            isTargetBlock = self.isTargetBlock(block)
            if isTargetBlock >= 0:
                # pick up block
                self._sendMessage('Found goal block ' + self.visualizeBlock(block)
                                  + " at location " + str(block['location'])
                                  , state[self.agent_id]['obj_id'])

                self.grabBlock = block['obj_id']
                break
            else:
                self.grabBlock = None
        # self._sendMessage('Block in room_x: (' + str(block['visualization']['size']) + ', ' + str(block['visualization']['shape']) + ', ' + str(block['visualization']['colour']) + ')', state[self.agent_id]['obj_id'])
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
                self._sendMessage('Moving to ' + self._door['room_name'], agent_name)
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
                self._navigator.reset_full()
                # search room phase
                self._phase = Phase.SEARCH_ROOM
                # make new waypoint to lower left corner of room
                coordinate = self._door['location']
                coordinate_1 = coordinate[0] - 2, coordinate[1] - 1
                coordinate_2 = coordinate[0] + 1, coordinate[1] - 1
                self._navigator.add_waypoints([coordinate_1, coordinate_2])
                # make new waypoint to lower right corner of room
                # self._navigator.add_waypoint([coordinate[0]+1, coordinate[1]-2])
                # Open door
                self._sendMessage('Opening door of ' + self._door['room_name'], agent_name)
                return OpenDoorAction.__name__, {'object_id': self._door['obj_id']}

            if Phase.SEARCH_ROOM == self._phase:
                self._sendMessage('Searching through ' + self._door['room_name'], agent_name)
                self._state_tracker.update(state)
                # Follow path to door
                action = self._navigator.get_move_action(self._state_tracker)
                if self.grabBlock is not None:
                    self._phase = Phase.DELIVER_BLOCK
                    goalblock = [block for block in self.state.values() if 'obj_id' in block and block['obj_id'] == self.grabBlock][0]
                    self._sendMessage('Picking up goal block ' + self.visualizeBlock(goalblock) + ' at ' + str(goalblock['location']), agent_name)
                    return GrabObject.__name__, {'object_id': self.grabBlock}
                if action != None:
                    return action, {}
                self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

            if Phase.DELIVER_BLOCK == self._phase:
                pass

    def isTargetBlock(self, block):
        for i in range(len(self.collect_blocks)):
            if block['visualization']['shape'] == self.collect_blocks[i]['visualization']['shape'] \
                    and block['visualization']['colour'] == self.collect_blocks[i]['visualization']['colour'] \
                    and block['visualization']['size'] == self.collect_blocks[i]['visualization']['size']:
                return i
        return -1

    def visualizeBlock(self, block):
        return '{"size": ' + str(block['visualization']['size']) + ', "shape": ' + str(block['visualization']['size']) \
               + ', "colour": "' + str(block['visualization']['colour']) + '"'

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
