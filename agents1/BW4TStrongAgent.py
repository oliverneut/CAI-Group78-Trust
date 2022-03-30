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
    SEARCH_ROOM = 4,
    DELIVER_BLOCK = 5,
    DROP_BLOCK = 6,
    REORDER_BLOCKS = 7,
    PICKDROP_BLOCKS = 8,
    DELIVER_2ND_BLOCK = 9


class StrongAgent(BaseLineAgent):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
        self._agentName = 0
        self._teamMembers = []
        self._goalBlocks = None
        self._visibleBlocks = None
        self._holdingBlocks = []
        self._destination = None
        self._retrievedGBlocks = [0, 0, 0]
        self._goalBlockIdx = []
        self._reorderActions = []

    def initialize(self):
        super().initialize()
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,
                                    action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)

    def filter_observations(self, state) -> State:
        if (self._goalBlocks == None):
            self._goalBlocks = [item for item in state.values() if
                                ('name' in item.keys() and item['name'] == 'Collect Block')]
            for g in self._goalBlocks:
                print(g)

        self._visibleBlocks = [item for item in state.values() if
                               'class_inheritance' in item and 'CollectableBlock' in item['class_inheritance']]

        return state

    def decide_on_bw4t_action(self, state: State):
        self._agentName = state[self.agent_id]['obj_id']
        # Add team members
        for member in state['World']['team_members']:
            if member != self._agentName and member not in self._teamMembers:
                self._teamMembers.append(member)
            # Process messages from team members
        receivedMessages = self._processMessages(self._teamMembers)
        # Update trust beliefs for team members
        self._trustBlief(self._teamMembers, receivedMessages)

        while True:
            if Phase.PLAN_PATH_TO_CLOSED_DOOR == self._phase:
                self.planPathToClosedDoor(state)
                self._phase = Phase.FOLLOW_PATH_TO_CLOSED_DOOR

            if Phase.FOLLOW_PATH_TO_CLOSED_DOOR == self._phase:
                res = self.followPathToClosedDoor(state)
                if res != -1:
                    return res
                else:
                    self._phase = Phase.OPEN_DOOR

            if Phase.OPEN_DOOR == self._phase:
                return self.openDoor(state)

            if Phase.SEARCH_ROOM == self._phase:
                res = self.searchRoom(state)
                if res != -1:
                    return res
                else:
                    self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

            if Phase.DELIVER_BLOCK == self._phase:
                res = self.deliverBlock(state)
                if res != -1:
                    return res
                else:
                    self._phase = Phase.DROP_BLOCK

            if Phase.DROP_BLOCK == self._phase:
                self._retrievedGBlocks[self._goalBlockIdx.pop()] = 1
                if sum(self._retrievedGBlocks) == 3:
                    self._phase = Phase.REORDER_BLOCKS
                else:
                    if len(self._holdingBlocks) == 2:
                        self._phase = Phase.DELIVER_2ND_BLOCK
                    else:
                        self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
                block = self._holdingBlocks.pop()
                self._sendMessage(
                    'Dropped goal block ' + self.visualize(block['visualization']) + ' at drop location ' + str(
                        self._destination), self._agentName)
                return DropObject.__name__, {'object_id': block['obj_id']}

            if Phase.DELIVER_2ND_BLOCK:
                print(len(self._holdingBlocks))
                print(self._retrievedGBlocks)
                print()

                self._navigator.reset_full()

                for gb in self._goalBlocks:
                    if self.visualize(gb['visualization']) == self.visualize(self._holdingBlocks[0]['visualization']):
                        self._navigator.add_waypoints([gb['location']])

                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                else:
                    self._phase = Phase.DROP_BLOCK

            if Phase.REORDER_BLOCKS == self._phase:
                self._navigator.reset_full()
                self._navigator.add_waypoints([self._goalBlocks[1]['location']])

                self._phase = Phase.PICKDROP_BLOCKS

            if Phase.PICKDROP_BLOCKS == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                else:
                    self._reorderActions.append((MoveSouth.__name__, {}))
                    self._state_tracker.update(state)
                    for gb in self._goalBlocks:
                        for vb in self._visibleBlocks:
                            if self.visualize(vb['visualization']) == self.visualize(gb['visualization']):
                                self._reorderActions.append((GrabObject.__name__, {'object_id': vb['obj_id']}))
                                self._reorderActions.append((DropObject.__name__, {'object_id': vb['obj_id']}))
                                self._reorderActions.append((MoveNorth.__name__, {}))
                                break
                    self._reorderActions.pop()

                    if len(self._reorderActions) != 0:
                        return self._reorderActions.pop(0)

    def deliverBlock(self, state):
        self._state_tracker.update(state)
        # Follow path to door
        action = self._navigator.get_move_action(self._state_tracker)
        if action != None:
            return action, {}
        else:
            return -1

    def searchRoom(self, state: State):
        self._sendMessage('Searching through ' + self._door['room_name'], self._agentName)
        # Follow path to door
        foundBlock, vBlock, delivery_loc = self.takeBlock()
        if foundBlock:
            self._sendMessage(
                'Picking up goal block ' + self.visualize(vBlock['visualization']) + ' at ' + str(vBlock['location']),
                self._agentName)
            self._destination = delivery_loc
            self._navigator.reset_full()
            self._holdingBlocks.append(vBlock)

            if len(self._holdingBlocks) > 1 or sum(self._retrievedGBlocks) == 2: # If this is the second block, or last block deliver blocks
                self._phase = Phase.DELIVER_BLOCK
                self._navigator.add_waypoints([self._destination])
            else:
                self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

            return GrabObject.__name__, {'object_id': self._holdingBlocks[-1]['obj_id']}

        self._state_tracker.update(state)
        action = self._navigator.get_move_action(self._state_tracker)
        if action != None:
            return action, {}
        else:
            return -1

    def openDoor(self, state: State):
        self._navigator.reset_full()
        # search room phase
        self._phase = Phase.SEARCH_ROOM
        # make new waypoint to lower left and right corner of room
        coordinate = self._door['location']
        coordinate_1 = coordinate[0] - 2, coordinate[1] - 1
        coordinate_2 = coordinate[0] + 1, coordinate[1] - 1
        self._navigator.add_waypoints([coordinate_1, coordinate_2])

        # Open door
        self._sendMessage('Opening door of ' + self._door['room_name'], self._agentName)
        return OpenDoorAction.__name__, {'object_id': self._door['obj_id']}

    def followPathToClosedDoor(self, state: State):
        self._state_tracker.update(state)
        # Follow path to door
        action = self._navigator.get_move_action(self._state_tracker)
        if action != None:
            return action, {}
        else:
            return -1

    def planPathToClosedDoor(self, state: State):
        self._navigator.reset_full()
        closedDoors = [door for door in state.values()
                       if 'class_inheritance' in door and 'Door' in door['class_inheritance'] and not door['is_open']]
        if len(closedDoors) == 0:
            return None, {}
        # Randomly pick a closed door
        self._door = random.choice(closedDoors)
        doorLoc = self._door['location']
        # Location in front of door is south from door
        doorLoc = doorLoc[0], doorLoc[1] + 1
        # Send message of current action
        self._sendMessage('Moving to ' + self._door['room_name'], self._agentName)
        self._navigator.add_waypoints([doorLoc])

    # returns true if it found a goalBlock and the specified obj_id
    def takeBlock(self):
        for i in range(len(self._goalBlocks)):
            gBlock = self._goalBlocks[i]
            if self._retrievedGBlocks[i] != 1:
                gb = gBlock['visualization']
                for vBlock in self._visibleBlocks:
                    vb = vBlock['visualization']
                    if (vb['shape'] == gb['shape']
                            and vb['colour'] == gb['colour']
                            and vb['size'] == gb['size']):
                        msg = 'Found goal block ' + self.visualize(vb) + ' at location ' + str(vBlock['location'])
                        self._sendMessage(msg, self._agentName)
                        self._goalBlockIdx.append(i)
                        return True, vBlock, gBlock['location']
        return False, None, None

    def visualize(self, block):
        res = '{"size": ' + str(block['size']) + ', "shape": ' + str(block['shape']) + ', "colour": "' + str(
            block['colour']) + '"}'
        return res

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
