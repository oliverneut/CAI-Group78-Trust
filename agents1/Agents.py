from typing import final, List, Dict, Final
import enum, random
from bw4t.BW4TBrain import BW4TBrain
from agents1.BW4TBaselineAgent import BaseLineAgent
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.actions.move_actions import MoveNorth, MoveEast, MoveSouth, MoveWest
from matrx.actions.door_actions import OpenDoorAction
from matrx.actions.object_actions import GrabObject, DropObject
from matrx.messages.message import Message


class Phase(enum.Enum):
    PLAN_PATH_TO_CLOSED_DOOR = 1,
    FOLLOW_PATH_TO_CLOSED_DOOR = 2,
    OPEN_DOOR = 3,
    SEARCH_ROOM = 4,
    DROP_GOALBLOCK = 5,
    REORDER_BLOCKS = 6,
    CHECK_DROP = 7

class BaseAgent(BaseLineAgent):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
        self._teamMembers = []
        self._agentName = None
        self._searching = False
        self._goalBlocks = None
        self._visibleBlocks = None
        self._droppedBlocks = [0, 0, 0]
        self._goalBlockFound = None
        self._holdingBlock = None
        self._dropping = False
        self._reordering = False
        self._blockOrderIdx = 0
        self._doors = None
        self._doorsPrevious = None
        self._amountOfMessages = {}
        self._observations = {}
        self._messages = {'found': [], 'picked up': [], 'dropped': [], 'self dropped': []}
        self._trustBeliefs = None

    def initialize(self):
        super().initialize()
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,
                                    action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)

    def filter_observations(self, state: State) -> State:
        if self._agentName == None:
            self._agentName = state[self.agent_id]['obj_id']

        if self._goalBlocks == None:
            self._goalBlocks = [item for item in state.values() if
                                ('name' in item.keys() and item['name'] == 'Collect Block')]
        if self._doors == None:
            self._doors = [door for door in state.values() if
                           'class_inheritance' in door and 'Door' in door['class_inheritance']]

        self._visibleBlocks = [item for item in state.values() if
                               'class_inheritance' in item and 'CollectableBlock' in item['class_inheritance']]

        return state

    def decide_on_bw4t_action(self, state: State):
        # Add team members
        for member in state['World']['team_members']:
            if member != self._agentName and member not in self._teamMembers:
                self._teamMembers.append(member)
        # Process messages from team members
        receivedMessages = self._processMessages(self._teamMembers)

        if self._doors != None:
            self._doorsPrevious = self._doors.copy()

        self._doors = {}
        for door in state.values():
            if 'class_inheritance' in door and 'Door' in door['class_inheritance']:
                self._doors[door['room_name']] = door

        if self._doorsPrevious == None:
            self._doorsPrevious = self._doors.copy()

        # Update trust beliefs for team members
        self._trustBelief(self._teamMembers, receivedMessages)

        while True:
            if Phase.PLAN_PATH_TO_CLOSED_DOOR == self._phase:
                # make a path to a door and announce that you are going there
                self._planPathToCLosedDoor(state)

                # non lazy agents go where they planned to go
                if self._type is not 'lazy' or random.random() > 0.5:
                    self._phase = Phase.FOLLOW_PATH_TO_CLOSED_DOOR

                # lazy agents don't do what they say they would 50% of the time
                else:
                    self._navigator.reset_full()
                    self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR


            if Phase.FOLLOW_PATH_TO_CLOSED_DOOR == self._phase:
                action = self._followPathToClosedDoor(state)
                if action != None:
                    return action, {}
                else:
                    self._phase = Phase.OPEN_DOOR

            if Phase.OPEN_DOOR == self._phase:
                # if the door is not open yet, agent says that he's going to open it
                if (not self._door['is_open']):
                    self._sendMessage('Opening door of ' + self._getRoom(self._door['room_name']), self._agentName)

                # non-lazy agents open the door after they say they will
                if self._type is not 'lazy' or random.random() > 0.5:
                    self._phase = Phase.SEARCH_ROOM
                    return OpenDoorAction.__name__, {'object_id': self._door['obj_id']}

                # lazy agents don't do what they say they would 50% of the time
                else:
                    self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

            if Phase.SEARCH_ROOM == self._phase:
                # add waypoints for searching the room and announce that you're going to searce the room
                res, params = self._searchRoom(state)

                # search the room
                if res != None:
                    return res, params

                # once done, stop searching
                self._searching = False
                self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

            if Phase.DROP_GOALBLOCK == self._phase:
                return self._dropGoalBlock(state)

            if Phase.CHECK_DROP == self._phase:
                self._navigator.add_waypoints([self._goalBlocks[1]['location']])
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)

                if action != None:
                    return action, {}
                else:
                    droppedByMe = self._messages['self dropped'].copy()
                    for block in state.values():
                        if 'name' in block and 'Block in' in block['name']:
                            bVis = self._accurateVisualization(block['visualization']) + ' at ' + str(block['location'])
                            if bVis in droppedByMe:
                                droppedByMe.remove(bVis)
                            else:
                                for supposedlyDropped in self._messages['dropped']:
                                    if supposedlyDropped['block'] == bVis:
                                        self._observations[supposedlyDropped['id']]['truths'] += 1
                                        self._messages['dropped'].remove(supposedlyDropped)
                                        self._sendIndirectObservation(supposedlyDropped['id'], True)
                                        self._sendReputations()

                    for message in self._messages['dropped']:
                        self._observations[message['id']]['lies'] += 1
                        self._sendIndirectObservation(message['id'], False)
                        self._sendReputations()

                    self._messages['dropped'] = []
                    self._phase = Phase.DROP_GOALBLOCK

    def _dropGoalBlock(self, state: State):
        if not self._dropping:
            self._navigator.reset_full()
            self._navigator.add_waypoints([self._goalBlocks[self._goalBlockFound]['location']])
        self._state_tracker.update(state)
        action = self._navigator.get_move_action(self._state_tracker)

        if action != None:
            return action, {}
        else:
            self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
            self._droppedBlocks[self._goalBlockFound] = 1
            self._dropping = False
            msg = 'Dropped goal block ' + self._visualize(self._holdingBlock['visualization']) + ' at drop location ' + str(
                self._goalBlocks[self._goalBlockFound]['location'])
            self._sendMessage(msg, self._agentName)
            self._messages['self dropped'].append(self._accurateVisualization(self._holdingBlock['visualization']) + ' at ' + str(
                self._goalBlocks[self._goalBlockFound]['location']))
            return DropObject.__name__, {'object_id': self._holdingBlock['obj_id']}

    def _searchRoom(self, state: State):
        self._sendMessage('Searching through ' + self._getRoom(self._door['room_name']), self._agentName)
        if not self._searching:
            # the lazy agent does not search the room 50% of the time
            if self._type is not 'lazy' or random.random() > 0.5:
                self._createPath(state)
            self._searching = True
        self._state_tracker.update(state)
        action = self._navigator.get_move_action(self._state_tracker), {}

        blockDetected, vBlock = self._detectBlocks(state)
        if blockDetected:
            msg = 'Picking up goal block ' + self._visualize(vBlock['visualization']) + ' at location ' + str(
                vBlock['location'])
            self._sendMessage(msg, self._agentName)

            # non-lazy agents pick up a block when they find it
            if self._type is not 'lazy' or random.random() > 0.5:
                self._holdingBlock = vBlock
                action = GrabObject.__name__, {'object_id': vBlock['obj_id']}
                self._phase = Phase.CHECK_DROP
                self._searching = False

        return action

    def _detectBlocks(self, state: State):
        for i in range(len(self._goalBlocks)):
            gb = self._goalBlocks[i]['visualization']
            if i == self._blockOrderIdx:
                if not self._droppedBlocks[i]:
                    for v in self._visibleBlocks:
                        vb = v['visualization']
                        if self._accurateVisualization(vb) == self._accurateVisualization(gb):
                            print(vb)
                            msg = 'Found goal block ' + self._visualize(vb) + ' at location ' + str(v['location'])
                            self._sendMessage(msg, self._agentName)
                            self._goalBlockFound = i
                            self._blockOrderIdx += 1
                            return True, v
        return False, None

    def _visualize(self, block):
        print(block)
        """
		This method makes the visualization that the agent can use for the self._sendMessage() method
		"""
        # Colorblind agents pick a random color as the color
        if self._type == 'colorblind':
            return '{"size": ' + str(block['size']) + ', "shape": ' + str(block['shape']) + ', "colour": "' + str(
                random.choice(self._goalBlocks)['visualization']['colour']) + '"}'

            # Liar visualizes a random different block 80% of the time
        if self._type == 'liar' and random.random() < 0.8:
            block = self._randomBlock(block)

            # Normal and strong agents give a correct and full visualization of a block
        return '{"size": ' + str(block['size']) + ', "shape": ' + str(block['shape']) + ', "colour": "' + str(
            block['colour']) + '"}'

    def _randomBlock(self, block):
        """
        This method returns a random goal block for the lying agent
        """
        return random.choice([item for item in self._goalBlocks
                              if self._accurateVisualization(item['visualization']) != self._accurateVisualization(
                block)])['visualization']

    def _getRoom(self, room):
        # Non-Liars return the correct room
        if self._type is not 'liar' or random.random() > 0.8:
            return room

        # Lying agent returns a random door 80% of the time
        return random.choice(list(self._doors))

    def _accurateVisualization(self, block):
        """
		This visualization is used by the agent to check whether two blocks are the same according to him/her
		"""

        # Colorblind agent cannot see color
        if self._type == 'colorblind':
            return '{"size": ' + str(block['size']) + ', "shape": ' + str(block['shape']) + '"}'

        # Other agent can compare everything
        return '{"size": ' + str(block['size']) + ', "shape": ' + str(block['shape']) + ', "colour": "' + str(
            block['colour']) + '"}'

    def _createPath(self, state: State):
        self._navigator.reset_full()
        # make new waypoint to lower left and right corner of room
        coordinate = self._door['location']
        coordinate_1 = coordinate[0] - 2, coordinate[1] - 1
        coordinate_2 = coordinate[0] + 1, coordinate[1] - 1
        self._navigator.add_waypoints([coordinate_1, coordinate_2])

    def _followPathToClosedDoor(self, state: State):
        self._state_tracker.update(state)
        # Follow path to door
        action = self._navigator.get_move_action(self._state_tracker)
        return action

    def _planPathToCLosedDoor(self, state: State):
        self._navigator.reset_full()
        closedDoors = [door for door in state.values() if
                       'class_inheritance' in door and 'Door' in door['class_inheritance'] and not door['is_open']]
        if len(closedDoors) == 0:
            return None, {}
        # Randomly pick a closed door
        self._door = random.choice(closedDoors)
        # self._door = closedDoors[2]
        doorLoc = self._door['location']
        # Location in front of door is south from door
        doorLoc = doorLoc[0], doorLoc[1] + 1
        # Send message of current action
        self._sendMessage('Moving to ' + self._getRoom(self._door['room_name']), self._agentName)
        self._navigator.add_waypoints([doorLoc])

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

    def _sendReputations(self):
        msg = "Reputations - "

        if self._type is not 'liar' or random.random() > 0.8:
            for member in self._trustBeliefs.keys():
                # update trustBeliefs for each member
                self._trustBeliefs[member] = self._observations[member]['truths'] / (
                        self._observations[member]['truths'] + self._observations[member]['lies'])

                # add name and reputation of each agent
                msg += str(member) + ": " + str(self._trustBeliefs[member]) + ", "

        # Lying agent just makes up random numbers
        else:
            for member in self._trustBeliefs.keys():
                # update trustBeliefs for each member
                self._trustBeliefs[member] = self._observations[member]['truths'] / (
                        self._observations[member]['truths'] + self._observations[member]['lies'])

                # add name and random number for each agent
                msg += str(member) + ": " + str(random.random()) + ", "

        # send the message
        self._sendMessage(msg[:-2], self._agentName)

    def _updateReputations(self, reputations, sender):
        for reputation in reputations.split(", "):
            id = str(reputation.split(": ")[0])
            if id == self._agentName:
                continue
            rep = float(reputation.split(": ")[1])
            diff = rep - self._trustBeliefs[id]
            self._trustBeliefs[id] += diff * self._trustBeliefs[sender] / 5.0

    def _sendIndirectObservation(self, agentId, isTrue: bool):
        # Liar agent lies 80% of the time
        if self._type == 'liar' and random.random() < 0.8:
            isTrue = not isTrue

        if isTrue:
            self._sendMessage(agentId + ' told the truth', self._agentName)
        else:
            self._sendMessage(agentId + ' has lied', self._agentName)


    def _trustBelief(self, member, received):
        '''
		Baseline implementation of a trust belief. Creates a dictionary with trust belief scores for each team member, for example based on the received messages.
		'''
        # You can change the default value to your preference
        default = 0.5
        trustBeliefs = {}

        # Initialize _observations
        if self._observations == {}:
            for member in received.keys():
                self._observations[member] = {"truths": 1, "lies": 1}

        received_new = {}
        for member in received.keys():
            # Only parse the new messages
            if member in self._amountOfMessages.keys():
                received_new[member] = received[member][self._amountOfMessages[member]:]
            else:
                received_new[member] = received[member]

        for member in received_new.keys():
            trustBeliefs[member] = default
        for member in received_new.keys():
            for i, message in enumerate(received_new[member]):

                # if 'Found' in message and 'colour' not in message:
                # 	trustBeliefs[member]-=0.1
                # 	break
                if 'Opening door of' in message:
                    roomname = message.split(' ')[-1]
                    for door in self._doors.values():
                        if door['room_name'] == roomname:
                            if self._doors[roomname]['is_open'] and not self._doorsPrevious[roomname]['is_open']:
                                self._observations[member]['truths'] += 1
                                self._sendIndirectObservation(member, True)
                                self._sendReputations()
                            else:
                                self._observations[member]['lies'] += 1
                                self._sendIndirectObservation(member, False)
                                self._sendReputations()
                if 'Found goal block' in message:
                    visualization = " ".join(message.split(" ")[3:-4])
                    location = " ".join(message.split(" ")[-2:])
                    self._messages['found'].append({'id': member, 'block': visualization + " at " + location})
                if 'Picking up goal block' in message:
                    visualization = " ".join(message.split(" ")[4:-3])
                    location = " ".join(message.split(" ")[-2:])
                    self._messages['picked up'].append({'id': member, 'block': visualization + " at " + location})
                if 'Dropped goal block' in message:
                    visualization = " ".join(message.split(" ")[3:-5])
                    location = " ".join(message.split(" ")[-2:])
                    self._messages['dropped'].append({'id': member, 'block': visualization + " at " + location})
                if 'told the truth' in message:
                    aboutAgent = message.split(" ")[0]
                    if aboutAgent == self._agentName:
                        continue
                    self._observations[aboutAgent]['truths'] += self._trustBeliefs[member]
                if 'has lied' in message:
                    aboutAgent = message.split(" ")[0]
                    if aboutAgent == self._agentName:
                        continue
                    self._observations[aboutAgent]['lies'] += self._trustBeliefs[member]
                if 'Reputation' in message:
                    self._updateReputations(message.split(" - ")[1], member)

        self._amountOfMessages = {}
        for member in received.keys():
            self._amountOfMessages[member] = len(received[member])
            trustBeliefs[member] = self._observations[member]['truths'] / (
                    self._observations[member]['truths'] + self._observations[member]['lies'])
        self._trustBeliefs = trustBeliefs

        return trustBeliefs


class NormalAgent(BaseAgent):
    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._type = 'normal'

class StrongAgent(BaseAgent):
    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._type = 'strong'

class LyingAgent(BaseAgent):
    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._type = 'liar'

class ColorblindAgent(BaseAgent):
    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._type = 'colorblind'

class LazyAgent(BaseAgent):
    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._type = 'lazy'