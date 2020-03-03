from dialogue_config import usersim_default_key, FAIL,UNSUITABLE, NO_OUTCOME, SUCCESS, NO_VALUE, GOOD_INFORM, usersim_required_init_inform_keys, \
    no_query_keys
from utils import reward_function
import random, copy


class UserSimulator:
    """Simulates a real user, to train the agent with reinforcement learning."""

    def __init__(self, goal_list, constants, database):
        """
        The constructor for UserSimulator. Sets dialogue config variables.

        Parameters:
            goal_list (list): User goals loaded from file
            constants (dict): Dict of constants loaded from file
            database (dict): The database in the format dict(long: dict)
        """

        self.goal_list = goal_list
        self.max_round = constants['run']['max_round_num']
        self.default_key = usersim_default_key
        # A list of REQUIRED to be in the first action inform keys
        self.init_informs = usersim_required_init_inform_keys
        self.no_query = no_query_keys
        self.agent_additional_informs = []
        self.success = NO_OUTCOME
        # TEMP ----
        self.database = database
        # ---------
        self.empty_count = 0
    def reset(self):
        """
        Resets the user sim. by emptying the state and returning the initial action.

        Returns:
            dict: The initial action of an episode
        """

        self.goal = random.choice(self.goal_list)
        # Add default slot to requests of goal
        self.goal['request_slots'][self.default_key] = 'UNK'
        self.state = {}
        # Add all inform slots informed by agent or user sim to this dict
        self.state['history_slots'] = {}
        # Any inform slots for the current user sim action, empty at start of turn
        self.state['inform_slots'] = {}
        # Current request slots the user sim wants to request
        self.state['request_slots'] = {}
        # Init. all informs and requests in user goal, remove slots as informs made by user or agent
        self.state['rest_slots'] = {}
        self.state['rest_slots'].update(self.goal['inform_slots'])
        self.state['rest_slots'].update(self.goal['request_slots'])
        self.state['intent'] = ''
        # False for failure, true for success, init. to failure
        self.constraint_check = FAIL
        self.success = NO_OUTCOME
        return self._return_init_action()
    def reset_empty_count(self):
        total = self.empty_count
        self.empty_count = 0
        return total
    def _return_init_action(self):
        """
        Returns the initial action of the episode.

        The initial action has an intent of request, required init. inform slots and a single request slot.

        Returns:
            dict: Initial user response
        """

        # Always request
        self.state['intent'] = 'request'

        if self.goal['inform_slots']:
            # Pick all the required init. informs, and add if they exist in goal inform slots
            for inform_key in self.init_informs:
                if inform_key in self.goal['inform_slots']:
                    self.state['inform_slots'][inform_key] = self.goal['inform_slots'][inform_key]
                    self.state['rest_slots'].pop(inform_key)
                    self.state['history_slots'][inform_key] = self.goal['inform_slots'][inform_key]
            # If nothing was added then pick a random one to add
            if not self.state['inform_slots']:
                key, value = random.choice(list(self.goal['inform_slots'].items()))
                self.state['inform_slots'][key] = value
                self.state['rest_slots'].pop(key)
                self.state['history_slots'][key] = value

        # Now add a request, do a random one if something other than def. available
        self.goal['request_slots'].pop(self.default_key)
        if self.goal['request_slots']:
            req_key = random.choice(list(self.goal['request_slots'].keys()))
        else:
            req_key = self.default_key
        self.goal['request_slots'][self.default_key] = 'UNK'
        self.state['request_slots'][req_key] = 'UNK'

        user_response = {}
        user_response['intent'] = self.state['intent']
        user_response['request_slots'] = copy.deepcopy(self.state['request_slots'])
        user_response['inform_slots'] = copy.deepcopy(self.state['inform_slots'])
        print("user: {}".format(user_response))
        return user_response

    def step(self, agent_action):
        """
        Return the response of the user sim. to the agent by using rules that simulate a user.

        Given the agent action craft a response by using deterministic rules that simulate (to some extent) a user.
        Some parts of the rules are stochastic. Check if the agent has succeeded or lost or still going.

        Parameters:
            agent_action (dict): The agent action that the user sim. responds to

        Returns:
            dict: User sim. response
            int: Reward
            bool: Done flag
            int: Success: -1, 0 or 1 for loss, neither win nor loss, win
        """
        
        # Assertions -----
        # No UNK in agent action informs
        for value in agent_action['inform_slots'].values():
            assert value != 'UNK'
            assert value != 'PLACEHOLDER'
        # No PLACEHOLDER in agent at all
        for value in agent_action['request_slots'].values():
            assert value != 'PLACEHOLDER'
        # ----------------

        self.state['inform_slots'].clear()
        # self.state['intent'] = ''

        done = False
        self.success = NO_OUTCOME
        # if user intent is thanks and agent not reply with done intent, then punish it
        if self.state['intent'] == 'thanks' and agent_action['intent'] != 'done':
            self.success = UNSUITABLE
        # First check round num, if equal to max then fail
        if agent_action['round'] == self.max_round:
            # print("max round reached")
            done = True
            self.success = FAIL
            self.state['intent'] = 'done'
            self.state['request_slots'].clear()
        else:
            agent_intent = agent_action['intent']
            if agent_intent == 'request':
                self._response_to_request(agent_action)
            elif agent_intent == 'inform':
                self._response_to_inform(agent_action)
            elif agent_intent == 'match_found':
                self._response_to_match_found(agent_action)
            elif agent_intent == 'done':
                self._response_to_done()
                self.state['intent'] = 'done'
                self.state['request_slots'].clear()
                done = True

        # Assumptions -------
        # If request intent, then make sure request slots
        if self.state['intent'] == 'request':
            assert self.state['request_slots']
        # If inform intent, then make sure inform slots and NO request slots
        if self.state['intent'] == 'inform':
            assert self.state['inform_slots']
            assert not self.state['request_slots']
        assert 'UNK' not in self.state['inform_slots'].values()
        assert 'PLACEHOLDER' not in self.state['request_slots'].values()
        # No overlap between rest and hist
        for key in self.state['rest_slots']:
            assert key not in self.state['history_slots']
        for key in self.state['history_slots']:
            assert key not in self.state['rest_slots']
        # All slots in both rest and hist should contain the slots for goal
        for inf_key in self.goal['inform_slots']:
            # assert self.state['history_slots'].get(inf_key, False) or self.state['rest_slots'].get(inf_key, False)
            assert inf_key in self.state['history_slots'] or inf_key in self.state['rest_slots'], inf_key

        
        for req_key in self.goal['request_slots']:
            # assert self.state['history_slots'].get(req_key, False) or self.state['rest_slots'].get(req_key,
            #                                                                                        False), req_key
            assert req_key in self.state['history_slots'] or req_key in self.state['rest_slots'], req_key
        # Anything in the rest should be in the goal
        for key in self.state['rest_slots']:
            # assert self.goal['inform_slots'].get(key, False) or self.goal['request_slots'].get(key, False)
            assert key in self.goal['inform_slots'] or key in self.goal['request_slots'], key
        assert self.state['intent'] != ''
        # -----------------------

        user_response = {}
        user_response['intent'] = self.state['intent']
        user_response['request_slots'] = copy.deepcopy(self.state['request_slots'])
        user_response['inform_slots'] = copy.deepcopy(self.state['inform_slots'])

        reward = reward_function(self.success, self.max_round)

        return user_response, reward, done, True if self.success is 1 else False

    def _response_to_request(self, agent_action):
        """
        Augments the state in response to the agent action having an intent of request.

        There are 4 main cases for responding.

        Parameters:
            agent_action (dict): Intent of request with standard action format (including 'speaker': 'Agent' and
                                 'round_num': int)
        """
        # success = NO_OUTCOME
        agent_request_key = list(agent_action['request_slots'].keys())[0]
        if agent_request_key in self.state['request_slots'] or agent_request_key in self.state['history_slots']:
            self.success = UNSUITABLE
        
        # First Case: if agent requests for something that is in the user sims goal inform slots, then inform it
        if agent_request_key in self.goal['inform_slots']:
            
            self.state['intent'] = 'inform'
            self.state['inform_slots'][agent_request_key] = self.goal['inform_slots'][agent_request_key]
            self.state['request_slots'].clear()
            self.state['rest_slots'].pop(agent_request_key, None)
            self.state['history_slots'][agent_request_key] = self.goal['inform_slots'][agent_request_key]
        # Second Case: if the agent requests for something in user sims goal request slots and it has already been
        # informed, then inform it
        elif agent_request_key in self.goal['request_slots'] and agent_request_key in self.state['history_slots']:
            self.success = UNSUITABLE
            self.state['intent'] = 'inform'
            self.state['inform_slots'][agent_request_key] = self.state['history_slots'][agent_request_key]
            self.state['request_slots'].clear()
            assert agent_request_key not in self.state['rest_slots']
        # Third Case: if the agent requests for something in the user sims goal request slots and it HASN'T been
        # informed, then request it with a random inform
        elif agent_request_key in self.goal['request_slots'] and agent_request_key in self.state['rest_slots']:
            self.state['request_slots'].clear()
            self.state['intent'] = 'request'
            self.state['request_slots'][agent_request_key] = 'UNK'
            rest_informs = {}
            for key, value in list(self.state['rest_slots'].items()):
                if value != 'UNK':
                    rest_informs[key] = value
            if rest_informs:
                key_choice, value_choice = random.choice(list(rest_informs.items()))
                self.state['inform_slots'][key_choice] = value_choice
                self.state['rest_slots'].pop(key_choice)
                self.state['history_slots'][key_choice] = value_choice
        # Fourth and Final Case: otherwise the user sim does not care about the slot being requested, then inform
        # 'anything' as the value of the requested slot
        else:
            assert agent_request_key not in self.state['rest_slots']
            self.state['intent'] = 'inform'
            self.state['inform_slots'][agent_request_key] = ['anything']
            self.state['request_slots'].clear()
            self.state['history_slots'][agent_request_key] = ['anything']
        # return success

    def _response_to_inform(self, agent_action):
        """
        Augments the state in response to the agent action having an intent of inform.

        There are 2 main cases for responding. Add the agent inform slots to history slots,
        and remove the agent inform slots from the rest and request slots.

        Parameters:
            agent_action (dict): Intent of inform with standard action format (including 'speaker': 'Agent' and
                                 'round_num': int)
        """
        # success = NO_OUTCOME
        agent_inform_key = list(agent_action['inform_slots'].keys())[0]
        agent_inform_value = agent_action['inform_slots'][agent_inform_key]

        assert agent_inform_key != self.default_key
        # if agent_inform_key in self.state['history_slots'].keys():
        #     self.success = UNSUITABLE
        # if (isinstance(agent_inform_value, list) and len(agent_inform_value) == 0) or agent_inform_key in self.state['request_slots'].keys():
        #     self.empty_count += 1
        #     self.success = NO_VALUE
        
        if agent_inform_key in self.state['request_slots'].keys():
            if len(agent_inform_value) == 0:
                print("inform user request: {} with empty value".format(agent_inform_key))
        elif agent_inform_key in self.state['history_slots'].keys():
            self.success = NO_VALUE
        else:
            if isinstance(agent_inform_value, list):
                if len(agent_inform_value) > 0:
                    self.empty_count -= 1
                    self.success = GOOD_INFORM
                else:
                    self.empty_count += 1
                    self.success = UNSUITABLE
        

         # Zero case: If value that agent inform is no match available then random remove 1 slot from inform list
        if agent_inform_value == 'no match available':
            ### warning user and ask them to change value
            #random choose slot to remove
            slots_lists = [key for key, value in self.state['history_slots'].items() if ('anything' not in value and key != self.default_key)]
            if len(slots_lists) > 0: 
                slot_to_remove = random.choice(slots_lists)
                self.state['intent'] = 'inform'
                self.state['inform_slots'][slot_to_remove] = ['anything']
                self.state['request_slots'].clear()
                self.state['history_slots'][slot_to_remove] = ['anything']
                self.goal['inform_slots'].pop(slot_to_remove, None)
                self.state['rest_slots'].pop(slot_to_remove, None)
                # print(self.goal['inform_slots'])
                # print(self.state['history_slots'])
                # print(self.state['rest_slots'])
                # return success

        # Add all informs (by agent too) to hist slots
        self.state['history_slots'][agent_inform_key] = agent_inform_value
        
        # if agent_inform_key in self.state['rest_slots'] and self.state['rest_slots'][agent_inform_key] == 'UNK':
        #   print(">>>>> CONGRATULATIONS: bot can inform user request by itself!")
        # Remove from rest slots if in it
        self.state['rest_slots'].pop(agent_inform_key, None)
        
        # Remove from request slots if in it
        self.state['request_slots'].pop(agent_inform_key, None)
    
       

        # First Case: If agent informs something that is in goal informs and the value it informed doesnt match,
        # then inform the correct value
        if agent_inform_value != self.goal['inform_slots'].get(agent_inform_key, agent_inform_value):
            self.state['intent'] = 'inform'
            self.state['inform_slots'][agent_inform_key] = self.goal['inform_slots'][agent_inform_key]
            self.state['request_slots'].clear()
            self.state['history_slots'][agent_inform_key] = self.goal['inform_slots'][agent_inform_key]
        # Second Case: Otherwise pick a random action to take
        else:
            # - If anything in state requests then request it
            if self.state['request_slots']:
                self.state['intent'] = 'request'
            # - Else if something to say in rest slots, pick something
            elif self.state['rest_slots']:
                def_in = self.state['rest_slots'].pop(self.default_key, False)
                if self.state['rest_slots']:
                    key, value = random.choice(list(self.state['rest_slots'].items()))
                    if value != 'UNK':
                        self.state['intent'] = 'inform'
                        self.state['inform_slots'][key] = value
                        self.state['rest_slots'].pop(key)
                        self.state['history_slots'][key] = value
                    else:
                        self.state['intent'] = 'request'
                        self.state['request_slots'][key] = 'UNK'
                else:
                    self.state['intent'] = 'request'
                    self.state['request_slots'][self.default_key] = 'UNK'
                if def_in == 'UNK':
                    self.state['rest_slots'][self.default_key] = 'UNK'
            # - Otherwise respond with 'nothing to say' intent
            else:
                self.state['intent'] = 'thanks'
                
        # return success
    def _response_to_match_found(self, agent_action):
        """
        Augments the state in response to the agent action having an intent of match_found.

        Check if there is a match in the agent action that works with the current goal.

        Parameters:
            agent_action (dict): Intent of match_found with standard action format (including 'speaker': 'Agent' and
                                 'round_num': int)
        """
        # success = NO_OUTCOME
        
        if self.state['intent'] == 'reject':
            self.success = UNSUITABLE
        agent_informs = agent_action['inform_slots']

        self.state['intent'] = 'thanks'
        self.constraint_check = SUCCESS

        assert self.default_key in agent_informs
        self.state['rest_slots'].pop(self.default_key, None)

        self.state['history_slots'][self.default_key] = str(agent_informs[self.default_key])
        self.state['request_slots'].pop(self.default_key, None)

        if agent_informs[self.default_key] == 'no match available':
            self.constraint_check = FAIL
            history_slots_list = [key for key, value in self.state['history_slots'].items() if ('anything' not in value and key != self.default_key)]
            # print(history_slots_list)
            history_slots_list = [item for item in history_slots_list if item != 'activity']
            # print(history_slots_list)
            if len(history_slots_list) > 0:
                slot_to_remove = random.choice(history_slots_list)
                # print("slot to remove {}".format(slot_to_remove))
                self.state['intent'] = 'inform'
                self.state['inform_slots'][slot_to_remove] = ['anything']
                self.state['request_slots'].clear()
                self.state['history_slots'][slot_to_remove] = ['anything']
                self.goal['inform_slots'].pop(slot_to_remove, None)
                self.state['rest_slots'].pop(slot_to_remove, None)

        # Check to see if all goal informs are in the agent informs, and that the values match
        for key, value in self.goal['inform_slots'].items():
            assert value != None
            # For items that cannot be in the queries don't check to see if they are in the agent informs here
            if key in self.no_query:
                continue
            # Will return true if key not in agent informs OR if value does not match value of agent informs[key]
            if value != agent_informs.get(key, None):
                self.constraint_check = FAIL
                break

        if self.constraint_check == FAIL:
            self.state['intent'] = 'reject'

            ### IMPORTANT: should we clear it?
            self.state['request_slots'].clear()
            # print("reject: goal inform slots: {}".format(self.goal['inform_slots']))
        # print("constraint check: {}".format(self.constraint_check))
        
        
        ##### IMPORTANT: if everythings is ok, then remove request slot to help the agent success
        
        if self.state['intent'] == 'thanks':
            for key in self.goal['request_slots'].keys():
                self.state['rest_slots'].pop(key, None)
                # self.goal['request_slots'].pop(key)
                
                print("success, remove slot {} from rest and request".format(key))
            self.goal['request_slots'].clear()
            
            self.state['request_slots'].clear()
            
        # return success
    def _response_to_done(self):
        """
        Augments the state in response to the agent action having an intent of done.

        If the constraint_check is SUCCESS and both the rest and request slots of the state are empty for the agent
        to succeed in this episode/conversation.

        Returns:
            int: Success: -1, 0 or 1 for loss, neither win nor loss, win
        """
        
        if self.constraint_check == FAIL:
            self.success = FAIL
            return

        if not self.state['rest_slots']:
            assert not self.state['request_slots']
        if self.state['rest_slots']:
            print("rest_slots: {}".format(self.state['rest_slots']))
            self.success = FAIL
            return
        # print("constraint_check: {0}".format(self.constraint_check))
        # TEMP: ----
        assert self.state['history_slots'][self.default_key] != 'no match available'
        # print(int(self.state['history_slots'][self.default_key]))
        match = copy.deepcopy(self.database[int(self.state['history_slots'][self.default_key])])

        for key, value in self.goal['inform_slots'].items():
            assert value != None
            if key in self.no_query:
                continue
            if value != match.get(key, None):
                assert True is False, 'match: {}\ngoal: {}'.format(match, self.goal)
                break
        # ----------

        self.success = SUCCESS
