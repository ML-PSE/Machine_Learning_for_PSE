#%% Tank class
import numpy as np
import random

class tank_environment:
    """
    Description:
        Create an OpenAI style environment
    """
    
    def __init__(self, pre_def_dist=False, pre_distFlow=[1]):
        # all dimesnions are in SI unit
        # tank related
        self.height = 10  
        self.radius = 7
        self.level = 0.5 * self.height # initial level is at 50%
        self.hard_max_level = 0.8 * self.height
        self.hard_min_level = 0.2 * self.height
        self.soft_max_level = 0.525 * self.height
        self.soft_min_level = 0.475 * self.height
        self.pipe_radius = 0.5
        self.pipe_Aout = np.pi*self.pipe_radius*self.pipe_radius
        
        # disturbance related
        self.pre_def_dist = pre_def_dist 
        self.pre_distFlow = pre_distFlow # 1D numpy array contains disturbance flows used during testing
        self.distFlow = [1] # stores disturbance flows during a training episode
    
    def get_disturbanceFlow(self, t=0):
        """
        Arguments: 
            t: corresponds to the step number of an episode
        Returns: a (scalar) disturbance flow
        """
        
        if self.pre_def_dist:
            return self.pre_distFlow[t]
        else:
            new_flow = random.normalvariate(self.distFlow[-1], 0.1)
            # impose bounds on disturbance flow
            if new_flow > 2:
                new_flow = 2
            elif new_flow < 0.7:
                new_flow = 0.7
            
            self.distFlow.append(new_flow)
            return new_flow
   
    def step(self, action):
        """ 
        Description: 
            accepts an action and returns a tuple (observation, reward, done).
        
        Args:
            action (valve opening): an action provided by the agent
            
        Returns:
            observation (numpy array of size (3,)): agent's observation of the current environment
            reward (float) : amount of reward returned after taking the action
            done (bool): whether the episode has ended
        """
        
        # parameters
        g = 9.81
        
        for i in range(5):
            # compute rate of change of tank level
            q_dist = self.get_disturbanceFlow()
            q_out = action*self.pipe_Aout*np.sqrt(2*g*self.level)
            dhdt = (q_dist - q_out)/(np.pi*self.radius*self.radius)
            
            # compute new tank level
            self.level = self.level + dhdt
             
            # check termination status
            done = False
            if self.level < self.hard_min_level:
                done = True
                break
            elif self.level > self.hard_max_level:
                done = True
                break
        
        # check level above 50%
        if self.level >= 0.5*self.height:
            above = 1
        else:
            above = 0
        
        # compute reward
        if done:
            reward = -10
        elif self.level > self.soft_min_level and self.level < self.soft_max_level:
            reward = 1
        else:
            reward = 0
        
        # generate observation/state vector
        next_state = np.array([self.level/self.height, dhdt, above])
        
        return next_state, reward, done
    
    def step_test(self, action, t):
        """
        Run one timestep of the environment's dynamics. 
        Accepts an action and returns a tuple (observation, reward, done, info).
        
        Args:
            action (valve opening): an action provided by the agent
            t(step number): used to fetch correct pre-defined disturbance flow 
            
        Returns:
            observation (numpy array of size (3,)): agent's observation of the current environment
            reward (float) : amount of reward returned after taking the action
            done (bool): indicates whether an episode has ended
        """
        
        # parameters
        g = 9.81
        
        # compute rate of change of tank level
        q_dist = self.get_disturbanceFlow(t)
        q_out = action*self.pipe_Aout*np.sqrt(2*g*self.level)
        dhdt = (q_dist - q_out)/(np.pi*self.radius*self.radius)
        
        # compute new tank level
        self.level = self.level + dhdt
         
        # check termination status
        done = False
        if self.level < self.hard_min_level:
            done = True
        elif self.level > self.hard_max_level:
            done = True
             
        # check level above 50%
        if self.level >= 0.5*self.height:
            above = 1
        else:
            above = 0
        
        # compute reward
        if done:
            reward = -10
        elif self.level > self.soft_min_level and self.level < self.soft_max_level:
            reward = 1
        else:
            reward = 0
        
        # generate observation/state vector
        next_state = np.array([self.level/self.height, dhdt, above]) # using normalized level
        
        return next_state, reward, done
    
    def reset(self):
        """
        Description: Reset tank environment to initial conditions
        Returns: Initial state of the environment 

        """
        self.level = 0.5 * self.height
        self.distFlow = [1]
        
        return np.array([self.level/self.height, 0, 1])
