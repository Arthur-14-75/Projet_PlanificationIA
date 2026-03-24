class MultiAgentBuffer:
    def __init__ (self):
        self.observations = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []

    def add (self, observation, action, logprob, reward, done):
        self.observations.append(observation)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.dones.clear()