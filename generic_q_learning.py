import numpy as np

class Q_Agent:
    def __init__(self, alpha, gamma, location_to_state, actions, rewards, state_to_location, Q):
        self.alpha = alpha
        self.gamma = gamma

        self.location_to_state = location_to_state
        self.actions = actions
        self.rewards = rewards
        self.state_to_location = state_to_location
        
        self.Q = Q


    def training(self, start_location, end_location, iterations):
        self.Q = np.zeros([9,9])

        rewards_copy = np.copy(self.rewards)

        ending_state = self.location_to_state[end_location]

        rewards_copy[ending_state, ending_state] = 999

        for i in range(iterations):
            current_state = np.random.randint(0,9)

            playable_actions = []

            for j in range(len(self.actions)):
                if rewards_copy[current_state, j] > 0:
                    playable_actions.append(j)

            next_state = np.random.choice(playable_actions)

            TD = rewards_copy[current_state, next_state] + self.gamma*self.Q[next_state, np.argmax(self.Q[next_state, ])] - self.Q[current_state, next_state]
            self.Q[current_state, next_state] += self.alpha * TD

        route = [start_location]
        next_location = start_location

        self.get_optimal_route(start_location=start_location, end_location=end_location, next_location=next_location, route = route)


    def get_optimal_route(self, start_location, end_location, next_location, route):
            while(next_location != end_location):
                starting_state = self.location_to_state[start_location]
                next_state = np.argmax(self.Q[starting_state, ])
                next_location = self.state_to_location[next_state]
                route.append(next_location)
                start_location = next_location

            print(route)



rewards = np.array([[0,1,0,0,0,0,0,0,0],
              [1,0,1,0,1,0,0,0,0],
              [0,1,0,0,0,1,0,0,0],
              [0,0,0,0,0,0,1,0,0],
              [0,1,0,0,0,0,0,1,0],
              [0,0,1,0,0,0,0,0,0],
              [0,0,0,1,0,0,0,1,0],
              [0,0,0,0,1,0,1,0,1],
              [0,0,0,0,0,0,0,1,0]])

alpha = 0.9
gamma = 0.75
location_to_state = {
    "L1" : 0,
    "L2" : 1,
    "L3" : 2,
    "L4" : 3,
    "L5" : 4,
    "L6" : 5,
    "L7" : 6,
    "L8" : 7,
    "L9" : 8
}

actions = [0,1,2,3,4,5,6,7,8]
state_to_location = dict((state, location) for location, state in location_to_state.items())

Q = np.zeros((9,9))

qagent = Q_Agent(alpha, gamma, location_to_state, actions, rewards, state_to_location, Q)

qagent.training("L9", "L1", 1000)



    
