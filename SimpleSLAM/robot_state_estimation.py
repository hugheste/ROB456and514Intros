#!/usr/bin/env python3

import numpy as np

from world_state import WorldState
from door_sensor import DoorSensor


# Belief about world/robot state
class RobotStateEstimation:
    def __init__(self):

        # Probability representation (discrete)
        self.probabilities = []
        self.reset_probabilities(10)

        # Kalman (Gaussian) probabilities
        self.mean = 0.5
        self.standard_deviation = 0.4
        self.reset_kalman()

    def reset_probabilities(self, n_probability):
        """ Initialize discrete probability resolution with uniform distribution """
        div = 1.0 / n_probability
        self.probabilities = np.ones(n_probability) * div

    def update_belief_sensor_reading(self, ws, ds, sensor_reading_has_door):
        """ Update your probabilities based on the sensor reading being true (door) or false (no door)
        :param ws World state - has where the doors are
        :param ds Door Sensor - has probabilities for door sensor readings
        :param sensor_reading_has_door - contains true/false from door sensor
        """
        new_probs = np.zeros(len(self.probabilities))
        # begin homework 2 : problem 3
        # Normalize - all the denominators are the same because they're the sum of all cases
        # P(B|A=door): ds.prob_see_door_if_door
        # P(B|A=no door): ds.prob_see_door_if_no_door
        # P(A): Prob. door, starts at 0.5
        # P(B): Prob. no door, starts at 0.5
        # Trying to find P(A|B): Probability robot is in front of door given sensor reading

        #div = 1.0 / len(self.probabilities)
        #new_probs = np.ones(len(self.probabilities)) * div

        print("before ws.doors: ",ws.doors)
        print("after ws.doors: ",ws.doors[0] * len(self.probabilities))
        print("after ws.doors: ", int(ws.doors[0] * len(self.probabilities)))
        print("self.probabilities: ",self.probabilities)
        print("sensor_reading_has_door: ",sensor_reading_has_door)

        door_indexes = np.zeros(len(ws.doors))
        for i in range(len(ws.doors)):
            door_indexes[i] = int(ws.doors[i] * len(self.probabilities))

        for i in range(len(self.probabilities)):
            for j in range(len(door_indexes)):
                if i == door_indexes[j]:
                    if sensor_reading_has_door:
                        # likelihood * belief
                        new_probs[i] = ds.prob_see_door_if_door * self.probabilities[i]
                    else:
                        new_probs[i] = ds.prob_see_door_if_no_door * self.probabilities[i]

        # Normalize
        print("new_probs: ",new_probs)
        total = sum(self.probabilities)
        print("total: ",total)
        #if total != 0:
        for i in range(len(new_probs)):
            new_probs[i] = new_probs[i]/total

        self.probabilities = new_probs

        #numerator = ds.prob_see_door_if_door * 0.5
        #denominator = ds.prob_see_door_if_door * 0.5 + ds.prob_see_door_if_no_door * 0.5
        #bayes = numerator/denominator

        # end homework 2 : problem 3

    # Distance to wall sensor (state estimation)
    def update_dist_sensor(self, ws, dist_reading):
        """ Update state estimation based on sensor reading
        :param ws - for standard deviation of wall sensor
        :param dist_reading - distance reading returned from the sensor, in range 0,1 (essentially, robot location) """

        # Standard deviation of error
        standard_deviation = ws.wall_standard_deviation
        # begin homework 2 : Extra credit
            # Sample from probability
        # Normalize - all the denominators are the same
        # end homework 2 : Extra credit
        return self.mean, self.standard_deviation

    def update_belief_move_left(self, rs):
        """ Update the probabilities assuming a move left.
        :param rs - robot state, has the probabilities"""

        # begin homework 2 problem 4
        # Check probability of left, no, right sum to one
        if rs.prob_move_left_if_left + rs.prob_move_right_if_left + rs.prob_no_move_if_left != 1:
            print("Error: Probabilities are not equal to 1!")
            rs.prob_no_move_if_left = 1 - (rs.prob_move_left_if_left + rs.prob_move_right_if_left)

        # Left edge - put move left probability into zero square along with stay-put probability
        size = len(self.probabilities)
        new_probs = np.zeros(size)
        new_probs[0] = rs.prob_move_left_if_left + rs.prob_no_move_if_left

        # Right edge - put move right probability into last square
        new_probs[size-1] = rs.prob_move_right_if_left

        # Normalize - sum should be one, except for numerical rounding
        new_probs /= sum(new_probs)
        print("In update_belief_move_left, new_probs: ",new_probs)

        '''
        # Determine new belief based on move left
        move = rs.move_left()
        n = len(self.probabilities)
        result = np.zeros(n)
        for i in range(n):
            result[i] = self.probabilities[(i - move) % n]
        self.probabilities[:] = result
        '''

        '''
        new_probs = np.zeros(len(self.probabilities))
        prob_move_left_if_left = rs.prob_move_left_if_left
        for i in range(len(self.probabilities)):
            if rs.move_left < 0:
                new_probs[i] = rs.prob_move_left_if_left * self.probabilities[i]
            else:
                new_probs[i] = rs.prob_move_right_if_left * self.probabilities[i]

        # Normalize
        total = sum(new_probs)
        for i in range(len(new_probs)):
            new_probs[i] /= total

        self.probabilities = new_probs
        '''

        # end homework 2 problem 4

    def update_belief_move_right(self, rs):
        """ Update the probabilities assuming a move right.
        :param rs - robot state, has the probabilities"""

        # begin homework 2 problem 4
        # Check probability of left, no, right sum to one
        # Left edge - put move left probability into zero square along with stay-put probability
        # Right edge - put move right probability into last square
        # Normalize - sum should be one, except for numerical rounding
        # end homework 2 problem 4

    # Put robot in the middle with a really broad standard deviation
    def reset_kalman(self):
        self.mean = 0.5
        self.standard_deviation = 0.4

    # Given a movement, update Gaussian
    def update_kalman_move(self, rs, amount):
        """ Kalman filter update mean/standard deviation with move (the prediction step)
        :param rs : robot state - has the standard deviation error for moving
        :param amount : The requested amount to move
        :return : mean and standard deviation of my current estimated location """

        # begin homework 3 : Problem 2
        # end homework 3 : Problem 2
        return self.mean, self.standard_deviation

    # Sensor reading, distance to wall (Kalman filtering)
    def update_gauss_sensor_reading(self, ws, dist_reading):
        """ Update state estimation based on sensor reading
        :param ws - for standard deviation of wall sensor
        :param dist_reading - distance reading returned"""

        # begin homework 3 : Problem 1
        # end homework 3 : Problem 1
        return self.mean, self.standard_deviation


if __name__ == '__main__':
    ws_global = WorldState()

    ds_global = DoorSensor()

    rse_global = RobotStateEstimation()

    # Check out these cases
    # We have two possibilities - either in front of door, or not - cross two sensor readings
    #   saw door versus not saw door
    uniform_prob = rse_global.probabilities[0]

    # begin homework 2 problem 4
    # Four cases - based on default door probabilities of
    # DoorSensor.prob_see_door_if_door = 0.8
    # DoorSensor.prob_see_door_if_no_door = 0.2
    #  and 10 probability divisions. Three doors visible.
    # probability saw door if door, saw door if no door, etc
    # Resulting probabilities, assuming 3 doors
    # Check that our probabilities are updated correctly
    # Spacing of bins
    # end homework 2 problem 4

    print("Passed tests")
