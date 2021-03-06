import math

import numpy as np

import gym
from gym import spaces, logger
from gym.utils import seeding

class TransitCartPole:

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02
        self.kinematics_integrator = 'euler'
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        high = np.array([self.x_threshold * 2,
                         np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2,
                         np.finfo(np.float32).max],
                        dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.steps_beyond_done = None

    def __call__(self, states, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        x, x_dot, theta, theta_dot = states
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                    self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        nextStates = np.array([x, x_dot, theta, theta_dot])
        return nextStates


class IsTerminalCartPole:

    def __init__(self):
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

    def __call__(self, states):
        x, x_dot, theta, theta_dot = states
        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        return done


class ResetCartPole:

    def __init__(self, seed):
        self.seed = seed

    def __call__(self):
        np_random, seed = seeding.np_random(self.seed)
        states = np_random.uniform(low=-0.05, high=0.05, size=(4,))
        return np.array(states)


class RewardCartPole:

    def __init__(self):
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

    def __call__(self, states, action, nextStates):
        steps_beyond_done = None
        x, x_dot, theta, theta_dot = nextStates
        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        if not done:
            reward = 1.0
        elif steps_beyond_done is None:
            steps_beyond_done = 0
            reward = 1.0
        else:
            if steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            steps_beyond_done += 1
            reward = 0.0
        return reward


class VisualizeCartPole:

    def __init__(self):
        self.viewer = None
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        self.length = 0.5

    def __call__(self, trajectory):
        mode = 'human'
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        for timeStep in range(len(trajectory)):
            states = trajectory[timeStep]
            if self.viewer is None:
                from gym.envs.classic_control import rendering
                self.viewer = rendering.Viewer(screen_width, screen_height)
                l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
                axleoffset = cartheight / 4.0
                cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                self.carttrans = rendering.Transform()
                cart.add_attr(self.carttrans)
                self.viewer.add_geom(cart)
                l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
                pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                pole.set_color(.8, .6, .4)
                self.poletrans = rendering.Transform(translation=(0, axleoffset))
                pole.add_attr(self.poletrans)
                pole.add_attr(self.carttrans)
                self.viewer.add_geom(pole)
                self.axle = rendering.make_circle(polewidth / 2)
                self.axle.add_attr(self.poletrans)
                self.axle.add_attr(self.carttrans)
                self.axle.set_color(.5, .5, .8)
                self.viewer.add_geom(self.axle)
                self.track = rendering.Line((0, carty), (screen_width, carty))
                self.track.set_color(0, 0, 0)
                self.viewer.add_geom(self.track)

                self._pole_geom = pole

            if states is None:
                return None

            # Edit the pole polygon vertex
            pole = self._pole_geom
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole.v = [(l, b), (l, t), (r, t), (r, b)]

            x = states
            cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
            self.carttrans.set_translation(cartx, carty)
            self.poletrans.set_rotation(-x[2])
            self.viewer.render(return_rgb_array=mode == 'rgb_array')

        if self.viewer:
            self.viewer.close()
            self.viewer = None

        return





