import gym

if __name__ == '__main__':
    env = gym.make('FrozenLake8x8-v0')
    obs = env.reset()

    for t in range(1000):
        env.render()

        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        if done:
            env.render()
            print(f'Episode finished after {t+1} timesteps')
            break

    env.close()
