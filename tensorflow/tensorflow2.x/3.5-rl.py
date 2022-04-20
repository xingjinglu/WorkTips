import gym

# 实例化一个游戏环境，参数为游戏名称
env = gym.make('CartPole-v1')

# 初始化环境，获得初始状态
state = env.reset()
while True:
	env.render()
	action = model.predict(state)

	# 让环境执行动作，获得执行完动作的的下一个状态，动作的奖励，游戏是否已经结束及额外信息
	next_state, reward, done, info = env.step(action)
	if done:
		break

