import os
import numpy as np
import math

class forwardModel():
	def __init__(self, transition_matrix, observation_matrix, initial_state, T, T_set = False):
		"""
		:param transition_matrix: 状态转移概率矩阵,shape = (N, N)
		:param observation_matrix: 观察概率矩阵,shape = (N, M)
		:param initial_state: π, 初始状态概率分布,shape = (1, N)
		:param T: 时间序列数目
		:param T_set: 时间序列集合
		"""
		self.transition_matrix = transition_matrix
		self.observation_matrix = observation_matrix
		self.init_state = initial_state
		self.T_numbers = T

		self.state_number = np.size(self.transition_matrix, axis = 0)
		self.observation_number = np.size(self.observation_matrix, axis = 1)
		self.T_series = None
		if T_set == False:
			self.T_series = np.random.randint(low = 0, high = self.observation_number, size = (self.T_numbers))
		else:
			self.T_series = T_set

	def matrix_rowSum_check(self):
		# 计算状态转移概率矩阵的每一行是否和为1
		rowSum = np.sum(self.transition_matrix, axis = 1)
		try:
			print("转移矩阵每一行总和为", rowSum)
		except ValueError:
			print("值不对")
		else:
			print('验证结束')
		# 计算状态观测概率矩阵每一行是否为1
		rowSum = np.sum(self.observation_matrix, axis = 1)
		try:
			print("转移矩阵每一行总和为", rowSum)
		except ValueError:
			print("值不对")
		else:
			print('验证结束')

	def forward(self):
		# 对观测概率矩阵进行转置
		B_matrix = self.observation_matrix.T
		# Pi最开始的向量表示为self.init_state
		Pi = self.init_state
		# 初始状态
		Pi = Pi * B_matrix[0]
		if len(self.T_series) > 1:
			for value in self.T_series[1:]:
				# 与状态转移矩阵相乘
				Pi = np.dot(Pi, self.transition_matrix)
				Pi = Pi * B_matrix[value]
				print(Pi)
		return np.sum(Pi)

	def backward(self):
		return -1


if __name__ == '__main__':
	tran_matrix = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
	B = np.array([[0.5, 0.5],[0.4, 0.6], [0.7, 0.3]])
	T = 3
	initial_state = np.array([0.2, 0.4, 0.4])
	model = forwardModel(transition_matrix = tran_matrix,
						 initial_state = initial_state,
						 observation_matrix = B,
						 T = T,
						 T_set = [0, 1, 0])
	model.matrix_rowSum_check()
	print(model.T_series)
	Pi = model.forward()
	print(Pi)

