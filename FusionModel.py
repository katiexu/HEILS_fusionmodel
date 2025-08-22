import torch
import torchquantum as tq
from math import pi
import numpy as np
import torch.nn as nn


def translator(single_code, enta_code, trainable, arch_code, fold=1):
    def gen_arch(change_code, base_code):  # start from 1, not 0
        # arch_code = base_code[1:] * base_code[0]
        n_qubits = base_code[0]
        arch_code = ([i for i in range(2, n_qubits + 1, 1)] + [1]) * base_code[1]
        if change_code != None:
            if type(change_code[0]) != type([]):
                change_code = [change_code]

            for i in range(len(change_code)):
                q = change_code[i][0]  # the qubit changed
                for id, t in enumerate(change_code[i][1:]):
                    arch_code[q - 1 + id * n_qubits] = t
        return arch_code

    def prune_single(change_code):
        single_dict = {}
        single_dict['current_qubit'] = []
        if change_code != None:
            if type(change_code[0]) != type([]):
                change_code = [change_code]
            length = len(change_code[0])
            change_code = np.array(change_code)
            change_qbit = change_code[:, 0] - 1
            change_code = change_code.reshape(-1, length)
            single_dict['current_qubit'] = change_qbit
            j = 0
            for i in change_qbit:
                single_dict['qubit_{}'.format(i)] = change_code[:, 1:][j].reshape(-1, 2).transpose(1, 0)
                j += 1
        return single_dict

    def qubit_fold(jobs, phase, fold=1):
        if fold > 1:
            job_list = []
            for job in jobs:
                q = job[0]
                if phase == 0:
                    job_list.append([2 * q] + job[1:])
                    job_list.append([2 * q - 1] + job[1:])
                else:
                    job_1 = [2 * q]
                    job_2 = [2 * q - 1]
                    for k in job[1:]:
                        if q < k:
                            job_1.append(2 * k)
                            job_2.append(2 * k - 1)
                        elif q > k:
                            job_1.append(2 * k - 1)
                            job_2.append(2 * k)
                        else:
                            job_1.append(2 * q)
                            job_2.append(2 * q - 1)
                    job_list.append(job_1)
                    job_list.append(job_2)
        else:
            job_list = jobs
        return job_list
    single_code = qubit_fold(single_code, 0, fold)
    enta_code = qubit_fold(enta_code, 1, fold)
    n_qubits = arch_code[0]
    n_layers = arch_code[1]

    updated_design = {}
    updated_design = prune_single(single_code)
    net = gen_arch(enta_code, arch_code)

    if trainable == 'full' or enta_code == None:
        updated_design['change_qubit'] = None
    else:
        if type(enta_code[0]) != type([]): enta_code = [enta_code]
        updated_design['change_qubit'] = enta_code[-1][0]

    # num of layers
    updated_design['n_layers'] = n_layers

    for layer in range(updated_design['n_layers']):
        # categories of single-qubit parametric gates
        for i in range(n_qubits):
            updated_design['rot' + str(layer) + str(i)] = 'U3'
        # categories and positions of entangled gates
        for j in range(n_qubits):
            if net[j + layer * n_qubits] > 0:
                updated_design['enta' + str(layer) + str(j)] = ('CU3', [j, net[j + layer * n_qubits] - 1])
            else:
                updated_design['enta' + str(layer) + str(j)] = ('CU3', [abs(net[j + layer * n_qubits]) - 1, j])

    updated_design['total_gates'] = updated_design['n_layers'] * n_qubits * 2
    return updated_design

class TQLayer(tq.QuantumModule):
    def __init__(self, n_qubits, design, seq_length,n_class):
        super().__init__()
        self.n_qubits = n_qubits
        self.design = design
        self.qdev = tq.QuantumDevice(n_wires=n_qubits, bsz=1)
        self.g=0.5
        self.M=self.get_M_matrix()
        self.seq_length = seq_length
        self.fc = nn.Linear(seq_length*3, n_class)
        
        self.rots, self.entas = tq.QuantumModuleList(), tq.QuantumModuleList()
        # self.design['change_qubit'] = 3
        self.q_params_rot, self.q_params_enta = [], []
        for i in range(self.n_qubits):
            self.q_params_rot.append(pi * torch.rand(self.design['n_layers'], 3)) # each U3 gate needs 3 parameters
            self.q_params_enta.append(pi * torch.rand(self.design['n_layers'], 3)) # each CU3 gate needs 3 parameters

        for layer in range(self.design['n_layers']):
            for q in range(self.n_qubits):
                # 'trainable' option
                if self.design['change_qubit'] is None:
                    rot_trainable = True
                    enta_trainable = True
                elif q == self.design['change_qubit']:
                    rot_trainable = True
                    enta_trainable = True
                else:
                    rot_trainable = False
                    enta_trainable = False
                # single-qubit parametric gates
                if self.design['rot' + str(layer) + str(q)] == 'U3':
                     self.rots.append(tq.U3(has_params=True, trainable=rot_trainable,
                                           init_params=self.q_params_rot[q][layer]))
                # entangled gates
                if self.design['enta' + str(layer) + str(q)][0] == 'CU3':
                    self.entas.append(tq.CU3(has_params=True, trainable=enta_trainable,
                                             init_params=self.q_params_enta[q][layer]))
    def get_M_matrix(self):
        g = self.g
        Mi = torch.tensor([[1.0, np.exp(-g ** 2 / 2)],
                        [np.exp(-g ** 2 / 2), 1]], dtype=torch.complex64)

        # 对第一个量子位应用测量，其他量子位用单位矩阵
        M = Mi
        for _ in range(self.n_qubits - 1):
            M = torch.kron(M, Mi)
        return M

    def vqc(self,kob):
        state_vector = density_matrix_to_states(kob)

        # 设置状态到量子设备
        self.qdev.set_states(state_vector)
        for layer in range(self.design['n_layers']):
            for j in range(self.n_qubits):
                if not (j in self.design['current_qubit'] and self.design['qubit_{}'.format(j)][1][layer] == 0):
                    self.rots[j + layer * self.n_qubits](self.qdev, wires=j)

            for j in range(self.n_qubits):
                if self.design['enta' + str(layer) + str(j)][1][0] != self.design['enta' + str(layer) + str(j)][1][1]:
                    self.entas[j + layer * self.n_qubits](self.qdev, wires=self.design['enta' + str(layer) + str(j)][1])

        # 获取状态向量并转换为密度矩阵
        state_vector = self.qdev.states.reshape(-1)  # 展平为向量
        density_matrix = torch.outer(state_vector, state_vector.conj())
        return density_matrix

    def xtorho(self, x_values):
        rho_list = []
        for x in x_values:
            rho_A = torch.tensor([
                [1.0 - x, np.sqrt((1.0 - x) * x)],
                [np.sqrt((1.0 - x) * x), x]
            ], dtype=torch.complex64)
            rho_list.append(rho_A)

        # 使用 torch.kron 替代 jnp.kron
        rho=rho_list[0]
        for i in range(1,len(rho_list)):
            rho = torch.kron(rho_list[i], rho)
        return rho

    def partial_trace(self, rho, keep_wires):
        dim_keep = 2 ** keep_wires
        dim_trace = 2 ** (self.n_qubits - keep_wires)

        # 将密度矩阵重塑为 (dim_trace, dim_keep, dim_trace, dim_keep)
        rho_reshaped = rho.reshape((dim_trace, dim_keep, dim_trace, dim_keep))

        # 使用 einsum 计算部分迹: 对第一个和第三个索引求和
        rho_reduced = torch.einsum('ijik->jk', rho_reshaped)

        return rho_reduced

    def get_first_3qubit_expvals(self, rho):
        """分别计算前3个qubit的Pauli-Z期望值"""
        expvals = []

        # 定义Pauli-Z矩阵
        pauli_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
        identity = torch.eye(2, dtype=torch.complex64)

        # 为每个qubit构造对应的观测算符
        for qubit_idx in [0, 1, 2]:  # 前3个qubit
            # 构造观测算符：目标qubit是Z，其他是I
            obs = None
            for i in range(self.n_qubits):
                if i == qubit_idx:
                    component = pauli_z
                else:
                    component = identity

                if obs is None:
                    obs = component
                else:
                    obs = torch.kron(obs, component)

            # 计算期望值：Tr(ρ·O)
            expval = torch.trace(rho @ obs).real
            expvals.append(expval)

        return torch.stack(expvals)
    def single_forward(self,sequence):
        rho = self.xtorho([0, 0, 0,0,0,0])
        all_value=[]
        for x in sequence:
            rho=self.partial_trace(rho,self.n_qubits-len(x))

            rho=torch.kron(rho, self.xtorho(x))
            rho=self.vqc(rho)

            rho = self.M @ rho @ self.M.conj().T
            rho /= torch.trace(rho).real

            # 测量
            expval = self.get_first_3qubit_expvals(rho)
            all_value.append(expval)
        return torch.stack(all_value).flatten()

    def forward(self, x):
        x=torch.stack([self.single_forward(batch) for batch in x])
        x=self.fc(x)
        return x


def density_matrix_to_states(rho, n_samples=1):
    """
    将密度矩阵转换为可能的状态向量样本。

    对于纯态密度矩阵，直接返回对应的状态向量。
    对于混合态，通过对角化分解并采样。

    Args:
        rho (torch.Tensor): 密度矩阵 [dim, dim]
        n_samples (int): 要生成的样本数

    Returns:
        torch.Tensor: 状态向量 [n_samples, dim]
    """
    # 确保输入是密度矩阵
    assert rho.dim() == 2 and rho.size(0) == rho.size(1)

    # 确保矩阵是厄米特矩阵（Hermitian）
    rho = (rho + rho.conj().T) / 2

    # 添加小的正则化项以确保正定性
    epsilon = 1e-10
    identity = torch.eye(rho.size(0), device=rho.device, dtype=rho.dtype)
    rho = (1 - epsilon) * rho + epsilon * identity / rho.size(0)

    # 确保迹为1
    trace = torch.trace(rho).real
    if abs(trace - 1.0) > 1e-6:
        rho = rho / trace

    # 使用简单的幂迭代法近似最大特征向量
    v = torch.rand(rho.size(0), dtype=rho.dtype, device=rho.device)
    v = v / torch.norm(v)

    for _ in range(10):  # 10次迭代通常足够
        v = torch.matmul(rho, v)
        v = v / torch.norm(v)

    return v.unsqueeze(0).repeat(n_samples, 1)
if __name__ == '__main__':
    n_layers=4
    n_qubits=6
    single = [[i] + [1] * 2 * n_layers for i in range(1, n_qubits + 1)]
    enta = [[i] + [i + 1] * n_layers for i in range(1, n_qubits)] + [[n_qubits] + [1] * n_layers]

    design = translator(single, enta, 'full', (n_qubits, n_layers), 1)
    model=TQLayer(n_qubits, design,16,4)

    text_data=torch.rand(32,16,3)
    res=model(text_data)
    print(res.shape)
    print(res)



