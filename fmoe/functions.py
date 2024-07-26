r"""
The fmoe.functions module contains functions that are directly warped up from
C/CUDA functions to complete distributed communication, computation and gradient
computation.
"""

import torch
import torch.distributed as dist
from torch.autograd import Function
import fmoe_cuda
from .utils import get_torch_default_comm
import copy

_moe_group = None
import time

def ensure_comm(t, comm):
    if comm is None:
        comm = get_torch_default_comm()
    global _moe_group
    _moe_group = comm
    fmoe_cuda.ensure_nccl(comm, t)


def get_moe_group():
    return _moe_group


def count_by_gate(gate, num_expert_in_devices, experts_to_devices, world_size, require_pos=True):
    a_gemm = 1.4851e-3
    b_gemm = 5.9699e-2 / 1e12
    a_intra = 0.9624e-4
    b_intra = 7.2716e-4 / 1e6
    a_inter = 1.72e-5
    b_inter = 2.96e-10
    M = 512
    H = 1024
    W = M * H * 2 + M + H
    max_num_expert_in_bag = 32
    use_DP = 1

    def writeFile_add(filename, data):
        file_handle = open(filename, mode='a')
        file_handle.write(data)
        file_handle.close()

    # print('call count_by_gate by ', gate.device)
    # print('gate.device.type', type(gate.device))
    # print('experts_to_devices: ', experts_to_devices, gate.device)
    with torch.no_grad():
        local_expert_count = torch.zeros(
            sum(len(row) for row in experts_to_devices), device=gate.device, dtype=torch.int32
        )
        fmoe_cuda.expert_count(gate, local_expert_count)
        local_expert_count = local_expert_count.long()

        group = []
        for i in range(world_size):
            group.append(i)
        group = dist.new_group(group)

        local_expert_count_sum = torch.zeros_like(local_expert_count)

        local_expert_count_list = [torch.zeros_like(local_expert_count) for _ in range(world_size)]
        dist.all_gather(local_expert_count_list, local_expert_count, group=group)
        # res = 'local_expert_count_list: {}\n device: {}\n'.format(local_expert_count_list, gate.device)
        # filename = '/home/jianhongbai/gyq/BalanceMOE/balancemoe/examples/transformer-xl/LM-TFM-enwik8/count_by_gate.txt'
        # writeFile_add(filename, res)
        # print('local_expert_count_list:', local_expert_count_list, 'device:', gate.device, '\n')

        if use_DP == 1:
            if gate.device == torch.device('cuda:0'):
                for i in range(world_size):
                    local_expert_count_sum += local_expert_count_list[i]

                reorder_expert_count_sum = torch.zeros_like(local_expert_count_sum)
                experts_to_devices_copy = sum(copy.deepcopy(experts_to_devices), [])
                experts_to_devices_copy = torch.tensor(experts_to_devices_copy, dtype=torch.long).to(gate.device)
                index = 0

                for i in range(len(local_expert_count)):
                    reorder_expert_count_sum[index] = local_expert_count_sum[int(experts_to_devices_copy[i].item())]
                    index = index + 1

                local_expert_count_sum = local_expert_count_sum.tolist()
                reorder_expert_count_sum = reorder_expert_count_sum.tolist()

                # balancemoe expert transfer mechanism
                def count_computing_load(num_expert_in_devices, reorder_expert_count_sum):
                    computing_load = []
                    start_index = 0

                    for num in num_expert_in_devices:
                        end_index = start_index + num
                        computing_load.append(sum(reorder_expert_count_sum[start_index: end_index]))
                        start_index = end_index
                    return computing_load

                computing_load = count_computing_load(num_expert_in_devices, reorder_expert_count_sum)
                # print(computing_load)

                def get_worker_pair(computing_load):
                    sorted_index = sorted(range(len(computing_load)), key=lambda k: computing_load[k])

                    worker_pair = [[sorted_index[i], sorted_index[-i - 1]] for i in range(len(sorted_index) // 2)]

                    return worker_pair

                worker_pair = get_worker_pair(computing_load)  # small-big
                # print(worker_pair)

                def computing_tranfer_expert(worker_pair, expert_pond, num_expert_in_pair, bag_capacity,
                                             max_num_expert_in_bag,
                                             expert_weight, expert_comp_value, expert_comm_value, expert_transfer_matrix):
                    K = [[[0 for w in range(bag_capacity + 1)]
                          for j in range(max_num_expert_in_bag + 1)]
                         for i in range(num_expert_in_pair + 1)]

                    non_overlap_comp = [[[0 for w in range(bag_capacity + 1)]
                                         for j in range(max_num_expert_in_bag + 1)]
                                        for i in range(num_expert_in_pair + 1)]

                    smaller_chosen = []

                    # Build table K[][] in bottom up manner
                    for i in range(num_expert_in_pair + 1):
                        for j in range(max_num_expert_in_bag + 1):
                            for w in range(bag_capacity + 1):
                                if i == 0 or w == 0 or j == 0:
                                    K[i][j][w] = 0
                                    non_overlap_comp[i][j][w] = 0
                                elif expert_weight[i - 1] <= w and j > 0:
                                    if expert_comm_value[i - 1] == 0:
                                        if expert_comp_value[i - 1] + K[i - 1][j - 1][w - expert_weight[i - 1]] >= \
                                                K[i - 1][j][w]:
                                            K[i][j][w] = expert_comp_value[i - 1] + K[i - 1][j - 1][
                                                w - expert_weight[i - 1]]
                                            if expert_comp_value[i - 1] == 0:
                                                non_overlap_comp[i][j][w] = non_overlap_comp[i - 1][j - 1][
                                                    w - expert_weight[i - 1]]
                                            else:
                                                non_overlap_comp[i][j][w] = expert_comp_value[i - 1] + \
                                                                            non_overlap_comp[i - 1][j - 1][
                                                                                w - expert_weight[i - 1]]
                                        else:
                                            K[i][j][w] = K[i - 1][j][w]
                                            non_overlap_comp[i][j][w] = non_overlap_comp[i - 1][j][w]
                                    else:
                                        if expert_comp_value[i - 1] + K[i - 1][j - 1][w - expert_weight[i - 1]] + \
                                                min(0, non_overlap_comp[i - 1][j - 1][w - expert_weight[i - 1]] -
                                                       expert_comm_value[i - 1]) \
                                                > K[i - 1][j][w]:
                                            K[i][j][w] = expert_comp_value[i - 1] + K[i - 1][j - 1][
                                                w - expert_weight[i - 1]] + \
                                                         min(0, non_overlap_comp[i - 1][j - 1][w - expert_weight[i - 1]] -
                                                             expert_comm_value[i - 1])
                                            non_overlap_comp[i][j][w] = max(0, non_overlap_comp[i - 1][j - 1][w - expert_weight[i - 1]] - expert_comm_value[i - 1])
                                        else:
                                            K[i][j][w] = K[i - 1][j][w]
                                            non_overlap_comp[i][j][w] = non_overlap_comp[i - 1][j][w]
                                else:
                                    K[i][j][w] = K[i - 1][j][w]
                                    non_overlap_comp[i][j][w] = non_overlap_comp[i - 1][j][w]

                    res = K[num_expert_in_pair][max_num_expert_in_bag][bag_capacity]
                    total_non_overlap_comp = non_overlap_comp[num_expert_in_pair][max_num_expert_in_bag][bag_capacity]
                    # print("Maximum value: ", res)
                    # print("total_non_overlap_comp: ", total_non_overlap_comp)

                    # item selection
                    w = bag_capacity
                    for i in range(num_expert_in_pair, 0, -1):
                        if res <= 0:
                            break
                        if res == K[i - 1][max_num_expert_in_bag][w]:
                            continue
                        else:
                            smaller_chosen.append(i - 1)
                            if expert_comm_value[i - 1] == 0:
                                total_non_overlap_comp = total_non_overlap_comp - expert_comp_value[i - 1]
                                res = res - expert_comp_value[i - 1]
                            else:
                                total_non_overlap_comp = total_non_overlap_comp + expert_comm_value[i - 1]
                                res = res - expert_comp_value[i - 1] - min(0, total_non_overlap_comp - expert_comm_value[
                                    i - 1])
                            w = w - expert_weight[i - 1]
                            max_num_expert_in_bag -= 1

                    bigger_chosen = list(set(range(num_expert_in_pair)) - set(smaller_chosen))

                    for i in smaller_chosen:
                        if i >= len(expert_pond[0]):
                            expert_transfer_matrix[expert_pond[1][i - len(expert_pond[0])]][0] = worker_pair[1]  # source worker
                            expert_transfer_matrix[expert_pond[1][i - len(expert_pond[0])]][1] = worker_pair[0]  # destination worker

                    for i in bigger_chosen:
                        if i < len(expert_pond[0]):
                            expert_transfer_matrix[expert_pond[0][i]][0] = worker_pair[0]  # source worker
                            expert_transfer_matrix[expert_pond[0][i]][1] = worker_pair[1]  # destination worker

                    return expert_transfer_matrix

                expert_transfer_matrix = [[-1 for i in range(2)] for j in range(sum(num_expert_in_devices))]

                start_time = time.time()
                for i in range(len(worker_pair)):  # stay in experts with smaller computing load
                    num_expert_in_pair = num_expert_in_devices[worker_pair[i][0]] + num_expert_in_devices[worker_pair[i][1]]
                    bag_capacity = (computing_load[worker_pair[i][0]] + computing_load[worker_pair[i][1]]) // 2

                    if bag_capacity == 0:
                        continue

                    expert_pond = [experts_to_devices[worker_pair[i][0]], experts_to_devices[worker_pair[i][1]]]
                    expert_weight = []
                    expert_comp_value = []
                    expert_comm_value = []
                    for j in experts_to_devices[worker_pair[i][0]]:
                        expert_weight.append(local_expert_count_sum[j])
                        if local_expert_count_sum[j] != 0:
                            expert_comp_value.append(2 * b_gemm * local_expert_count_sum[j] * M * H + 2 * a_gemm)
                        else:
                            expert_comp_value.append(0)
                        expert_comm_value.append(0)
                    for j in experts_to_devices[worker_pair[i][1]]:
                        expert_weight.append(local_expert_count_sum[j])
                        if local_expert_count_sum[j] != 0:
                            expert_comp_value.append(2 * b_gemm * local_expert_count_sum[j] * M * H + 2 * a_gemm)
                        else:
                            expert_comp_value.append(0)
                        # using b_inter and a_inter if workers in worker_pair belong to different nodes
                        expert_comm_value.append(2 * b_intra * W + 2 * a_intra)

                    expert_transfer_matrix = computing_tranfer_expert(worker_pair[i], expert_pond,
                                                                      num_expert_in_pair, bag_capacity,
                                                                      max_num_expert_in_bag,
                                                                      expert_weight, expert_comp_value, expert_comm_value,
                                                                      expert_transfer_matrix)

                    # print('The items packed are', expert_transfer_matrix)
                print('======================== DP time:', time.time() - start_time, '=========================')

                new_num_expert_in_devices = []
                new_experts_to_devices = copy.deepcopy(experts_to_devices)

                for i in range(sum(num_expert_in_devices)):
                    if expert_transfer_matrix[i][0] != -1:
                        new_experts_to_devices[expert_transfer_matrix[i][0]].remove(i)
                        new_experts_to_devices[expert_transfer_matrix[i][1]].append(i)

                for i in range(world_size):
                    new_num_expert_in_devices.append(len(new_experts_to_devices[i]))

                # print('old_experts_to_devices:', experts_to_devices)
                # print('new_experts_to_devices:', new_experts_to_devices)
                # print('new_num_expert_in_devices:', new_num_expert_in_devices)

                expert_transfer_matrix = torch.tensor(expert_transfer_matrix, dtype=torch.long).to(gate.device)
                ######################################

                # new_experts_to_devices = [[3], [0, 1, 2]]
                # new_num_expert_in_devices = [1, 3]
                max_len = max(len(row) for row in new_experts_to_devices)
                for row in new_experts_to_devices:
                    row.extend([-1] * (max_len - len(row)))
                max_len = torch.tensor(max_len, dtype=torch.long).to(gate.device)
                new_experts_to_devices = sum(new_experts_to_devices, [])
            else:
                expert_transfer_matrix = [[-1 for i in range(2)] for j in range(sum(num_expert_in_devices))]
                expert_transfer_matrix = torch.tensor(expert_transfer_matrix, dtype=torch.long).to(gate.device)

                max_len = torch.tensor(0, dtype=torch.long).to(gate.device)
                # print('do not find rank 0!')

            dist.broadcast(expert_transfer_matrix, src=0, group=group)
            # print('expert_transfer_matrix:', expert_transfer_matrix)
            dist.broadcast(max_len, src=0, group=group)
            # print('max_len:', max_len)
            if gate.device == torch.device('cuda:0'):
                new_experts_to_devices = torch.tensor(new_experts_to_devices, dtype=torch.long).to(gate.device)
                new_num_expert_in_devices = torch.tensor(new_num_expert_in_devices, dtype=torch.long).to(gate.device)
            else:
                new_experts_to_devices = torch.zeros(world_size * int(max_len.item()), dtype=torch.long).to(gate.device)
                new_num_expert_in_devices = torch.zeros(world_size, dtype=torch.long).to(gate.device)
            # print('new_experts_to_devices:', new_experts_to_devices)

            # dist.broadcast(local_expert_count_sum, src=0, group=group)

        ################################################################################################################
        ######### no expert transfer ###################################################################################
        ################################################################################################################
        else:
            if gate.device == torch.device('cuda:0'):
                expert_transfer_matrix = [[-1 for i in range(2)] for j in range(sum(num_expert_in_devices))]
                expert_transfer_matrix = torch.tensor(expert_transfer_matrix, dtype=torch.long).to(gate.device)

                max_len = max(len(row) for row in experts_to_devices)
                for row in experts_to_devices:
                    row.extend([-1] * (max_len - len(row)))
                max_len = torch.tensor(max_len, dtype=torch.long).to(gate.device)
                new_experts_to_devices = sum(experts_to_devices, [])
            else:
                expert_transfer_matrix = [[-1 for i in range(2)] for j in range(sum(num_expert_in_devices))]
                expert_transfer_matrix = torch.tensor(expert_transfer_matrix, dtype=torch.long).to(gate.device)

                max_len = torch.tensor(0, dtype=torch.long).to(gate.device)

            dist.broadcast(expert_transfer_matrix, src=0, group=group)
            # print('expert_transfer_matrix:', expert_transfer_matrix)
            dist.broadcast(max_len, src=0, group=group)
            # print('max_len:', max_len)
            if gate.device == torch.device('cuda:0'):
                experts_to_devices_copy = sum(copy.deepcopy(experts_to_devices), [])
                new_experts_to_devices = torch.tensor(experts_to_devices_copy, dtype=torch.long).to(gate.device)
                new_num_expert_in_devices = torch.tensor(num_expert_in_devices, dtype=torch.long).to(gate.device)
            else:
                new_experts_to_devices = torch.zeros(world_size * int(max_len.item()), dtype=torch.long).to(gate.device)
                new_num_expert_in_devices = torch.zeros(world_size, dtype=torch.long).to(gate.device)
        ################################################################################################################
        ################################################################################################################
        ################################################################################################################

        dist.broadcast(new_experts_to_devices, src=0, group=group)
        dist.broadcast(new_num_expert_in_devices, src=0, group=group)
        reorder_local_expert_count = torch.zeros_like(local_expert_count)
        index = 0
        for i in range(world_size * int(max_len.item())):
            if int(new_experts_to_devices[i].item()) != -1:
                reorder_local_expert_count[index] = local_expert_count[int(new_experts_to_devices[i].item())]
                index = index + 1

        # res = 'reorder_local_expert_count: {}\n new_experts_to_devices: {}\n device: {}\n'.format(reorder_local_expert_count, new_experts_to_devices, gate.device)
        # filename = '/home/jianhongbai/gyq/BalanceMOE/balancemoe/examples/transformer-xl/LM-TFM-enwik8/count_by_gate.txt'
        # writeFile_add(filename, res)

        # new_experts_to_devices = new_experts_to_devices.tolist()
        # print('new_experts_to_devices:', new_experts_to_devices)
        device_str = str(gate.device)
        device_index = int(device_str.split(':')[1])
        # new_num_expert_in_devices = new_num_expert_in_devices.tolist()
        num_expert = int(new_num_expert_in_devices[device_index].item())
        global_expert_count_zeros = torch.zeros(world_size, num_expert, dtype=torch.long).to(gate.device)
        # print('num_expert:', num_expert, gate.device)

        if world_size > 1:
            # print('call global_expert_count!')
            global_expert_count = fmoe_cuda.expert_exchange(
                local_expert_count, num_expert, new_experts_to_devices, max_len.item(), reorder_local_expert_count,
                global_expert_count_zeros, world_size
            )
            # print('call global_expert_count over!')
            # global_expert_count = fmoe_cuda.expert_exchange(
            #     local_expert_count, num_expert, world_size
            # )
        else:
            global_expert_count = local_expert_count
        if not require_pos:
            pos = None
        else:
            lec_cum = torch.cumsum(reorder_local_expert_count, dim=0).int()
            pos_size = lec_cum[-1].item()
            pos = torch.empty((pos_size,), device=gate.device, dtype=torch.long)
            fake_gate = torch.zeros_like(gate)
            acc_gap = 0
            for i in range(len(gate)):
                for j in range(len(new_experts_to_devices)):
                    if int(new_experts_to_devices[j].item()) == -1:
                        acc_gap += 1
                    if gate[i] == new_experts_to_devices[j]:
                        fake_gate[i] = j - acc_gap
                acc_gap = 0
            fmoe_cuda.assign_pos(lec_cum, fake_gate, pos)

    return pos, local_expert_count, global_expert_count, num_expert_in_devices, experts_to_devices, \
           new_num_expert_in_devices, new_experts_to_devices, max_len.item(), reorder_local_expert_count, \
           expert_transfer_matrix


def prepare_forward(gate, num_expert_in_devices, experts_to_devices, world_size):
    r"""
    Prepare necessary information from gate output for MoE computation.

    Args:
        gate: a 1-d Long Tensor representing the target expert of each input
        sample.
        num_expert: number of experts on each worker.
        world_size: number of workers that hold different experts.
        comm: the communicator of all workers in the expert-parallel group.
    """
    pos, local_expert_count, global_expert_count, num_expert_in_devices, experts_to_devices, new_num_expert_in_devices,\
    new_experts_to_devices, max_len, reorder_local_expert_count, expert_transfer_matrix = count_by_gate(gate,
            num_expert_in_devices, experts_to_devices, world_size)

    # balancemoe
    device_str = str(gate.device)
    device_index = int(device_str.split(':')[1])
    num_expert = int(new_num_expert_in_devices[device_index].item())
    ############

    with torch.no_grad():
        fwd_expert_count = global_expert_count.view(world_size,
                num_expert).sum(dim=0)
        fwd_batch_size = int(fwd_expert_count.sum().item())

    local_expert_count_cpu = local_expert_count.cpu()
    global_expert_count_cpu = global_expert_count.cpu()
    reorder_local_expert_count_cpu = reorder_local_expert_count.cpu()
    fwd_expert_count_cpu = fwd_expert_count.cpu()
    new_experts_to_devices_cpu = new_experts_to_devices.cpu()
    new_num_expert_in_devices_cpu = new_num_expert_in_devices.cpu()

    del local_expert_count
    del global_expert_count
    del reorder_local_expert_count
    del fwd_expert_count
    del new_experts_to_devices
    del new_num_expert_in_devices
    torch.cuda.empty_cache()

    # return (
    #     pos,
    #     local_expert_count.cpu(),
    #     global_expert_count.cpu(),
    #     reorder_local_expert_count.cpu(),
    #     fwd_expert_count.cpu(),
    #     fwd_batch_size,
    #     num_expert,
    #     experts_to_devices,
    #     num_expert_in_devices,
    #     new_experts_to_devices.cpu(),
    #     new_num_expert_in_devices.cpu(),
    #     max_len,
    #     expert_transfer_matrix,
    # )
    return (
        pos,
        local_expert_count_cpu,
        global_expert_count_cpu,
        reorder_local_expert_count_cpu,
        fwd_expert_count_cpu,
        fwd_batch_size,
        num_expert,
        experts_to_devices,
        num_expert_in_devices,
        new_experts_to_devices_cpu,
        new_num_expert_in_devices_cpu,
        max_len,
        expert_transfer_matrix,
    )


def _local_scatter(inp, pos):
    # print("inp first dimension is", inp.shape[0])
    # print("pos:", pos)
    inp_buf = torch.index_select(inp, 0, pos)
    return inp_buf


def _local_gather(inp, pos, out_batch_size, maybe_overlap=True):
    inp_buf = torch.zeros(out_batch_size, inp.shape[-1],
            dtype=inp.dtype, device=inp.device)
    if maybe_overlap:
        inp_buf.index_add_(0, pos, inp)
    else:
        inp_buf.index_copy_(0, pos, inp)
    return inp_buf


class MOEScatter(Function):
    r"""
    Scatter input samples from [batch x sequences] to contiguous alone experts.
    If `world_size` is greater than 1, the samples will first be locally
    scattered, and then exchanged across workers.
    """

    @staticmethod
    def forward(
        ctx,
        inp,
        pos,
        local_expert_count,
        global_expert_count,
        reorder_local_expert_count,
        fwd_batch_size,
        world_size,
        num_expert,
        new_experts_to_devices,
        max_len,
    ):
        def writeFile_add(filename, data):
            file_handle = open(filename, mode='a')
            file_handle.write(data)
            file_handle.close()

        local_input_buf = _local_scatter(inp, pos)

        # res = 'local_input_buf: {}\n '.format(local_input_buf)
        # filename = '/home/jianhongbai/gyq/BalanceMOE/balancemoe/examples/transformer-xl/LM-TFM-enwik8/MOEScatter.txt'
        # writeFile_add(filename, res)

        if world_size > 1:
            global_input_buf = fmoe_cuda.global_scatter(
                local_input_buf,
                reorder_local_expert_count,
                global_expert_count,
                fwd_batch_size,
                world_size,
                num_expert,
                new_experts_to_devices,
                max_len,
            )
            # res = 'local_input_buf: {}\n global_input_buf: {}\n'.format(local_input_buf, global_input_buf)
            # filename = '/home/jianhongbai/gyq/BalanceMOE/balancemoe/examples/transformer-xl/LM-TFM-enwik8/MOEScatter.txt'
            # writeFile_add(filename, res)
        else:
            global_input_buf = local_input_buf
        ctx.moe_args = inp.shape[0], pos.shape[0], world_size, num_expert, max_len
        variables = (pos, reorder_local_expert_count, global_expert_count, new_experts_to_devices)
        ctx.save_for_backward(*variables)
        return global_input_buf

    @staticmethod
    def backward(ctx, global_grad_in):
        (pos, reorder_local_expert_count, global_expert_count, new_experts_to_devices) = ctx.saved_tensors
        (inp_batch_size, buf_batch_size, world_size, num_expert, max_len) = ctx.moe_args

        if world_size > 1:
            local_grad_in = fmoe_cuda.global_gather(
                global_grad_in,
                reorder_local_expert_count,
                global_expert_count,
                buf_batch_size,
                world_size,
                num_expert,
                new_experts_to_devices,
                max_len,
            )
        else:
            local_grad_in = global_grad_in
        grad_in = _local_gather(local_grad_in, pos, inp_batch_size)
        return grad_in, None, None, None, None, None, None, None, None, None

class MOEGather(Function):
    r"""
    Gather output samples from contiguous alone experts back to [batch x
    sequences]. Works symmetrically with MOEScatter.
    """

    @staticmethod
    def forward(
        ctx,
        global_output_buf,
        pos,
        local_expert_count,
        global_expert_count,
        reorder_local_expert_count,
        local_batch_size,
        world_size,
        num_expert,
        new_experts_to_devices,
        max_len,
    ):
        if world_size > 1:
            local_output_buf = fmoe_cuda.global_gather(
                global_output_buf,
                reorder_local_expert_count,
                global_expert_count,
                pos.shape[0],
                world_size,
                num_expert,
                new_experts_to_devices,
                max_len,
            )
        else:
            local_output_buf = global_output_buf
        output = _local_gather(local_output_buf, pos, local_batch_size,
                maybe_overlap=False)

        ctx.moe_args = (global_output_buf.shape[0], world_size, num_expert, max_len)
        variables = (pos, reorder_local_expert_count, global_expert_count, new_experts_to_devices)
        ctx.save_for_backward(*variables)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        pos, reorder_local_expert_count, global_expert_count, new_experts_to_devices = ctx.saved_tensors
        fwd_batch_size, world_size, num_expert, max_len = ctx.moe_args
        grad_out_buf = _local_scatter(grad_out.contiguous(), pos)
        if world_size > 1:
            global_grad_out_buf = fmoe_cuda.global_scatter(
                grad_out_buf,
                reorder_local_expert_count,
                global_expert_count,
                fwd_batch_size,
                world_size,
                num_expert,
                new_experts_to_devices,
                max_len,
            )
        else:
            global_grad_out_buf = grad_out_buf
        return global_grad_out_buf, None, None, None, None, None, None, None, None, None


class AllGather(Function):
    r"""
    A wrapper for the All-Gather function to support auto-differentiation.
    """

    @staticmethod
    def forward(ctx, inp, rank, world_size, group):
        tensor_list = [torch.empty_like(inp) for _ in range(world_size)]
        torch.distributed.all_gather(tensor_list, inp, group=group)
        torch.cuda.synchronize()
        output = torch.cat(tensor_list, dim=0)
        ctx.args = rank, inp.shape[0]
        return output

    @staticmethod
    def backward(ctx, grad_out):
        rank, dim0 = ctx.args
        return grad_out[rank * dim0 : (rank + 1) * dim0], None, None, None


class Slice(Function):
    r"""
    A wrapper for the Slice function to support auto-differentiation.
    """

    @staticmethod
    def forward(ctx, inp, rank, world_size, group):
        B: int = inp.shape[0]
        local_batch_size = B // world_size
        batch_start = local_batch_size * rank
        batch_end = min(batch_start + local_batch_size, B)
        inp = inp[batch_start:batch_end]
        ctx.args = world_size, group
        return inp

    @staticmethod
    def backward(ctx, grad_out):
        world_size, group = ctx.args
        tensor_list = [torch.empty_like(grad_out) for _ in range(world_size)]
        torch.distributed.all_gather(tensor_list, grad_out, group=group)
        torch.cuda.synchronize()
        grad_out = torch.cat(tensor_list, dim=0)
        return grad_out, None, None, None
