r"""
FMoE core layer
"""
import tree
import os
import torch
import torch.nn as nn
import torch.distributed as dist

from .functions import prepare_forward, ensure_comm
from .functions import MOEScatter, MOEGather
from .functions import AllGather, Slice
from .gates import NaiveGate

from .fastermoe.config import switch_from_env

import time

def mark_module_parallel_comm(module, comm):
    r"""
    Mark all parameters in `module` as doing data parallel in `comm`, where
    `comm` may be one of `'world', 'dp', 'none'`.
    """
    for p in module.parameters():
        setattr(p, "dp_comm", comm)


def _fmoe_general_global_forward(inp, gate, expert_fn, num_expert_in_devices, experts_to_devices, world_size, **kwargs):
    r"""
    A private function that performs the following steps to complete the MoE
    computation.
    * Count the number of tokens from each worker to each expert.
    * Send the features to their target position so that input features to each
    expert are contiguous in memory.
    * Perform the forward computation of the experts using `expert_fn`
    * Gather the output features of experts back, and reorder them as sentences.
    Intermediate results like expert counts are hidden from users by this
    function.
    """

    def writeFile_add(filename, data):
        file_handle = open(filename, mode='a')
        file_handle.write(data)
        file_handle.close()

    # balancemoe
    # device_str = str(gate.device)
    # device_index = int(device_str.split(':')[1])
    # num_expert = num_expert_in_devices[device_index]
    ############

    # res = 'inp: {}\n gate: {}\n expert_fn: {}\n world_size: {}\n'.format(inp, gate, expert_fn, world_size)
    # filename = '/home/jianhongbai/gyq/BalanceMOE/balancemoe/examples/transformer-xl/LM-TFM-enwik8/input.txt'
    # writeFile_add(filename, res)

    (
        pos,
        local_expert_count,
        global_expert_count,
        reorder_local_expert_count,
        fwd_expert_count,
        fwd_batch_size,
        num_expert,
        old_experts_to_devices,
        old_num_expert_in_devices,
        new_experts_to_devices,
        new_num_expert_in_devices,
        max_len,
        expert_transfer_matrix,
    ) = prepare_forward(gate, num_expert_in_devices, experts_to_devices, world_size)

    # res = 'pos: {}\n local_expert_count: {}\n global_expert_count: {}\n reorder_local_expert_count: {}\n fwd_expert_count: {}\n fwd_batch_size: {}\n'.format(pos, local_expert_count, global_expert_count, reorder_local_expert_count, fwd_expert_count, fwd_batch_size)
    # filename = '/home/jianhongbai/gyq/BalanceMOE/balancemoe/examples/transformer-xl/LM-TFM-enwik8/prepare_forward.txt'
    # writeFile_add(filename, res)

    res = 'device: {} fwd_batch_size: {}\n'.format(pos.device, fwd_batch_size)
    filename = '/home/jianhongbai/gyq/BalanceMOE/balancemoe/examples/transformer-xl/results/workload.txt'
    writeFile_add(filename, res)

    topk = 1
    if len(gate.shape) == 2:
        topk = gate.shape[1]

    def scatter_func(tensor):
        return MOEScatter.apply(
            tensor,
            torch.div(pos, topk, rounding_mode='floor'),
            local_expert_count,
            global_expert_count,
            reorder_local_expert_count,
            fwd_batch_size,
            world_size,
            num_expert,
            new_experts_to_devices,
            max_len,
        )

    x = tree.map_structure(scatter_func, inp)
    # res = 'x: {}\n '.format(x)
    # filename = '/home/jianhongbai/gyq/BalanceMOE/balancemoe/examples/transformer-xl/LM-TFM-enwik8/tree.map_structure.txt'
    # writeFile_add(filename, res)

    # reorder x according to global_expert_count
    global_expert_count_1_dim = global_expert_count.view(-1)
    global_expert_count_1_dim = torch.cumsum(global_expert_count_1_dim, dim=0)
    new_indices = []
    start_idx = 0
    for j in range(num_expert):
        for i in range(world_size):
            start_idx = 0 if i * num_expert + j == 0 else global_expert_count_1_dim[i * num_expert + j - 1]
            new_indices.extend(list(range(start_idx, start_idx + global_expert_count[i][j])))
    new_indices = torch.tensor(new_indices).long().to(gate.device)

    x = torch.index_select(x, 0, new_indices)

    start_time = time.time()

    x = expert_fn(x, fwd_expert_count, num_expert, old_experts_to_devices, old_num_expert_in_devices,
                  new_experts_to_devices.tolist(), new_num_expert_in_devices.tolist(), expert_transfer_matrix.tolist())

    print('======================== expert_fn time:', time.time() - start_time, '=========================')

    # reorder x according to fake_global_expert_count
    fake_global_expert_count = torch.zeros_like(global_expert_count.view(-1))
    for j in range(num_expert):
        for i in range(world_size):
            fake_global_expert_count[j*world_size + i] = global_expert_count[i][j]
    fake_global_expert_count_1_dim = torch.cumsum(fake_global_expert_count, dim=0)
    new_indices = []
    start_idx = 0
    for i in range(world_size):
        for j in range(num_expert):
            start_idx = 0 if j * world_size + i == 0 else fake_global_expert_count_1_dim[j * world_size + i - 1]
            new_indices.extend(list(range(start_idx, start_idx + fake_global_expert_count[j * world_size + i])))
    new_indices = torch.tensor(new_indices).long()
    x = torch.index_select(x, 0, new_indices.to(gate.device))

    # res = 'new_indices: {} {} \n'.format(new_indices, gate.device)
    # filename = '/home/jianhongbai/gyq/BalanceMOE/balancemoe/examples/transformer-xl/LM-TFM-enwik8/tree.map_structure.txt'
    # writeFile_add(filename, res)

    out_batch_size = tree.flatten(inp)[0].shape[0]
    if len(gate.shape) == 2:
        out_batch_size *= gate.shape[1]

    def gather_func(tensor):
        return MOEGather.apply(
            tensor,
            pos,
            local_expert_count,
            global_expert_count,
            reorder_local_expert_count,
            out_batch_size,
            world_size,
            num_expert,
            new_experts_to_devices,
            max_len,
        )

    outp = tree.map_structure(gather_func, x)
    # res = 'outp: {}\n '.format(outp)
    # filename = '/home/jianhongbai/gyq/BalanceMOE/balancemoe/examples/transformer-xl/LM-TFM-enwik8/tree.map_structure.txt'
    # writeFile_add(filename, res)
    return outp, new_experts_to_devices, new_num_expert_in_devices


fmoe_faster_schedule = False
if switch_from_env('FMOE_FASTER_SCHEDULE_ENABLE', False):
    fmoe_faster_schedule = True
    from .fastermoe.schedule import _fmoe_general_global_forward


class FMoE(nn.Module):
    r"""
    A general moe implementation that supports an arbitrary module as the
    expert.
    * `num_expert` stands for the number of experts on **each** worker.
    * `world_size` stands for the total number of workers that contains
    different experts.
    * `slice_group` can be a torch's communication group, indicating that
    specific model parallel is applied across the group, and workers in the
    group hold the same copy of input feature, and requires the same copy of
    the output. For each worker, FMoE only computes the output of a certain
    slice of the input batch, and will all-gather the outputs after
    computation.
    * `mp_group` is a deprecated alias of `slice_group`
    * `moe_group` stands for the group of process that performs expert
    parallelism. The default value `None` means all processes. See the
    parallelism document for more details of the groups.
    * `top_k` stands for the number of experts each token is going to.
    * `gate` is a gate class which can found in `fmoe.gates`.
    * `expert` can be specified as a module class, it is used to generate
    `num_expert` expert modules.
    * `gate_bias` is only valid for naive_gate and its subclasses, it means
    whether to add bias to the gate module.
    """

    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        world_size=1,
        mp_group=None,  # being deprecated
        slice_group=None,
        moe_group=None,
        top_k=2,
        gate=NaiveGate,
        expert=None,
        gate_hook=None,
        mask=None,
        mask_dict=None,
        gate_bias=True,
    ):
        super().__init__()
        self.num_expert = num_expert
        self.d_model = d_model
        self.world_size = world_size
        print('=============================== world_size: ', self.world_size, '==================================')
        self.expert = expert

        # balancemoe
        self.num_expert_in_devices = []
        for i in range(self.world_size):
            self.num_expert_in_devices.append(self.num_expert)

        self.experts_to_devices = [[0 for _ in range(self.num_expert)] for _ in range(self.world_size)]
        for i in range(self.world_size):
            for j in range(self.num_expert):
                self.experts_to_devices[i][j] = i * self.num_expert + j
        print('======================== num_expert_in_devices: ', self.num_expert_in_devices, '===================')
        print('======================== experts_to_devices: ', self.experts_to_devices, '=========================')
        ############

        self.slice_group = slice_group
        if mp_group is not None:
            print("[Warning] mp_group is being deprecated")
            self.slice_group = mp_group
        if self.slice_group is None:
            self.slice_size = 1
            self.slice_rank = 0
        else:
            self.slice_size = self.slice_group.size()
            self.slice_rank = self.slice_group.rank()

        self.top_k = top_k
        if type(expert) is list:
            self.experts = nn.ModuleList([e(d_model) for e in expert])
            self.experts_fused = False
            self.num_expert = num_expert = len(expert)
        elif expert is not None:
            self.experts = nn.ModuleList([expert(d_model) for _ in range(num_expert)])
            self.experts_fused = False
        else:
            self.experts_fused = True

        if issubclass(gate, NaiveGate):
            self.gate = gate(d_model, num_expert, world_size, top_k, gate_bias=gate_bias)
        else:
            self.gate = gate(d_model, num_expert, world_size, top_k)
        self.gate_hook = gate_hook
        self.mask = mask
        self.mask_dict = mask_dict
        self.moe_group = moe_group

    def expert_fn(self, inp, fwd_expert_count, num_expert, old_experts_to_devices, old_num_expert_in_devices,
                  new_experts_to_devices, new_num_expert_in_devices, expert_transfer_matrix):
        r"""
        The default expert function which either calls the experts as a whole
        or as separate experts.
        """
        if inp.device == torch.device('cuda:0'):
            print('old_experts_to_devices:', old_experts_to_devices)
            print('new_experts_to_devices:', new_experts_to_devices)
            # print('new_num_expert_in_devices:', new_num_expert_in_devices)

        def writeFile_add(filename, data):
            file_handle = open(filename, mode='a')
            file_handle.write(data)
            file_handle.close()

        def repackage(_experts_to_devices, _num_expert_in_devices):
            _new_experts_to_devices = []
            index = 0
            for num in _num_expert_in_devices:
                _new_experts_to_devices.append(_experts_to_devices[index:index + num])
                index += num
            return _new_experts_to_devices

        def update_experts(old_experts, added_experts, removed_experts):
            fake_new_experts = []
            for expert in old_experts:
                if not expert in removed_experts:
                    fake_new_experts.append(expert)

            fake_new_experts.extend(added_experts)
            return fake_new_experts

        if self.experts_fused:
            return self.experts(inp, fwd_expert_count)
        if isinstance(fwd_expert_count, torch.Tensor):
            fwd_expert_count_cpu = fwd_expert_count.cpu().numpy()
        outputs = []
        base_idx = 0

        for i, expert in enumerate(self.experts):
            for name, param in expert.named_parameters():
                device_str = str(param.device)
                device_index = int(device_str.split(':')[1])
                break

        new_experts_to_devices = [element for element in new_experts_to_devices if element != -1]
        new_experts_to_devices = repackage(new_experts_to_devices, new_num_expert_in_devices)

        expert_sample = torch.cat([x.view(-1) for x in self.experts[0].parameters()])
        comm_req = []

        # balancemoe expert transfer mechanism ##################################################################
        expert_comm_req = []
        def array_diff(old, new):
            old_set = set(old)
            new_set = set(new)

            added_elements = list(new_set - old_set)
            removed_elements = list(old_set - new_set)

            return added_elements, removed_elements

        old_experts = old_experts_to_devices[device_index]
        new_experts = new_experts_to_devices[device_index]
        added_experts, removed_experts = array_diff(old_experts, new_experts)

        old_len_experts = len(new_experts)
        # if inp.device == torch.device('cuda:0'):
        #     print('len(self.experts):', len(self.experts))

        if len(added_experts) + len(removed_experts) > 0:
            for i in range(len(expert_transfer_matrix)):
                if device_index == expert_transfer_matrix[i][0]:
                    pair_device_index = expert_transfer_matrix[i][1]
                    break
                if device_index == expert_transfer_matrix[i][1]:
                    pair_device_index = expert_transfer_matrix[i][0]
                    break
            if device_index < pair_device_index:
                for i in removed_experts:
                    expert_pos = old_experts_to_devices[device_index].index(i)
                    expert_para = torch.cat([x.view(-1) for x in self.experts[expert_pos].parameters()])
                    expert_comm_req.append(dist.isend(expert_para, dst=pair_device_index))
                for i in added_experts:
                    expert_para = torch.empty_like(expert_sample)
                    expert_comm_req.append(dist.irecv(expert_para, src=pair_device_index))
                    new_expert = self.expert(self.d_model).to(inp.device)
                    index = 0
                    for param in new_expert.parameters():
                        param_length = torch.prod(torch.tensor(param.shape))
                        param.data.copy_(expert_para[index:index + param_length].view(param.shape))
                        index += param_length
                    # Append the new expert to the experts list
                    self.experts.append(new_expert)
                # for i in removed_experts:
                #     expert_pos = old_experts_to_devices[device_index].index(i)
                #     del self.experts[expert_pos]
            else:
                for i in added_experts:
                    expert_para = torch.empty_like(expert_sample)
                    expert_comm_req.append(dist.irecv(expert_para, src=pair_device_index))
                    new_expert = self.expert(self.d_model).to(inp.device)
                    index = 0
                    for param in new_expert.parameters():
                        param_length = torch.prod(torch.tensor(param.shape))
                        param.data.copy_(expert_para[index:index + param_length].view(param.shape))
                        index += param_length
                    # Append the new expert to the experts list
                    self.experts.append(new_expert)
                for i in removed_experts:
                    expert_pos = old_experts_to_devices[device_index].index(i)
                    expert_para = torch.cat([x.view(-1) for x in self.experts[expert_pos].parameters()])
                    expert_comm_req.append(dist.isend(expert_para, dst=pair_device_index))
                # for i in removed_experts:
                #     expert_pos = old_experts_to_devices[device_index].index(i)
                #     del self.experts[expert_pos]

        # if inp.device == torch.device('cuda:0'):
        #     print('len(self.experts):', len(self.experts))

        # overlap non-transfer experts' computing and transfer experts' communication
        use_comp_comm_parallel = False

        if use_comp_comm_parallel:
            fake_new_experts = update_experts(old_experts, added_experts, removed_experts)
            for i in range(len(old_experts) - len(removed_experts)):
                idx = new_experts.index(fake_new_experts[i])
                batch_size = fwd_expert_count_cpu[idx]
                inp_slice = inp[sum(fwd_expert_count_cpu[:idx]): sum(fwd_expert_count_cpu[:idx]) + batch_size]
                outputs.append(self.experts[i](inp_slice, torch.tensor([fwd_expert_count[idx]])))

            for i in range(len(expert_comm_req)):
                expert_comm_req[i].wait()

            for i in range(len(added_experts)):
                idx = new_experts.index(fake_new_experts[i + len(old_experts) - len(removed_experts)])
                batch_size = fwd_expert_count_cpu[idx]
                inp_slice = inp[sum(fwd_expert_count_cpu[:idx]): sum(fwd_expert_count_cpu[:idx]) + batch_size]
                outputs.append(self.experts[i + len(old_experts) - len(removed_experts)](inp_slice, torch.tensor([fwd_expert_count[idx]])))

            # reorder outputs and experts
            new_outputs_order = []
            new_experts_order = nn.ModuleList()
            for new_index in new_experts:
                old_index = fake_new_experts.index(new_index)
                new_outputs_order.append(outputs[old_index])
                new_experts_order.append(self.experts[old_index])

            outputs = new_outputs_order
            self.experts = new_experts_order

        else:
            for i in range(len(expert_comm_req)):
                expert_comm_req[i].wait()

            # fake_new_experts = update_experts(old_experts, added_experts, removed_experts)
            #
            # new_experts_order = nn.ModuleList()
            #
            # for new_index in new_experts:
            #     old_index = fake_new_experts.index(new_index)
            #     new_experts_order.append(self.experts[old_index])
            #
            # self.experts = new_experts_order

            for i in range(num_expert):
                batch_size = fwd_expert_count_cpu[i]
                inp_slice = inp[base_idx : base_idx + batch_size]
                outputs.append(self.experts[i](inp_slice, torch.tensor([fwd_expert_count[i]])))
                base_idx += batch_size
        ##############################################

        for i in removed_experts:
            expert_pos = old_experts_to_devices[device_index].index(i)
            if expert_pos < len(self.experts):
                del self.experts[expert_pos]

        # if len(self.experts) > old_len_experts:
        #     for i in range(len(self.experts) - old_len_experts):
        #         print('================================================== ', inp.device, len(self.experts), i)
        #         del self.experts[0]

        # res = 'self.experts: {}\n old_experts_to_devices: {}\n old_num_expert_in_devices: {}\n new_experts_to_devices: {}\n new_num_expert_in_devices: {}\n'.format(
        #     self.experts, old_experts_to_devices, old_num_expert_in_devices,
        #     new_experts_to_devices, new_num_expert_in_devices)
        # filename = '/home/jianhongbai/gyq/BalanceMOE/balancemoe/examples/transformer-xl/LM-TFM-enwik8/expert_fn.txt'
        # writeFile_add(filename, res)

        return torch.cat(outputs, dim=0)

    def expert_fn_single(self, inp, fwd_expert_count, idx):
        r"""
        forward single expert for smart scheduling.
        """
        assert not self.experts_fused, "should not use fused experts"
        output = self.experts[idx](inp, fwd_expert_count)
        return output

    def mark_parallel_comm(self, expert_dp_comm="none"):
        r"""
        Automatically mark the data parallel comms of the parameters within the
        module. This can be typically called at the end of the __init__ function
        in child classes.
        """
        if self.experts is not None:
            comm = expert_dp_comm
            if isinstance(self.experts, list):
                for e in self.experts:
                    mark_module_parallel_comm(e, comm)
            else:
                mark_module_parallel_comm(self.experts, comm)
        mark_module_parallel_comm(self.gate, "gate")

    def forward(self, moe_inp):
        r"""
        The FMoE module first computes gate output, and then conduct MoE forward
        according to the gate.  The score of the selected gate given by the
        expert is multiplied to the experts' output tensors as a weight.
        """

        moe_inp_batch_size = tree.flatten(
            tree.map_structure(lambda tensor: tensor.shape[0], moe_inp)
        )
        assert all(
            [batch_size == moe_inp_batch_size[0] for batch_size in moe_inp_batch_size]
        ), "MoE inputs must have the same batch size"

        if self.world_size > 1:

            def ensure_comm_func(tensor):
                ensure_comm(tensor, self.moe_group)

            tree.map_structure(ensure_comm_func, moe_inp)
        if self.slice_size > 1:

            def slice_func(tensor):
                return Slice.apply(
                    tensor, self.slice_rank, self.slice_size, self.slice_group
                )

            moe_inp = tree.map_structure(slice_func, moe_inp)

        gate_top_k_idx, gate_score = self.gate(moe_inp)

        if self.gate_hook is not None:
            self.gate_hook(gate_top_k_idx, gate_score, None)

        # delete masked tensors
        if self.mask is not None and self.mask_dict is not None:
            # TODO: to fix
            def delete_mask_func(tensor):
                # to: (BxL') x d_model
                tensor = tensor[mask == 0, :]
                return tensor

            mask = self.mask.view(-1)
            moe_inp = tree.map_structure(delete_mask_func, moe_inp)
            gate_top_k_idx = gate_top_k_idx[mask == 0, :]

        fwd, new_experts_to_devices, new_num_expert_in_devices = _fmoe_general_global_forward(
            moe_inp, gate_top_k_idx, self.expert_fn_single if fmoe_faster_schedule else self.expert_fn,
            self.num_expert_in_devices, self.experts_to_devices, self.world_size,
            experts=self.experts
        )

        # balancemoe
        self.num_expert_in_devices = new_num_expert_in_devices.tolist()
        self.experts_to_devices = [element for element in new_experts_to_devices.tolist() if element != -1]

        def repackage(_experts_to_devices, _num_expert_in_devices):
            _new_experts_to_devices = []
            index = 0
            for num in _num_expert_in_devices:
                _new_experts_to_devices.append(_experts_to_devices[index:index + num])
                index += num
            return _new_experts_to_devices

        self.experts_to_devices = repackage(self.experts_to_devices, self.num_expert_in_devices)

        # print('self.experts_to_devices:', self.experts_to_devices)
        ############

        # recover deleted tensors
        if self.mask is not None and self.mask_dict is not None:

            def recover_func(tensor):
                # to: (BxL') x top_k x dim
                dim = tensor.shape[-1]
                tensor = tensor.view(-1, self.top_k, dim)
                # to: (BxL) x top_k x d_model
                x = torch.zeros(
                    mask.shape[0],
                    self.top_k,
                    dim,
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
                # recover
                x[mask == 0] = tensor
                for k, v in self.mask_dict.items():
                    x[mask == k] = v
                return x

            moe_outp = tree.map_structure(recover_func, fwd)
        else:

            def view_func(tensor):
                dim = tensor.shape[-1]
                tensor = tensor.view(-1, self.top_k, dim)
                return tensor

            moe_outp = tree.map_structure(view_func, fwd)

        gate_score = gate_score.view(-1, 1, self.top_k)

        def bmm_func(tensor):
            dim = tensor.shape[-1]
            tensor = torch.bmm(gate_score, tensor).reshape(-1, dim)
            return tensor

        moe_outp = tree.map_structure(bmm_func, moe_outp)

        if self.slice_size > 1:

            def all_gather_func(tensor):
                return AllGather.apply(
                    tensor, self.slice_rank, self.slice_size, self.slice_group
                )

            moe_outp = tree.map_structure(all_gather_func, moe_outp)

        moe_outp_batch_size = tree.flatten(
            tree.map_structure(lambda tensor: tensor.shape[0], moe_outp)
        )
        assert all(
            [batch_size == moe_outp_batch_size[0] for batch_size in moe_outp_batch_size]
        ), "MoE outputs must have the same batch size"
        return moe_outp
