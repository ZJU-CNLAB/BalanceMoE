#include "global_exchange.h"
#include "utils/fmoe_utils.h"
#include <torch/extension.h>

#ifdef FMOE_USE_NCCL
#include <nccl.h>

void fmoe_cuda_expert_exchange_impl(
        const long* local_expert_count,
        long* global_expert_count,
        int n_expert, int world_size,
        CudaStreamManager* smgr) {
    NCCL_SAFE_CALL(ncclGroupStart());

    for (int i = 0; i < world_size; ++i) {
        NCCL_SAFE_CALL(ncclSend(
                local_expert_count + n_expert * i,
                n_expert,
                ncclInt64,
                i,
                smgr->ncclcomm,
                smgr->torchStream()));
        NCCL_SAFE_CALL(ncclRecv(
                global_expert_count + n_expert * i,
                n_expert,
                ncclInt64,
                i,
                smgr->ncclcomm,
                smgr->torchStream()));
    }
    NCCL_SAFE_CALL(ncclGroupEnd());
}

void fmoe_cuda_expert_exchange_impl2(
        const long* local_expert_count,
        long* global_expert_count,
        long* new_experts_to_devices,
        int max_len, int n_expert, int world_size,
        CudaStreamManager* smgr) {
    NCCL_SAFE_CALL(ncclGroupStart());

    for (int i = 0; i < world_size; ++i) {
        for (int j = 0; j < n_expert; ++j) {
            NCCL_SAFE_CALL(ncclSend(
                    local_expert_count + *(new_experts_to_devices + i * max_len + j),
                    1,
                    ncclInt64,
                    i,
                    smgr->ncclcomm,
                    smgr->torchStream()));
            NCCL_SAFE_CALL(ncclRecv(
                    global_expert_count + *(new_experts_to_devices + i * max_len + j),
                    1,
                    ncclInt64,
                    i,
                    smgr->ncclcomm,
                    smgr->torchStream()));
        }
    }

    NCCL_SAFE_CALL(ncclGroupEnd());
}

torch::Tensor _expert_exchange(
        torch::Tensor local_expert_count,
        long n_expert,
        torch::Tensor new_experts_to_devices,
        long max_len,
        torch::Tensor reorder_local_expert_count,
        torch::Tensor global_expert_count_zeros,
        long n_workers) {
    auto global_expert_count = torch::empty_like(global_expert_count_zeros);
    auto smgr = getCudaStreamManager(local_expert_count.device().index());

//    std::cout << "call fmoe_cuda_expert_exchange_impl2" << std::endl;

    const long* local_expert_count_ptr = local_expert_count.data_ptr<long>();
    const long* reorder_local_expert_count_ptr = reorder_local_expert_count.data_ptr<long>();
    long* global_expert_count_ptr = global_expert_count.data_ptr<long>();
    int n_expert_int = static_cast<int>(n_expert);
    int max_len_int = static_cast<int>(max_len);
    int world_size = static_cast<int>(n_workers);

    int acc_expert = 0;
    int total_acc_expert = 0;

    NCCL_SAFE_CALL(ncclGroupStart());
    for (int i = 0; i < world_size; ++i) {

        for (int j = 0; j < max_len_int; ++j) {
            if(new_experts_to_devices[i*max_len_int + j].item<int>() != -1){
                acc_expert = acc_expert + 1;
                total_acc_expert = total_acc_expert + 1;
            }
        }

        NCCL_SAFE_CALL(ncclSend(
            reorder_local_expert_count_ptr + total_acc_expert - acc_expert,
            acc_expert,
            ncclInt64,
            i,
            smgr->ncclcomm,
            smgr->torchStream()));

        NCCL_SAFE_CALL(ncclRecv(
            global_expert_count_ptr + n_expert_int * i,
            n_expert_int,
            ncclInt64,
            i,
            smgr->ncclcomm,
            smgr->torchStream()));

        acc_expert = 0;
    }
    NCCL_SAFE_CALL(ncclGroupEnd());

//    std::cout << "call fmoe_cuda_expert_exchange_impl2 over" << std::endl;
    return global_expert_count;
}

torch::Tensor _global_scatter(
        torch::Tensor input_buf,
        torch::Tensor local_expert_count,
        torch::Tensor global_expert_count,
        long batch_size, long n_workers,
        long num_expert,
        torch::Tensor new_experts_to_devices, long max_len) {
    CHECK_INPUT(input_buf);

    auto n_expert = local_expert_count.size(0) / n_workers;
    auto in_feat = input_buf.size(1);
    auto global_input_buf = input_buf.new_empty({batch_size, in_feat});
    auto smgr = getCudaStreamManager(input_buf.device().index());

    std::cout << "call fmoe_cuda_global_scatter" << std::endl;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
            input_buf.scalar_type(), "fmoe_cuda_global_scatter", ([&] {
        fmoe_cuda_global_scatter_impl<scalar_t>(
            input_buf.data_ptr<scalar_t>(),
            local_expert_count.data_ptr<long>(),
            global_expert_count.data_ptr<long>(),
            global_input_buf.data_ptr<scalar_t>(),
            in_feat, n_expert, n_workers,
            num_expert,
            new_experts_to_devices.data_ptr<long>(),
            max_len,
            smgr
        );
    }));

    std::cout << "call fmoe_cuda_global_scatter over" << std::endl;

    return global_input_buf;
}

torch::Tensor _global_gather(
        torch::Tensor output_buf,
        torch::Tensor local_expert_count,
        torch::Tensor global_expert_count,
        long batch_size, long n_workers,
        long num_expert,
        torch::Tensor new_experts_to_devices, long max_len) {
    CHECK_INPUT(output_buf);

    auto n_expert = local_expert_count.size(0) / n_workers;
    auto out_feat = output_buf.size(1);
    auto local_output_buf = output_buf.new_empty({batch_size, out_feat});
    auto smgr = getCudaStreamManager(output_buf.device().index());

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
            output_buf.scalar_type(), "fmoe_cuda_global_gather", ([&] {
        fmoe_cuda_global_gather_impl<scalar_t>(
            output_buf.data_ptr<scalar_t>(),
            local_expert_count.data_ptr<long>(),
            global_expert_count.data_ptr<long>(),
            local_output_buf.data_ptr<scalar_t>(),
            out_feat, n_expert, n_workers,
            num_expert,
            new_experts_to_devices.data_ptr<long>(),
            max_len,
            smgr
        );
    }));
    return local_output_buf;
}

#if defined(TORCH_VERSION_MAJOR) && (TORCH_VERSION_MAJOR > 1 || \
        (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 13))
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#else
#include <c10d/ProcessGroupNCCL.hpp>
#endif

class HackNCCLGroup: public c10d::ProcessGroupNCCL {
public:
    ncclComm_t getcomm(at::Device dev) {
        ncclUniqueId ncclID;
        int rank = getRank();
        if (rank == 0) {
            ncclGetUniqueId(&ncclID);
        }
#if defined(TORCH_VERSION_MAJOR) && (TORCH_VERSION_MAJOR > 1 || \
        (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 12))
        broadcastUniqueNCCLID(&ncclID,
                false,
                "fastmoe_nccl_comm",
                rank);
#elif defined(TORCH_VERSION_MAJOR) && (TORCH_VERSION_MAJOR > 1 || \
        (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 8))
        broadcastUniqueNCCLID(&ncclID,
                c10d::OpType::SEND,
                "fastmoe_nccl_comm",
                rank);
#else
        broadcastUniqueNCCLID(&ncclID);
#endif
        ncclComm_t comm;
        NCCL_SAFE_CALL(ncclCommInitRank(&comm, getSize(), ncclID, rank));
        return comm;
    }
};

#if defined(TORCH_VERSION_MAJOR) && (TORCH_VERSION_MAJOR >= 2)
void _ensure_nccl(c10d::ProcessGroup& p, torch::Tensor t) {
#else
void _ensure_nccl(c10d::ProcessGroupNCCL& p, torch::Tensor t) {
#endif  // TORCH_VERSION
    auto smgr = getCudaStreamManager(t.device().index());
    if (smgr->ncclgood) {
        return;
    }
#if defined(TORCH_VERSION_MAJOR) && (TORCH_VERSION_MAJOR >= 2)
    HackNCCLGroup* h = (HackNCCLGroup*)(void*)
        (p.getBackend(c10d::ProcessGroup::NCCL).get());
#else
    HackNCCLGroup* h = (HackNCCLGroup*)(void*)&p;
#endif  // TORCH_VERSION
    smgr->ncclcomm = h->getcomm(t.device());
    if (smgr->ncclcomm != 0) {
        smgr->ncclgood = 1;
    } else {
        std::cerr << "Nccl initialization failed\n";
    }
}

#endif  // FMOE_USE_NCCL
