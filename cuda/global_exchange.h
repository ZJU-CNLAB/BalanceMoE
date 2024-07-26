#include "stream_manager.h"
#include <iostream>
#ifdef FMOE_USE_NCCL

void fmoe_cuda_expert_exchange_impl(
        const long* local_expert_count,
        long* global_expert_count,
        int n_expert, int world_size,
        CudaStreamManager* smgr);

void fmoe_cuda_expert_exchange_impl2(
        const long* local_expert_count,
        long* global_expert_count,
        long* new_experts_to_devices,
        int max_len, int n_expert, int world_size,
        CudaStreamManager* smgr);


template<typename scalar_t>
void fmoe_cuda_global_scatter_impl(
    const scalar_t* local_input_buf,
    const long* local_expert_count,
    const long* global_expert_count,
    scalar_t* input_buf,
    size_t in_feat, size_t n_expert, size_t world_size,
    size_t num_expert,
    long* new_experts_to_devices,
    size_t max_len,
    CudaStreamManager* smgr) {
    // assert world_size > 1
    int recv_ptr = 0;
    /* TODO: may save for backward */
    long*expert_ptr = new long[n_expert * world_size];
    expert_ptr[0] = 0;
    for (size_t i = 1; i < n_expert * world_size; ++i) {
        expert_ptr[i] = expert_ptr[i - 1] + local_expert_count[i - 1];
    }

//    for (size_t i = 0; i < n_expert; ++i) {
//        NCCL_SAFE_CALL(ncclGroupStart());
//        for (size_t j = 0; j < world_size; ++j) {
//            int idx = i + j * n_expert;
//            if (local_expert_count[idx]) {
//                NCCL_SAFE_CALL(ncclSend(
//                        local_input_buf + expert_ptr[idx] * in_feat,
//                        local_expert_count[idx] * in_feat * sizeof(scalar_t),
//                        ncclChar,
//                        j,
//                        smgr->ncclcomm,
//                        smgr->torchStream()));
//            }
//            if (global_expert_count[idx]) {
//                NCCL_SAFE_CALL(ncclRecv(
//                        input_buf + recv_ptr * in_feat,
//                        global_expert_count[idx] * in_feat * sizeof(scalar_t),
//                        ncclChar,
//                        j,
//                        smgr->ncclcomm,
//                        smgr->torchStream()));
//                recv_ptr += global_expert_count[idx];
//            }
//        }
//        NCCL_SAFE_CALL(ncclGroupEnd());
//    }

    int acc_idx = 0;
    int total_acc_idx = 0;
    int total_acc_local_expert_count = 0;
    int total_acc_global_expert_count = 0;

    for (size_t i = 0; i < world_size; ++i) {
        NCCL_SAFE_CALL(ncclGroupStart());
        for (size_t j = 0; j < max_len; ++j) {
            if(static_cast<int>(new_experts_to_devices[i*max_len + j]) != -1){
                total_acc_local_expert_count = total_acc_local_expert_count + local_expert_count[total_acc_idx];
                acc_idx = acc_idx + 1;
                total_acc_idx = total_acc_idx + 1;
            }
        }
        for (size_t k = 0; k < num_expert; ++k) {
            total_acc_global_expert_count = total_acc_global_expert_count + global_expert_count[i*num_expert + k];
        }
        NCCL_SAFE_CALL(ncclSend(
                local_input_buf + expert_ptr[total_acc_idx - acc_idx] * in_feat,
                total_acc_local_expert_count * in_feat * sizeof(scalar_t),
                ncclChar,
                i,
                smgr->ncclcomm,
                smgr->torchStream()));
        NCCL_SAFE_CALL(ncclRecv(
                input_buf + recv_ptr * in_feat,
                total_acc_global_expert_count * in_feat * sizeof(scalar_t),
                ncclChar,
                i,
                smgr->ncclcomm,
                smgr->torchStream()));
        recv_ptr += total_acc_global_expert_count;
        total_acc_local_expert_count = 0;
        acc_idx = 0;
        total_acc_global_expert_count = 0;

        NCCL_SAFE_CALL(ncclGroupEnd());
    }

    delete [] expert_ptr;
}

template<typename scalar_t>
void fmoe_cuda_global_gather_impl(
    const scalar_t* output_buf,
    const long* local_expert_count,
    const long* global_expert_count,
    scalar_t* local_output_buf,
    size_t out_feat, size_t n_expert, size_t world_size,
    size_t num_expert,
    long* new_experts_to_devices,
    size_t max_len,
    CudaStreamManager* smgr) {
    long send_ptr = 0;
    /* TODO: may save for backward */
    long *expert_ptr = new long[n_expert * world_size];
    expert_ptr[0] = 0;
    for (size_t i = 1; i < n_expert * world_size; ++i) {
        expert_ptr[i] = expert_ptr[i - 1] + local_expert_count[i - 1];
    }

//    for (size_t i = 0; i < n_expert; ++i) {
//        NCCL_SAFE_CALL(ncclGroupStart());
//        for (size_t j = 0; j < world_size; ++j) {
//            int idx = i + j * n_expert;
//            if (global_expert_count[idx]) {
//                NCCL_SAFE_CALL(ncclSend(
//                        output_buf + send_ptr * out_feat,
//                        global_expert_count[idx] * out_feat * sizeof(scalar_t),
//                        ncclChar,
//                        j,
//                        smgr->ncclcomm,
//                        smgr->torchStream()));
//                send_ptr += global_expert_count[idx];
//            }
//            if (local_expert_count[idx]) {
//                NCCL_SAFE_CALL(ncclRecv(
//                        local_output_buf + expert_ptr[idx] * out_feat,
//                        local_expert_count[idx] * out_feat * sizeof(scalar_t),
//                        ncclChar,
//                        j,
//                        smgr->ncclcomm,
//                        smgr->torchStream()));
//            }
//        }
//        NCCL_SAFE_CALL(ncclGroupEnd());
//    }

    int acc_idx = 0;
    int total_acc_idx = 0;
    int total_acc_local_expert_count = 0;
    int total_acc_global_expert_count = 0;

    for (size_t i = 0; i < world_size; ++i) {
        NCCL_SAFE_CALL(ncclGroupStart());
        for (size_t j = 0; j < max_len; ++j) {
            if(static_cast<int>(new_experts_to_devices[i*max_len + j]) != -1){
                total_acc_local_expert_count = total_acc_local_expert_count + local_expert_count[total_acc_idx];
                acc_idx = acc_idx + 1;
                total_acc_idx = total_acc_idx + 1;
            }
        }
        for (size_t k = 0; k < num_expert; ++k) {
            total_acc_global_expert_count = total_acc_global_expert_count + global_expert_count[i*num_expert + k];
        }
        NCCL_SAFE_CALL(ncclSend(
                output_buf + send_ptr * out_feat,
                total_acc_global_expert_count * out_feat * sizeof(scalar_t),
                ncclChar,
                i,
                smgr->ncclcomm,
                smgr->torchStream()));
        send_ptr += total_acc_global_expert_count;
        NCCL_SAFE_CALL(ncclRecv(
                local_output_buf + expert_ptr[total_acc_idx - acc_idx] * out_feat,
                total_acc_local_expert_count * out_feat * sizeof(scalar_t),
                ncclChar,
                i,
                smgr->ncclcomm,
                smgr->torchStream()));
        total_acc_local_expert_count = 0;
        acc_idx = 0;
        total_acc_global_expert_count = 0;

        NCCL_SAFE_CALL(ncclGroupEnd());
    }

    delete [] expert_ptr;
}


#endif  // FMOE_USE_NCCL
