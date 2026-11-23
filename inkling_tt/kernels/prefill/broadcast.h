#pragma once

#include "api/dataflow/dataflow_api.h"

// Cores in a contiguous row share the same activation CB address. A receiver
// announces each reserved page before the sender multicasts into it.
class PrefillBroadcast {
    uint32_t first, last, row;
    uint32_t ready_addr = get_semaphore(0);
    uint32_t valid_addr = get_semaphore(1);
    volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
    volatile tt_l1_ptr uint32_t* valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(valid_addr);

public:
    bool sender;

    explicit PrefillBroadcast(uint32_t arg) :
        first(get_arg_val<uint32_t>(arg)), last(get_arg_val<uint32_t>(arg + 1)),
        row(get_arg_val<uint32_t>(arg + 2)), sender(get_arg_val<uint32_t>(arg + 3)) {}

    void distribute(uint32_t addr, uint32_t bytes) {
        if (sender) {
            const uint32_t receivers = last - first;
            if (receivers != 0) {
                const uint32_t start_x = noc_index == 0 ? first + 1 : last;
                const uint32_t end_x = noc_index == 0 ? last : first + 1;
                noc_semaphore_wait(ready, receivers);
                noc_semaphore_set(ready, 0);
                noc_async_write_multicast(addr, get_noc_multicast_addr(start_x, row, end_x, row, addr),
                                          bytes, receivers);
                noc_async_write_barrier();
                noc_semaphore_set_multicast(get_semaphore(2),
                    get_noc_multicast_addr(start_x, row, end_x, row, valid_addr), receivers);
            }
        } else {
            noc_semaphore_set(valid, 0);
            noc_semaphore_inc(get_noc_addr(first, row, ready_addr), 1);
            noc_semaphore_wait(valid, 1);
        }
    }

    void finish() {
        if (!sender) {
            noc_semaphore_set(valid, 0);
        }
    }
};
