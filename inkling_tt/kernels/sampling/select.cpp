#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t tiles = get_compile_time_arg_val(0);
constexpr uint32_t vocab = get_compile_time_arg_val(1);
constexpr uint32_t workers = get_compile_time_arg_val(2);
constexpr uint32_t stripe = tiles * 1024;
constexpr uint32_t digit_count = 16;
constexpr uint32_t bin_bytes = digit_count * 4;
constexpr uint32_t logits_width = get_compile_time_arg_val(3);
constexpr uint32_t small_k = get_compile_time_arg_val(4);

uint32_t bits(float x) {
    uint32_t result;
    __builtin_memcpy(&result, &x, 4);
    return result;
}

float floating(uint32_t bits) {
    float result;
    __builtin_memcpy(&result, &bits, 4);
    return result;
}

uint32_t order(uint32_t bits) {
    return (bits & 0x7fffffffu) == 0 ? 0x80000000u : (bits & 0x80000000u) ? ~bits : bits ^ 0x80000000u;
}

uint32_t unorder(uint32_t key) {
    return (key & 0x80000000u) ? key ^ 0x80000000u : ~key;
}

auto words(uint32_t address) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(address);
}

auto floats(uint32_t address) {
    return reinterpret_cast<volatile tt_l1_ptr float*>(address);
}
struct Candidate {
    uint32_t token;
    uint32_t score;
    float weight;
    uint32_t pad;
};

bool less(const Candidate& left, const Candidate& right) {
    return left.score == right.score ? left.token > right.token : left.score < right.score;
}

void sift(Candidate* heap, uint32_t size, uint32_t pos) {
    const Candidate value = heap[pos];
    for (uint32_t child = 2 * pos + 1; child < size; child = 2 * pos + 1) {
        if (child + 1 < size && less(heap[child + 1], heap[child])) {
            ++child;
        }
        if (!less(heap[child], value)) {
            break;
        }
        heap[pos] = heap[child];
        pos = child;
    }
    heap[pos] = value;
}

}  // namespace

void kernel_main() {
    constexpr auto logits_args = TensorAccessorArgs<5>();
    constexpr auto counts_args = TensorAccessorArgs<logits_args.next_compile_time_args_offset()>();
    constexpr auto cfg_args = TensorAccessorArgs<counts_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<cfg_args.next_compile_time_args_offset()>();
    const auto logits = TensorAccessor(logits_args, get_common_arg_val<uint32_t>(0), logits_width * 2);
    const auto counts = TensorAccessor(counts_args, get_common_arg_val<uint32_t>(1), stripe * 4);
    const auto config = TensorAccessor(cfg_args, get_common_arg_val<uint32_t>(2), 32);
    const auto result = TensorAccessor(out_args, get_common_arg_val<uint32_t>(3), 32);
    const uint32_t me = get_arg_val<uint32_t>(0);
    const uint32_t first = me * stripe;
    const uint32_t valid = vocab - first < stripe ? vocab - first : stripe;
    const auto peer = [](uint32_t worker, uint32_t address) {
        const uint32_t coords = get_common_arg_val<uint32_t>(4 + worker);
        return get_noc_addr(coords & 0xffff, coords >> 16, address);
    };

    const uint32_t scratch = get_write_ptr(4);
    const uint32_t receive = scratch + stripe * 4;
    const uint32_t send = receive + bin_bytes * workers;
    const uint32_t response = send + bin_bytes;
    const uint32_t config_l1 = response + 64;
    const uint32_t arrivals = get_semaphore(0);
    const uint32_t ready = get_semaphore(1);
    uint32_t phase = 0;
    auto exchange = [&](auto&& reduce) {
        ++phase;
        noc_async_write<bin_bytes>(send, peer(0, receive + bin_bytes * me), bin_bytes);
        noc_async_write_barrier();
        noc_semaphore_inc(peer(0, arrivals), 1);
        noc_async_atomic_barrier();
        if (me == 0) {
            noc_semaphore_wait(words(arrivals), workers * phase);
            invalidate_l1_cache();
            reduce();
            for (uint32_t worker = 1; worker < workers; ++worker) {
                noc_async_write<32>(response, peer(worker, response), 32);
            }
            noc_async_write_barrier();
            for (uint32_t worker = 1; worker < workers; ++worker) {
                noc_semaphore_inc(peer(worker, ready), 1);
            }
            noc_async_atomic_barrier();
        } else {
            noc_semaphore_wait(words(ready), phase);
            invalidate_l1_cache();
        }
    };

    noc_async_read<32>(config.get_noc_addr(0), config_l1, 32);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const auto cfg = words(config_l1);
    const uint32_t last = cfg[1];
    const float presence = floating(cfg[2]);
    const float frequency = floating(cfg[3]);
    const bool penalties = presence != 0.0f || frequency != 0.0f;
    const uint32_t top_k = cfg[4];
    const float top_p = floating(cfg[5]);
    const float draw = floating(cfg[6]);

    cb_reserve_back(2, 1);
    words(get_write_ptr(2))[0] = cfg[0];
    cb_push_back(2, 1);
    cb_reserve_back(0, tiles);
    cb_reserve_back(1, tiles);
    noc_async_read(logits.get_noc_addr(0, first * 2), get_write_ptr(0), stripe * 2);
    if (penalties) {
        noc_async_read(counts.get_noc_addr(me), scratch, stripe * 4);
    }
    noc_async_read_barrier();
    if (penalties) {
        invalidate_l1_cache();
        if (last >= first && last < first + valid) {
            const uint32_t i = last - first;
            words(scratch)[i] += 1;
            const uint32_t offset = (i * 4) & ~63u;
            noc_async_write<64>(scratch + offset, counts.get_noc_addr(me, offset), 64);
            noc_async_write_barrier();
        }
    }
    auto delta = floats(get_write_ptr(1));
    for (uint32_t i = 0; i < stripe; ++i) {
        const uint32_t count = penalties ? words(scratch)[i] : 0;
        delta[i] = count ? presence + frequency * static_cast<float>(count) : 0.0f;
    }
    cb_push_back(1, tiles);
    cb_push_back(0, tiles);
    cb_wait_front(16, tiles);
    invalidate_l1_cache();
    const auto values = words(get_read_ptr(16));
    uint32_t maximum = 0;
    for (uint32_t i = 0; i < valid; ++i) {
        const uint32_t key = order(values[i]);
        if (key > maximum) {
            maximum = key;
        }
    }
    words(send)[0] = maximum;
    exchange([&]() {
        uint32_t max_key = 0;
        for (uint32_t worker = 0; worker < workers; ++worker) {
            const uint32_t key = words(receive + worker * bin_bytes)[0];
            if (key > max_key) {
                max_key = key;
            }
        }
        words(response)[0] = unorder(max_key);
    });
    cb_reserve_back(3, 1);
    const uint32_t global_max_bits = words(response)[0];
    words(get_write_ptr(3))[0] = global_max_bits;
    cb_push_back(3, 1);
    cb_wait_front(17, tiles);
    invalidate_l1_cache();
    const auto weights = floats(get_read_ptr(17));
    auto emit = [&](uint32_t token) {
        words(send)[0] = token;
        noc_async_write<32>(send, result.get_noc_addr(0), 32);
        noc_async_write_barrier();
    };
    auto finish = [&]() {
        cb_pop_front(17, tiles);
        cb_pop_front(16, tiles);
        noc_semaphore_set(words(arrivals), 0);
        noc_semaphore_set(words(ready), 0);
    };
    if (top_k && top_k <= small_k) {
        const uint32_t heap_l1 = config_l1 + 64;
        const uint32_t inbox = heap_l1 + small_k * sizeof(Candidate);
        const uint32_t payload = top_k * sizeof(Candidate);
        auto heap = reinterpret_cast<tt_l1_ptr Candidate*>(heap_l1);
        auto candidate = [&](uint32_t i) { return Candidate{first + i, order(values[i]), weights[i], 0}; };
        for (uint32_t i = 0; i < top_k; ++i) {
            heap[i] = candidate(i);
        }
        for (uint32_t i = top_k / 2; i > 0; --i) {
            sift(heap, top_k, i - 1);
        }
        for (uint32_t i = top_k; i < valid; ++i) {
            const Candidate item = candidate(i);
            if (less(heap[0], item)) {
                heap[0] = item;
                sift(heap, top_k, 0);
            }
        }
        for (uint32_t size = top_k; size > 1; --size) {
            const Candidate smallest = heap[0];
            heap[0] = heap[size - 1];
            heap[size - 1] = smallest;
            sift(heap, size - 1, 0);
        }
        noc_async_write(heap_l1, peer(0, inbox + me * payload), payload);
        noc_async_write_barrier();
        exchange([&]() {
            const auto lists = reinterpret_cast<volatile tt_l1_ptr Candidate*>(inbox);
            uint32_t consumed[workers] = {};
            auto best = heap;
            float mass = 0.0f;
            for (uint32_t j = 0; j < top_k; ++j) {
                Candidate next = {0xffffffffu, 0, 0.0f, 0};
                uint32_t source = 0;
                for (uint32_t worker = 0; worker < workers; ++worker) {
                    const auto& item = lists[worker * top_k + consumed[worker]];
                    const Candidate entry = {item.token, item.score, item.weight, 0};
                    if (less(next, entry)) {
                        next = entry;
                        source = worker;
                    }
                }
                ++consumed[source];
                best[j] = next;
                mass += next.weight;
            }
            uint32_t keep = top_k;
            if (top_p < 1.0f) {
                const float limit = top_p * mass;
                float prefix = 0.0f;
                keep = 0;
                do {
                    prefix += best[keep].weight;
                    ++keep;
                } while (keep < top_k && prefix < limit);
            }
            for (uint32_t i = 1; i < keep; ++i) {
                const Candidate item = best[i];
                uint32_t j = i;
                while (j && best[j - 1].token > item.token) {
                    best[j] = best[j - 1];
                    --j;
                }
                best[j] = item;
            }
            float retained_mass = 0.0f;
            for (uint32_t i = 0; i < keep; ++i) {
                retained_mass += best[i].weight;
            }
            float target = draw * retained_mass;
            uint32_t index = 0;
            while (index + 1 < keep && (best[index].weight == 0.0f || target > best[index].weight)) {
                target -= best[index].weight;
                ++index;
            }
            words(response)[0] = best[index].token;
        });
        if (me == 0) {
            emit(words(response)[0]);
        }
        finish();
        return;
    }
    const auto rank = [&](uint32_t index) {
        return (static_cast<uint64_t>(order(values[index])) << 18) | (vocab - 1 - first - index);
    };

    auto select = [&](uint64_t floor, float target, bool by_mass) {
        uint64_t prefix = 0;
        uint64_t mask = 0;
        for (int shift = 48; shift >= 0; shift -= 4) {
            float bins[digit_count] = {};
            for (uint32_t i = 0; i < valid; ++i) {
                const uint64_t key = rank(i);
                if (key >= floor && (key & mask) == prefix) {
                    bins[(key >> shift) & 15] += by_mass ? weights[i] : 1.0f;
                }
            }
            for (uint32_t i = 0; i < digit_count; ++i) {
                floats(send)[i] = bins[i];
            }
            exchange([&]() {
                uint32_t picked = 0;
                float lower_mass = 0.0f;
                for (int digit = digit_count - 1; digit >= 0; --digit) {
                    float size = 0.0f;
                    for (uint32_t worker = 0; worker < workers; ++worker) {
                        size += floats(receive + worker * bin_bytes)[digit];
                    }
                    if (size > 0.0f) {
                        picked = digit;
                        lower_mass = size;
                        if (target <= size) {
                            break;
                        }
                        target -= size;
                    }
                }
                words(response)[0] = picked;
                floats(response)[1] = target <= lower_mass ? target : lower_mass;
            });
            prefix |= static_cast<uint64_t>(words(response)[0]) << shift;
            mask |= uint64_t(15) << shift;
            target = floats(response)[1];
        }
        return prefix;
    };
    uint64_t cutoff = top_k && top_k < vocab ? select(0, static_cast<float>(top_k), false) : 0;
    auto local_sum = [&](uint64_t bound) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < valid; ++i) {
            if (rank(i) >= bound) {
                sum += weights[i];
            }
        }
        return sum;
    };
    if (top_p < 1.0f) {
        floats(send)[0] = local_sum(cutoff);
        exchange([&]() {
            float total = 0.0f;
            for (uint32_t worker = 0; worker < workers; ++worker) {
                total += floats(receive + worker * bin_bytes)[0];
            }
            floats(response)[0] = top_p * total;
        });
        cutoff = select(cutoff, floats(response)[0], true);
    }
    floats(send)[0] = local_sum(cutoff);
    exchange([&]() {
        float total = 0;
        for (uint32_t worker = 0; worker < workers; ++worker) {
            total += floats(receive + worker * bin_bytes)[0];
        }
        float rest = draw * total;
        uint32_t chosen = 0;
        for (uint32_t worker = 0; worker < workers; ++worker) {
            const float mass = floats(receive + worker * bin_bytes)[0];
            if (mass > 0 && rest <= mass) {
                chosen = worker;
                break;
            }
            rest -= mass;
        }
        words(response)[0] = chosen;
        floats(response)[1] = rest;
    });
    if (words(response)[0] == me) {
        float target = floats(response)[1];
        uint32_t next = first;
        for (uint32_t i = 0; i < valid; ++i) {
            if (rank(i) >= cutoff && weights[i] > 0.0f) {
                next = first + i;
                target -= weights[i];
                if (target <= 0.0f) {
                    break;
                }
            }
        }
        emit(next);
    }
    finish();
}
