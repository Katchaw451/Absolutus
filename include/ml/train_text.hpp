#pragma once
#include <cstdint>
#include <string>

namespace absolutus {
namespace ml {

// Train MiniLM on a raw text corpus:
//   1. Train a BPE tokenizer on the corpus.
//   2. Tokenize the whole corpus into a flat id stream.
//   3. Slice into windows of seq_len, train with next-token prediction
//      (target = input shifted by 1).
//   4. Generate `n_sample` tokens greedily from a prompt and print.
int run_train_text(int epochs, double lr, uint64_t seed,
                   const std::string& corpus_path,
                   size_t n_merges,
                   size_t d_model, size_t d_ff, size_t n_blocks,
                   size_t seq_len, size_t n_sample);

}}
