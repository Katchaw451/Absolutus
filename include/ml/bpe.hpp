#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace absolutus {
namespace ml {

// Word-level Byte-Pair Encoding (Sennrich 2016 style).
// Corpus is split on whitespace; each word gets a "</w>" end-of-word
// marker, then merges greedily reduce each word to fewer, longer tokens.
// Merges cannot cross word boundaries because "</w>" is only at word end.

struct BPEMerge {
    int a, b, new_id;
};

class BPE {
public:
    std::vector<std::string> vocab;            // id -> string
    std::vector<BPEMerge> merges;              // in creation order
    std::map<std::string, int> str_to_id;      // string -> id

    int id_pad = 0;
    int id_unk = 1;
    int id_bos = 2;
    int id_eos = 3;
    int id_word_end = -1;                      // id of "</w>"

    // Train on a raw text corpus. Returns a BPE with:
    //   vocab = 4 specials + unique non-whitespace chars + "</w>" + n_merges
    static BPE train(const std::string& corpus, size_t n_merges);

    // Encode a single word (no whitespace) to ids. Adds </w> at the end.
    std::vector<int> encode_word(const std::string& word) const;

    // Decode ids back to a string. Strips </w> markers and inserts single
    // spaces at word boundaries. Skips special tokens (ids 0..3).
    std::string decode(const std::vector<int>& ids) const;

    bool save(const std::string& path) const;
    static BPE load(const std::string& path);
};

}}
