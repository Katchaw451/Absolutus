#include "ml/bpe.hpp"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <stdexcept>

namespace absolutus {
namespace ml {

static std::vector<std::string> split_words(const std::string& corpus) {
    std::vector<std::string> words;
    std::string cur;
    for (char c : corpus) {
        if (c == ' ' || c == '\n' || c == '\t' || c == '\r') {
            if (!cur.empty()) { words.push_back(cur); cur.clear(); }
        } else {
            cur += c;
        }
    }
    if (!cur.empty()) words.push_back(cur);
    return words;
}

BPE BPE::train(const std::string& corpus, size_t n_merges) {
    BPE b;
    b.vocab = {"<pad>", "<unk>", "<bos>", "<eos>"};
    b.str_to_id["<pad>"] = 0;
    b.str_to_id["<unk>"] = 1;
    b.str_to_id["<bos>"] = 2;
    b.str_to_id["<eos>"] = 3;

    std::map<char, bool> char_seen;
    for (char c : corpus) {
        if (c == ' ' || c == '\n' || c == '\t' || c == '\r') continue;
        char_seen[c] = true;
    }
    for (auto& kv : char_seen) {
        std::string s(1, kv.first);
        b.str_to_id[s] = (int)b.vocab.size();
        b.vocab.push_back(s);
    }

    b.id_word_end = (int)b.vocab.size();
    b.str_to_id["</w>"] = b.id_word_end;
    b.vocab.push_back("</w>");

    std::map<std::string, int> word_freq;
    for (auto& w : split_words(corpus)) word_freq[w]++;

    std::map<std::string, std::vector<int>> word_tokens;
    for (auto& kv : word_freq) {
        std::vector<int> toks;
        for (char c : kv.first) {
            std::string s(1, c);
            auto it = b.str_to_id.find(s);
            toks.push_back(it != b.str_to_id.end() ? it->second : b.id_unk);
        }
        toks.push_back(b.id_word_end);
        word_tokens[kv.first] = toks;
    }

    for (size_t iter = 0; iter < n_merges; ++iter) {
        std::map<std::pair<int,int>, int> pair_counts;
        for (auto& kv : word_freq) {
            auto& toks = word_tokens[kv.first];
            for (size_t i = 0; i + 1 < toks.size(); ++i)
                pair_counts[{toks[i], toks[i+1]}] += kv.second;
        }
        if (pair_counts.empty()) break;

        auto best = pair_counts.begin();
        for (auto it = pair_counts.begin(); it != pair_counts.end(); ++it)
            if (it->second > best->second) best = it;
        if (best->second < 2) break;

        const int ta = best->first.first;
        const int tb = best->first.second;
        const int new_id = (int)b.vocab.size();
        b.vocab.push_back(b.vocab[ta] + b.vocab[tb]);
        b.str_to_id[b.vocab.back()] = new_id;
        b.merges.push_back({ta, tb, new_id});

        for (auto& kv : word_tokens) {
            auto& toks = kv.second;
            std::vector<int> next;
            next.reserve(toks.size());
            for (size_t i = 0; i < toks.size(); ++i) {
                if (i + 1 < toks.size() && toks[i] == ta && toks[i+1] == tb) {
                    next.push_back(new_id);
                    ++i;
                } else {
                    next.push_back(toks[i]);
                }
            }
            toks.swap(next);
        }
    }
    return b;
}

std::vector<int> BPE::encode_word(const std::string& word) const {
    std::vector<int> toks;
    toks.reserve(word.size() + 1);
    for (char c : word) {
        std::string s(1, c);
        auto it = str_to_id.find(s);
        toks.push_back(it != str_to_id.end() ? it->second : id_unk);
    }
    auto it_we = str_to_id.find("</w>");
    if (it_we != str_to_id.end()) toks.push_back(it_we->second);

    for (auto& m : merges) {
        if (toks.size() < 2) break;
        std::vector<int> next;
        next.reserve(toks.size());
        bool changed = false;
        for (size_t i = 0; i < toks.size(); ++i) {
            if (i + 1 < toks.size() && toks[i] == m.a && toks[i+1] == m.b) {
                next.push_back(m.new_id);
                ++i;
                changed = true;
            } else {
                next.push_back(toks[i]);
            }
        }
        if (changed) toks.swap(next);
    }
    return toks;
}

std::string BPE::decode(const std::vector<int>& ids) const {
    std::string out;
    for (int id : ids) {
        if (id < 0 || (size_t)id >= vocab.size()) continue;
        if (id <= 3) continue;
        const std::string& piece = vocab[id];
        if (piece.size() >= 4 && piece.compare(piece.size() - 4, 4, "</w>") == 0) {
            out.append(piece, 0, piece.size() - 4);
            out += ' ';
        } else {
            out += piece;
        }
    }
    if (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

bool BPE::save(const std::string& path) const {
    std::ofstream f(path);
    if (!f) return false;
    f << vocab.size() << "\n";
    for (auto& v : vocab) f << v << "\n";
    f << merges.size() << "\n";
    for (auto& m : merges) f << m.a << " " << m.b << " " << m.new_id << "\n";
    return true;
}

BPE BPE::load(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("BPE::load: cannot open " + path);
    BPE b;
    size_t nv;
    if (!(f >> nv)) throw std::runtime_error("BPE::load: bad header");
    b.vocab.resize(nv);
    f.ignore(1, '\n');
    for (size_t i = 0; i < nv; ++i) {
        std::getline(f, b.vocab[i]);
        b.str_to_id[b.vocab[i]] = (int)i;
    }
    auto it_we = b.str_to_id.find("</w>");
    if (it_we != b.str_to_id.end()) b.id_word_end = it_we->second;

    size_t nm;
    if (!(f >> nm)) throw std::runtime_error("BPE::load: bad merge count");
    b.merges.resize(nm);
    for (size_t i = 0; i < nm; ++i)
        f >> b.merges[i].a >> b.merges[i].b >> b.merges[i].new_id;
    return b;
}

}}
