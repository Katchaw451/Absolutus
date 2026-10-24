#include "ml/minilm.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace absolutus {
namespace ml {

static bool all_finite(const std::vector<double>& v) {
    for (double x : v) if (!std::isfinite(x)) return false;
    return true;
}

static void report_nan(const char* stage, const std::vector<double>& v) {
    for (size_t i = 0; i < v.size(); ++i)
        if (!std::isfinite(v[i])) {
            std::fprintf(stderr, "DBG: NaN in %s idx=%zu val=%g\n", stage, i, v[i]);
            return;
        }
}

MiniLM::MiniLM(size_t sl, size_t dm, size_t dff, size_t vs,
               size_t nblocks, std::mt19937_64& rng, double mom)
    : seq_len(sl), d_model(dm), d_ff(dff), vocab_size(vs), n_blocks(nblocks),
      momentum(mom),
      embed(vs * dm), dembed(vs * dm, 0.0),
      pe(sl, dm),
      head(dm, vs, Activation::Linear, rng, mom),
      h0_(sl * dm, 0.0), logits_(sl * vs, 0.0),
      cur_(sl * dm, 0.0), grad_h0_(sl * dm, 0.0) {
    std::normal_distribution<double> dist(0.0, 0.1);
    for (auto& e : embed) e = dist(rng);
    blocks.reserve(nblocks);
    for (size_t b = 0; b < nblocks; ++b)
        blocks.emplace_back(sl, dm, dff, rng, true, mom);
}

const std::vector<double>& MiniLM::forward(const std::vector<size_t>& tokens) {
    if (tokens.size() != seq_len)
        throw std::invalid_argument("MiniLM::forward: token count");
    tokens_ = tokens;

    for (size_t t = 0; t < seq_len; ++t)
        for (size_t i = 0; i < d_model; ++i)
            h0_[t*d_model + i] = embed[tokens[t]*d_model + i] + pe.table[t*d_model + i];

    if (!all_finite(h0_)) { report_nan("h0_", h0_);
        throw std::runtime_error("MiniLM: NaN in h0_"); }

    cur_ = h0_;
    for (size_t b = 0; b < blocks.size(); ++b) {
        cur_ = blocks[b].forward(cur_);
        if (!all_finite(cur_)) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "after block %zu", b);
            report_nan(buf, cur_);
            throw std::runtime_error("MiniLM: NaN in block");
        }
    }

    for (size_t t = 0; t < seq_len; ++t) {
        std::vector<double> v(cur_.begin() + t*d_model, cur_.begin() + (t+1)*d_model);
        auto lg = head.forward(v);
        if (!all_finite(lg)) { report_nan("head_out", lg);
            throw std::runtime_error("MiniLM: NaN in head"); }
        std::copy(lg.begin(), lg.end(), logits_.begin() + t*vocab_size);
    }
    return logits_;
}

void MiniLM::backward(const std::vector<double>& grad_logits) {
    if (grad_logits.size() != seq_len * vocab_size)
        throw std::invalid_argument("MiniLM::backward: grad shape");

    std::fill(head.dW.begin(), head.dW.end(), 0.0);
    std::fill(head.db.begin(), head.db.end(), 0.0);
    std::vector<double> acc_head_dW(head.W.size(), 0.0);
    std::vector<double> acc_head_db(head.b.size(), 0.0);
    std::vector<double> grad_cur(seq_len * d_model, 0.0);

    for (size_t t = 0; t < seq_len; ++t) {
        std::vector<double> g(grad_logits.begin() + t*vocab_size,
                              grad_logits.begin() + (t+1)*vocab_size);
        std::vector<double> v(cur_.begin() + t*d_model, cur_.begin() + (t+1)*d_model);
        head.forward(v);
        auto g_v = head.backward(g);
        for (size_t i = 0; i < head.dW.size(); ++i) acc_head_dW[i] += head.dW[i];
        for (size_t i = 0; i < head.db.size(); ++i) acc_head_db[i] += head.db[i];
        for (size_t i = 0; i < d_model; ++i) grad_cur[t*d_model + i] = g_v[i];
    }
    std::copy(acc_head_dW.begin(), acc_head_dW.end(), head.dW.begin());
    std::copy(acc_head_db.begin(), acc_head_db.end(), head.db.begin());

    for (auto it = blocks.rbegin(); it != blocks.rend(); ++it)
        grad_cur = it->backward(grad_cur);

    std::fill(dembed.begin(), dembed.end(), 0.0);
    for (size_t t = 0; t < seq_len; ++t)
        for (size_t i = 0; i < d_model; ++i)
            dembed[tokens_[t]*d_model + i] += grad_cur[t*d_model + i];
}

void MiniLM::update(double lr) {
    for (size_t i = 0; i < embed.size(); ++i) embed[i] -= lr * dembed[i];
    for (auto& b : blocks) b.update(lr);
    head.update(lr);
}

void MiniLM::clip_gradients(double max_norm) {
    double n2 = 0.0;
    auto acc = [&](const std::vector<double>& g) {
        for (double v : g) n2 += v * v;
    };
    acc(dembed);
    acc(head.dW); acc(head.db);
    for (auto& b : blocks) {
        acc(b.ln1.dgamma); acc(b.ln1.dbeta);
        acc(b.ln2.dgamma); acc(b.ln2.dbeta);
        acc(b.attn.dWq); acc(b.attn.dWk); acc(b.attn.dWv);
        acc(b.ffn1.dW);  acc(b.ffn1.db);
        acc(b.ffn2.dW);  acc(b.ffn2.db);
    }
    const double norm = std::sqrt(n2);
    if (norm <= max_norm || norm == 0.0) return;
    const double scale = max_norm / norm;
    auto sc = [&](std::vector<double>& g) { for (auto& v : g) v *= scale; };
    sc(dembed);
    sc(head.dW); sc(head.db);
    for (auto& b : blocks) {
        sc(b.ln1.dgamma); sc(b.ln1.dbeta);
        sc(b.ln2.dgamma); sc(b.ln2.dbeta);
        sc(b.attn.dWq); sc(b.attn.dWk); sc(b.attn.dWv);
        sc(b.ffn1.dW);  sc(b.ffn1.db);
        sc(b.ffn2.dW);  sc(b.ffn2.db);
    }
}

}}
