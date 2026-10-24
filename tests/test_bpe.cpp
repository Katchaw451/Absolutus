#include "ml/bpe.hpp"
#include <cstdio>
#include <cstring>

using namespace absolutus::ml;

#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    return 1; } } while(0)

int main() {
    const std::string corpus =
        "the quick brown fox jumps over the lazy dog. "
        "the quick brown fox jumps over the lazy dog. "
        "the quick brown fox jumps over the lazy dog.";

    BPE b = BPE::train(corpus, 50);

    // 1. Vocab grew past the base char set
    CHECK(b.vocab.size() > 30);
    CHECK(b.merges.size() > 0);
    CHECK(b.merges.size() <= 50);

    // 2. Every merge has new_id == base + index (append order)
    for (size_t i = 0; i < b.merges.size(); ++i) {
        CHECK((int)b.merges[i].new_id == (int)(b.id_word_end + 1 + i));
    }

    // 3. Roundtrip on training words
    for (const char* w : {"the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog."}) {
        auto ids = b.encode_word(w);
        CHECK(!ids.empty());
        std::string decoded = b.decode(ids);
        CHECK(decoded == std::string(w));
    }

    // 4. Common word "the" compresses (originally 4 tokens: t,h,e,</w>)
    {
        auto ids = b.encode_word("the");
        CHECK(ids.size() <= 3);
    }

    // 5. Empty word = just </w>
    {
        auto ids = b.encode_word("");
        CHECK(ids.size() == 1);
        CHECK(ids[0] == b.id_word_end);
    }

    // 6. Unknown char becomes unk
    {
        auto ids = b.encode_word("@");  // not in lowercase corpus
        CHECK(!ids.empty());
        CHECK(ids[0] == b.id_unk);
    }

    // 7. save/load roundtrip preserves vocab, merges, encoding
    {
        const char* path = "/tmp/abs_test.bpe";
        CHECK(b.save(path));
        BPE b2 = BPE::load(path);
        CHECK(b2.vocab.size() == b.vocab.size());
        CHECK(b2.merges.size() == b.merges.size());
        CHECK(b2.id_word_end == b.id_word_end);

        auto a1 = b.encode_word("quick");
        auto a2 = b2.encode_word("quick");
        CHECK(a1 == a2);

        std::remove(path);
    }

    return 0;
}
