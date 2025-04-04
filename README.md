# Simple Bloom Filter implementation in Rust

For educational purposes. Uses the naive implementation (not cache-friendly) with [murmur3 hash function](https://en.wikipedia.org/wiki/MurmurHash) and [Kirsch-Mitzenmacher optimization](https://www.eecs.harvard.edu/~michaelm/postscripts/tr-02-05.pdf).

Hashable types should support the [Read trait](https://doc.rust-lang.org/nightly/std/io/trait.Read.html).