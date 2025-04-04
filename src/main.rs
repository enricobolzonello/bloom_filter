use bloom_filter::SimpleBloomFilter;

fn main() {
    let mut bloom_filter = SimpleBloomFilter::new_with_seed(100, 0.1, 1234);
    bloom_filter.insert("ciao");
    println!("{}", bloom_filter.test("ciao"));
    println!("{}", bloom_filter.test("hello world"));

    println!("{:?}", bloom_filter);
}
