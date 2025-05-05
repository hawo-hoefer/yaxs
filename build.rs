fn main() {
    println!("cargo::rerun-if-changed=./src/discretize_cuda.cu");
    cc::Build::new()
        .cuda(true)
        .cudart("static")
        .file("./src/discretize_cuda.cu")
        .compile("discretize_cuda");
}
