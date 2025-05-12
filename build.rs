fn main() {
    println!("cargo::rerun-if-changed=./src/discretize_cuda.cu");
    #[rustfmt::skip]
    cc::Build::new()
        .cuda(true)
        .cudart("static")
        .flag("--forward-unknown-to-host-compiler")
        .flag("-Wall")
        .flag("-Wextra")
        .flag("--dopt=on")
        .flag("--generate-line-info")
        .flag("-gencode").flag("arch=compute_80,code=sm_80")
        .flag("-gencode").flag("arch=compute_86,code=sm_86")
        .file("./src/discretize_cuda.cu")
        .compile("discretize_cuda");
}
