use std::process::Command;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=./src/discretize_cuda.cu");
    println!("cargo::rerun-if-changed=./src/texture_weight_cuda.cu");
    println!("cargo::rerun-if-changed=./src/cuda_common.cu");
    println!("cargo::rerun-if-changed=./src/cuda_init.cu");
    println!("cargo::rerun-if-changed=./src/cuda_merged.cu");
    if cfg!(feature = "use-gpu") {
        #[rustfmt::skip]
        cc::Build::new()
            .cuda(true)
            .cudart("static")
            .flag("-g")
            .flag("-G")
            .flag("--forward-unknown-to-host-compiler")
            .flag("-Wall")
            .flag("-Wextra")
            .flag("--dopt=on")
            .flag("--generate-line-info")
            .flag("-gencode").flag("arch=compute_80,code=sm_80")
            .flag("-gencode").flag("arch=compute_86,code=sm_86")
            .flag("-t0")
            .file("./src/cuda_merged.cu")
            .compile("cuda_lib");
    }

    let output = Command::new("git")
        .args(["describe", "--tags"])
        .output()
        .expect("need git executable and ability to execute process");
    let git_desc = String::from_utf8(output.stdout).expect("git hash is utf8");
    let git_desc = git_desc.trim();

    let output = Command::new("git")
        .args(["diff", "--quiet"])
        .output()
        .expect("needs to execute git diff");

    let version = if output
        .status
        .code()
        .expect("process needs to run successfully")
        != 0
    {
        // unstaged changes in worktree
        format!("{git_desc} (uncommitted changes)")
    } else {
        format!("{git_desc}")
    };
    println!("cargo::rustc-env=YAXS_VERSION={}", version);
}
