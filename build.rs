use std::process::Command;

fn main() {
    println!("cargo::rerun-if-changed=src/discretize_cuda.cu");
    if !cfg!(feature = "cpu-only") {
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

    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .expect("need git executable and ability to execute process");
    let git_hash = String::from_utf8(output.stdout).expect("git hash is utf8");
    let git_hash = git_hash.trim();
    let output = Command::new("git")
        .args(["diff", "--quiet"])
        .output()
        .expect("needs to execute git diff");
    let version = if output
        .status
        .code()
        .expect("process needs to run successfully")
        > 0
    {
        // no unstaged changes not in worktree
        format!("at commit: {git_hash} (unstaged changes)")
    } else {
        format!("at commit: {git_hash} (unstaged changes)")
    };
    eprintln!("{}", version);
    println!("cargo::rustc-env=YAXS_VERSION={}", version);
}
