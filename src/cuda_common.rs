use std::ffi::{c_char, c_int, CStr};

use lazy_static::lazy_static;
use log::{debug, error, info};

pub extern "C" fn c_error_handler(
    _file: *const c_char,
    _line: c_int,
    msg: *const c_char,
    cuda_err_code: c_int,
    cuda_err: *const c_char,
) {
    let msg = unsafe { CStr::from_ptr(msg) };
    let cuda_err = unsafe { CStr::from_ptr(cuda_err) };
    error!(
        "CUDA Error {} while {}: {}",
        cuda_err_code,
        msg.to_str().expect("valid utf-8"),
        cuda_err.to_str().expect("valid utf-8")
    );
}

pub extern "C" fn c_info_handler(msg: *const c_char) {
    let msg = unsafe { CStr::from_ptr(msg) };
    info!("CUDA: {}", msg.to_str().expect("valid utf-8"));
}

pub extern "C" fn c_debug_handler(msg: *const c_char) {
    let msg = unsafe { CStr::from_ptr(msg) };
    debug!("CUDA: {}", msg.to_str().expect("valid utf-8"));
}

pub struct DevInfo {
    pub device_name: String,
    pub available_memory_bytes: usize,
    pub init_free_memory_bytes: usize,
    pub mem_limit_bytes: usize,
    pub api_version: i32,
    pub runtime_version: i32,
    pub device_id: i32,
}

lazy_static! {
    pub static ref CUDA_DEVICE_INFO: DevInfo = init_cuda();
}

fn init_cuda() -> DevInfo {
    #[repr(C)]
    pub struct FFIDevInfo {
        pub device_name: *const c_char, // if device_name is null, init has failed
        pub available_memory_bytes: usize,
        pub init_free_memory_bytes: usize,
        pub api_version: c_int,
        pub runtime_version: c_int,
        pub device_id: c_int,
    }

    #[link(name = "cuda_lib")]
    #[rustfmt::skip]
    extern "C" {
        pub fn init_get_dev_info(
           error_print_handle: extern "C" fn(
               file: *const c_char,
               line: c_int,
               msg: *const c_char,
               cuda_err_code: c_int,
               cuda_err: *const c_char,
           ),
           info_print_handle: extern "C" fn(msg: *const c_char),
           debug_print_handle: extern "C" fn(msg: *const c_char),
        ) -> FFIDevInfo;
    }

    debug!("ffi calling init_get_dev_info");
    let di = unsafe { init_get_dev_info(c_error_handler, c_info_handler, c_debug_handler) };
    debug!("finished init_get_dev_info ffi call");
    if di.device_name.is_null() {
        error!("Could not initialize cuda or get device parameters. Quitting...");
        std::process::exit(1);
    }
    debug!("successfully got device info");

    let device_name = unsafe { CStr::from_ptr(di.device_name) }
        .to_str()
        .expect("cuda device name must be utf-8")
        .to_string();

    DevInfo {
        device_name,
        available_memory_bytes: di.available_memory_bytes,
        api_version: di.api_version as i32,
        runtime_version: di.runtime_version as i32,
        device_id: di.device_id as i32,
        init_free_memory_bytes: di.init_free_memory_bytes,
        mem_limit_bytes: di.init_free_memory_bytes * 9 / 10,
    }
}
