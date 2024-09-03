use crate::relaxed_ik::{RelaxedIK, Opt};
use crate::utils_rust::subscriber_utils::EEPoseGoalsSubscriber;
use std::sync::{Arc, Mutex};
use nalgebra::{Vector3, UnitQuaternion, Quaternion};
use std::io::{self, Write};
use std::os::raw::{c_double, c_int, c_char};
use std::ffi::CStr;
use std::ptr;
use std::panic;

// Modified from the original relaxed_ik_wrapper.rs
// to resemble that of the RangedIK.
//
// I. Multiple instances of RelaxedIK can be created and managed.
// II. Some error handling is added to prevent crashing the outside program.
// III. New helper functions are added to 
//  1) reset initial state
//  2) get the goal poses 
//  3) get the initial poses


// Create a new RelaxedIK instance given a robot info file path in Unitys
#[no_mangle]
pub unsafe extern "C" fn relaxed_ik_new(info_file_name: *const c_char) -> *mut RelaxedIK {
    // Instantiate a RelaxedIK instance with default robot info
    if info_file_name.is_null() {
        match RelaxedIK::from_loaded(1) {
            Ok(relaxed_ik) => Box::into_raw(Box::new(relaxed_ik)),
            Err(_) => return ptr::null_mut(),
        }
    }

    // Load robot info from a file
    else {
        // convert to a Rust string
        let c_str = CStr::from_ptr(info_file_name);
        let info_file_name_str = match c_str.to_str() {
            Ok(str) => str,
            Err(_) => return ptr::null_mut(),
        };
        // instantiate a RelaxedIK instance
        match RelaxedIK::from_info_file_name(info_file_name_str.to_owned(), 1) {
            Ok(relaxed_ik) => Box::into_raw(Box::new(relaxed_ik)),
            Err(_) => return ptr::null_mut(),
        }
    }
}

// Free a RelaxedIK instance given a robot info file name
#[no_mangle]
pub unsafe extern "C" fn relaxed_ik_free(ptr: *mut RelaxedIK) {
    if ptr.is_null() { return }
    Box::from_raw(ptr);
}

// Reset a RelaxedIK initial state given a joint state
#[no_mangle]
pub unsafe extern "C" fn reset(ptr: *mut RelaxedIK, 
    joint_state: *const c_double, joint_state_length: c_int) {

    // Check if the input is valid
    if ptr.is_null() || joint_state.is_null() {
        return;
    }
    let relaxed_ik = unsafe { &mut *ptr };

    let x_slice: &[c_double] = std::slice::from_raw_parts(joint_state, joint_state_length as usize);
    let x_vec = x_slice.to_vec();
    // Reset if length matches
    if x_vec.len() == relaxed_ik.vars.init_state.len() {
        relaxed_ik.vars.reset(x_vec.clone());
    }
}

// Solve a goal with a RelaxedIK instance
#[no_mangle]
pub unsafe extern "C" fn solve(ptr: *mut RelaxedIK, 
    pos_arr: *const c_double, pos_length: c_int, 
    quat_arr: *const c_double, quat_length: c_int) -> Opt {

    // Check if the input is valid
    if ptr.is_null() || pos_arr.is_null() || quat_arr.is_null() {
        return Opt {data: ptr::null(), length: 0}
    }
    let relaxed_ik = unsafe { &mut *ptr };

    let pos_slice: &[c_double] = std::slice::from_raw_parts(pos_arr, pos_length as usize);
    let quat_slice: &[c_double] = std::slice::from_raw_parts(quat_arr, quat_length as usize);
    let pos_vec = pos_slice.to_vec();
    let quat_vec = quat_slice.to_vec();

    // Solve only if length matches
    if (relaxed_ik.vars.robot.num_chains * 3) != pos_vec.len()
        || (relaxed_ik.vars.robot.num_chains * 4) != quat_vec.len() {
        return Opt {data: ptr::null(), length: 0}
    }

    let ja = solve_helper(relaxed_ik, pos_vec, quat_vec);
    let ptr = ja.as_ptr();
    let len = ja.len();
    std::mem::forget(ja);

    Opt {data: ptr, length: len as c_int}
}

fn solve_helper(relaxed_ik: &mut RelaxedIK, pos_goals: Vec<f64>, quat_goals: Vec<f64>) -> Vec<f64> {
    let mut x: Vec<f64> = Vec::new();
    let arc = Arc::new(Mutex::new(EEPoseGoalsSubscriber::new()));
    let mut g = arc.lock().unwrap();

    for i in 0..relaxed_ik.vars.robot.num_chains {
        g.pos_goals.push( Vector3::new(pos_goals[3*i], pos_goals[3*i+1], pos_goals[3*i+2]) );
        let tmp_q = Quaternion::new(quat_goals[4*i+3], quat_goals[4*i], quat_goals[4*i+1], quat_goals[4*i+2]);
        g.quat_goals.push( UnitQuaternion::from_quaternion(tmp_q) );
    }

    x = relaxed_ik.solve(&g);
    return x;
}

// Solve a goal with a RelaxedIK instance precisely
#[no_mangle]
pub unsafe extern "C" fn solve_precise(ptr: *mut RelaxedIK, 
    pos_arr: *const c_double, pos_length: c_int, 
    quat_arr: *const c_double, quat_length: c_int,
    max_iter: c_int
) -> Opt {

    // Check if the input is valid
    if ptr.is_null() || pos_arr.is_null() || quat_arr.is_null() {
        return Opt {data: ptr::null(), length: 0}
    }
    let relaxed_ik = unsafe { &mut *ptr };

    let pos_slice: &[c_double] = std::slice::from_raw_parts(pos_arr, pos_length as usize);
    let quat_slice: &[c_double] = std::slice::from_raw_parts(quat_arr, quat_length as usize);
    let pos_vec = pos_slice.to_vec();
    let quat_vec = quat_slice.to_vec();

    // Solve only if length matches
    if (relaxed_ik.vars.robot.num_chains * 3) != pos_vec.len()
        || (relaxed_ik.vars.robot.num_chains * 4) != quat_vec.len() {
        return Opt {data: ptr::null(), length: 0}
    }

    let ja = solve_precise_helper(relaxed_ik, pos_vec, quat_vec, max_iter as u32);
    let ptr = ja.as_ptr();
    let len = ja.len();
    std::mem::forget(ja);

    Opt {data: ptr, length: len as c_int}
}

fn solve_precise_helper(relaxed_ik: &mut RelaxedIK, pos_goals: Vec<f64>, quat_goals: Vec<f64>, max_iter: u32) -> Vec<f64> {
    let mut x: Vec<f64> = Vec::new();
    let arc = Arc::new(Mutex::new(EEPoseGoalsSubscriber::new()));
    let mut g = arc.lock().unwrap();

    for i in 0..relaxed_ik.vars.robot.num_chains {
        g.pos_goals.push( Vector3::new(pos_goals[3*i], pos_goals[3*i+1], pos_goals[3*i+2]) );
        let tmp_q = Quaternion::new(quat_goals[4*i+3], quat_goals[4*i], quat_goals[4*i+1], quat_goals[4*i+2]);
        g.quat_goals.push( UnitQuaternion::from_quaternion(tmp_q) );
    }

    x = relaxed_ik.solve_precise(&g, max_iter);
    return x;
}

// Get current end-effector pose (forwards kinematics) with a RelaxedIK instance
#[no_mangle]
pub unsafe extern "C" fn get_current_poses(ptr: *mut RelaxedIK) -> Opt {
    // Check if the input is valid
    if ptr.is_null() {
        return Opt {data: ptr::null(), length: 0}
    }
    let relaxed_ik = unsafe { &mut *ptr };
    
    // Get the poses
    let mut pose = Vec::new();
    let mut out_x = relaxed_ik.vars.xopt.clone();
    let ee_poses = relaxed_ik.vars.robot.get_ee_pos_and_quat_immutable(&out_x);
    for i in 0..relaxed_ik.vars.robot.num_chains {
        pose.push(ee_poses[i].0.x);
        pose.push(ee_poses[i].0.y);
        pose.push(ee_poses[i].0.z);
        pose.push(ee_poses[i].1.coords.x);
        pose.push(ee_poses[i].1.coords.y);
        pose.push(ee_poses[i].1.coords.z);
        pose.push(ee_poses[i].1.coords.w);
    }

    let ptr = pose.as_ptr();
    let len = pose.len();
    std::mem::forget(pose);
    std::mem::forget(out_x);

    Opt {data: ptr, length: len as c_int}
}

// Get the end-effector goal pose of a RelaxedIK instance
#[no_mangle]
pub unsafe extern "C" fn get_goal_poses(ptr: *mut RelaxedIK) -> Opt {
    // Check if the input is valid
    if ptr.is_null() {
        return Opt {data: ptr::null(), length: 0}
    }
    let relaxed_ik = unsafe { &mut *ptr };

    // Get the poses
    let mut pose = Vec::new();
    for i in 0..relaxed_ik.vars.goal_positions.len() {
        pose.push(relaxed_ik.vars.goal_positions[i].x);
        pose.push(relaxed_ik.vars.goal_positions[i].y);
        pose.push(relaxed_ik.vars.goal_positions[i].z);
    }
    for i in 0..relaxed_ik.vars.goal_quats.len() {
        pose.push(relaxed_ik.vars.goal_quats[i].coords.x);
        pose.push(relaxed_ik.vars.goal_quats[i].coords.y);
        pose.push(relaxed_ik.vars.goal_quats[i].coords.z);
        pose.push(relaxed_ik.vars.goal_quats[i].coords.w);
    }
    let ptr = pose.as_ptr();
    let len = pose.len();
    std::mem::forget(pose);

    Opt {data: ptr, length: len as c_int}
}

// Get the end-effector initial pose of a RelaxedIK instance
#[no_mangle]
pub unsafe extern "C" fn get_init_poses(ptr: *mut RelaxedIK) -> Opt {
    // Check if the input is valid
    if ptr.is_null() {
        return Opt {data: ptr::null(), length: 0}
    }
    let relaxed_ik = unsafe { &mut *ptr };

    // Get the poses
    let mut pose = Vec::new();
    for i in 0..relaxed_ik.vars.init_ee_positions.len() {
        pose.push(relaxed_ik.vars.init_ee_positions[i].x);
        pose.push(relaxed_ik.vars.init_ee_positions[i].y);
        pose.push(relaxed_ik.vars.init_ee_positions[i].z);
    }
    for i in 0..relaxed_ik.vars.init_ee_quats.len() {
        pose.push(relaxed_ik.vars.init_ee_quats[i].coords.x);
        pose.push(relaxed_ik.vars.init_ee_quats[i].coords.y);
        pose.push(relaxed_ik.vars.init_ee_quats[i].coords.z);
        pose.push(relaxed_ik.vars.init_ee_quats[i].coords.w);
    }
    let ptr = pose.as_ptr();
    let len = pose.len();
    std::mem::forget(pose);

    Opt {data: ptr, length: len as c_int}
}
