//! # walk (compatibility shim)
//!
//! This crate now re-exports [`graphops`] and exists only for compatibility.
//! New code should depend on and import `graphops` directly.
//!
//! In other words:
//! - **Before**: `use walk::pagerank`
//! - **After**:  `use graphops::pagerank`

pub use graphops::*;
