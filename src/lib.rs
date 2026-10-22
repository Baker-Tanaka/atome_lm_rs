#![no_std]

extern crate alloc;

mod model;

pub use model::{Atome, CachedGeneration, Error, Inference, CONTEXT, VOCAB};

pub const MODEL_BYTES: &[u8] = include_bytes!("../models/atome_944k.bin");
