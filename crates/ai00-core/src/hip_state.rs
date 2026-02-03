//! HIP backend state adapter implementing `web_rwkv::runtime::model::State`.
//!
//! This module bridges between ai00's state management (which uses `TensorCpu<f32>`)
//! and hip-rwkv's native `HipState` format. The adapter delegates `load`/`back` to
//! `HipRuntime::load_state`/`get_state`, converting between the two formats.
//!
//! The TensorCpu state format matches the v7 WebGPU layout:
//! `[n_embd, head_size + 2, n_layer, 1]` where per-layer data is:
//! - `[n_embd, 0..head_size]` — WKV recurrent state (n_head matrices of head_size x head_size)
//! - `[n_embd, head_size]` — attention token-shift state
//! - `[n_embd, head_size + 1]` — FFN token-shift state
//!
//! HipState stores these separately as:
//! - `att_states[layer]`: `PinnedBuffer<f32>` of length `head_size * head_size * n_head`
//! - `att_shift_states[layer]`: `PinnedBuffer<f16>` of length `n_embd`
//! - `ffn_states[layer]`: `PinnedBuffer<f16>` of length `n_embd`

use std::any::Any;
use std::sync::Arc;

use futures::future::BoxFuture;
use half::f16;

use hip_rwkv::hip::{HipRuntime, HipState, PinnedBuffer, Rwkv7ModelInfo, StateLayout};
use web_rwkv::runtime::model::{AsAny, State};
use web_rwkv::tensor::kind::ReadWrite;
use web_rwkv::tensor::shape::Shape;
use web_rwkv::tensor::{TensorCpu, TensorError, TensorErrorKind, TensorGpu, TensorGpuView, TensorInit, TensorShape};

/// Adapter implementing `web_rwkv::runtime::model::State` for the HIP backend.
///
/// Wraps an `Arc<HipRuntime>` and delegates state load/back operations to it,
/// converting between the TensorCpu<f32> format used by ai00 and the HipState
/// format used by hip-rwkv.
pub struct HipStateAdapter {
    runtime: Arc<HipRuntime>,
    info: Rwkv7ModelInfo,
    num_batch: usize,
}

impl HipStateAdapter {
    /// Create a new HipStateAdapter.
    ///
    /// # Arguments
    /// * `runtime` - Shared HipRuntime instance
    /// * `num_batch` - Number of batches to support
    pub fn new(runtime: Arc<HipRuntime>, num_batch: usize) -> Self {
        let info = runtime.info().clone();
        Self {
            runtime,
            info,
            num_batch,
        }
    }

    /// Helper to create a TensorError from an HIP error.
    fn hip_error(msg: impl std::fmt::Display) -> TensorError {
        // Use Deduce as a generic error kind; the actual error context is in the
        // log message. TensorErrorKind has no Runtime variant, so Deduce is the
        // closest fit for "operation failed for external reasons".
        log::error!("HIP state error: {}", msg);
        TensorError::new(TensorErrorKind::Deduce)
    }

    /// Convert a TensorCpu<f32> state (v7 WebGPU format) into a HipState.
    ///
    /// The input tensor has shape `[n_embd, head_size + 2, n_layer, 1]` and contains
    /// one batch of state data. This function extracts the WKV, att_shift, and ffn
    /// components for each layer and packs them into a HipState.
    ///
    /// Note: att_shift and ffn states are converted from f32 to f16 to match
    /// HipState's native format.
    fn tensor_to_hip_state(&self, tensor: &TensorCpu<f32>) -> Result<HipState, TensorError> {
        let n_layer = self.info.n_layer;
        let n_embd = self.info.n_embd;
        let head_size = self.info.head_size;
        let n_head = self.info.n_head;
        let row_stride = head_size + 2; // columns per layer in the tensor

        let data = tensor.data();

        let mut att_states = Vec::with_capacity(n_layer);
        let mut att_shift_states = Vec::with_capacity(n_layer);
        let mut ffn_states = Vec::with_capacity(n_layer);

        for layer in 0..n_layer {
            let layer_offset = layer * n_embd * row_stride;

            // WKV state: [n_embd, 0..head_size] = n_embd * head_size f32 values
            // In the tensor, data is column-major: for each column c, elements are
            // data[layer_offset + c * n_embd .. layer_offset + (c+1) * n_embd]
            let wkv_size = head_size * head_size * n_head; // = n_embd * head_size
            let mut wkv_buf = PinnedBuffer::<f32>::new(wkv_size)
                .map_err(|e| Self::hip_error(format!("pinned alloc failed: {}", e)))?;
            let wkv_slice = wkv_buf.as_slice_mut();
            for col in 0..head_size {
                let src_start = layer_offset + col * n_embd;
                let dst_start = col * n_embd;
                wkv_slice[dst_start..dst_start + n_embd]
                    .copy_from_slice(&data[src_start..src_start + n_embd]);
            }
            att_states.push(wkv_buf);

            // Att shift state: [n_embd, head_size] = n_embd f32 values -> f16
            let att_shift_offset = layer_offset + head_size * n_embd;
            let mut att_shift_buf = PinnedBuffer::<f16>::new(n_embd)
                .map_err(|e| Self::hip_error(format!("pinned alloc failed: {}", e)))?;
            let att_shift_slice = att_shift_buf.as_slice_mut();
            for i in 0..n_embd {
                att_shift_slice[i] = f16::from_f32(data[att_shift_offset + i]);
            }
            att_shift_states.push(att_shift_buf);

            // FFN state: [n_embd, head_size + 1] = n_embd f32 values -> f16
            let ffn_offset = layer_offset + (head_size + 1) * n_embd;
            let mut ffn_buf = PinnedBuffer::<f16>::new(n_embd)
                .map_err(|e| Self::hip_error(format!("pinned alloc failed: {}", e)))?;
            let ffn_slice = ffn_buf.as_slice_mut();
            for i in 0..n_embd {
                ffn_slice[i] = f16::from_f32(data[ffn_offset + i]);
            }
            ffn_states.push(ffn_buf);
        }

        Ok(HipState {
            batch_size: 1,
            att_states,
            att_shift_states,
            ffn_states,
            v_first: None,
            layout: StateLayout::Decode,
        })
    }

    /// Convert a HipState into a TensorCpu<f32> (v7 WebGPU format).
    ///
    /// Packs the per-layer WKV, att_shift, and ffn state from HipState into a
    /// single tensor with shape `[n_embd, head_size + 2, n_layer, 1]`.
    ///
    /// Note: att_shift and ffn states are converted from f16 to f32.
    fn hip_state_to_tensor(&self, state: &HipState) -> Result<TensorCpu<f32>, TensorError> {
        let n_layer = self.info.n_layer;
        let n_embd = self.info.n_embd;
        let head_size = self.info.head_size;
        let row_stride = head_size + 2;

        let total = n_embd * row_stride * n_layer;
        let mut data = vec![0.0f32; total];

        for layer in 0..n_layer {
            let layer_offset = layer * n_embd * row_stride;

            // WKV state: copy head_size columns of n_embd elements
            let wkv_data = state.att_states[layer].as_slice();
            for col in 0..head_size {
                let src_start = col * n_embd;
                let dst_start = layer_offset + col * n_embd;
                data[dst_start..dst_start + n_embd]
                    .copy_from_slice(&wkv_data[src_start..src_start + n_embd]);
            }

            // Att shift state: f16 -> f32
            let att_shift_data = state.att_shift_states[layer].as_slice();
            let att_shift_offset = layer_offset + head_size * n_embd;
            for i in 0..n_embd {
                data[att_shift_offset + i] = att_shift_data[i].to_f32();
            }

            // FFN state: f16 -> f32
            let ffn_data = state.ffn_states[layer].as_slice();
            let ffn_offset = layer_offset + (head_size + 1) * n_embd;
            for i in 0..n_embd {
                data[ffn_offset + i] = ffn_data[i].to_f32();
            }
        }

        let shape = Shape::new(n_embd, row_stride, n_layer, 1);
        TensorCpu::from_data(shape, data)
    }
}

impl AsAny for HipStateAdapter {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl State for HipStateAdapter {
    fn num_batch(&self) -> usize {
        self.num_batch
    }

    fn init_shape(&self) -> Shape {
        let head_size = self.info.head_size;
        Shape::new(self.info.n_embd, head_size + 2, self.info.n_layer, 1)
    }

    fn init(&self) -> TensorCpu<f32> {
        let shape = self.init_shape();
        let data = vec![0.0f32; shape.len()];
        TensorCpu::from_data(shape, data).expect("failed to create init state tensor")
    }

    fn load(&self, tensor: TensorCpu<f32>, _batch: usize) -> Result<(), TensorError> {
        let head_size = self.info.head_size;
        tensor.check_shape([self.info.n_embd, head_size + 2, self.info.n_layer, 1])?;

        let hip_state = self.tensor_to_hip_state(&tensor)?;
        self.runtime
            .load_state(&hip_state)
            .map_err(|e| Self::hip_error(format!("load_state failed: {}", e)))
    }

    fn back(&self, _batch: usize) -> BoxFuture<'_, Result<TensorCpu<f32>, TensorError>> {
        Box::pin(async move {
            let hip_state = self
                .runtime
                .get_state()
                .map_err(|e| Self::hip_error(format!("get_state failed: {}", e)))?;
            self.hip_state_to_tensor(&hip_state)
        })
    }

    fn att(&self, _layer: usize) -> Result<TensorGpuView<'_, f32>, TensorError> {
        Err(Self::hip_error(
            "HIP backend does not support per-layer att state access",
        ))
    }

    fn ffn(&self, _layer: usize) -> Result<TensorGpuView<'_, f32>, TensorError> {
        Err(Self::hip_error(
            "HIP backend does not support per-layer ffn state access",
        ))
    }

    fn write(&self, _tensor: TensorGpu<f32, ReadWrite>, _batch: usize) -> Result<(), TensorError> {
        Err(Self::hip_error(
            "HIP backend does not support GPU state write",
        ))
    }

    fn read(&self, _batch: usize) -> Result<TensorGpu<f32, ReadWrite>, TensorError> {
        Err(Self::hip_error(
            "HIP backend does not support GPU state read",
        ))
    }

    fn embed(&self, layer: usize, backed: TensorCpu<f32>) -> Result<TensorCpu<f32>, TensorError> {
        backed.slice(.., 0, layer, ..)
    }
}
