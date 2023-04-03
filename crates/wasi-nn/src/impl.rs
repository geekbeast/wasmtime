//! Implements the wasi-nn API.

use crate::ctx::{LoadedModel, RegisteredModel, WasiNnResult as Result};
use crate::witx::types::{
    ExecutionTarget, Graph, GraphBuilder, GraphBuilderArray, GraphEncoding, GraphExecutionContext,
    Tensor,
};
use crate::witx::wasi_ephemeral_nn::WasiEphemeralNn;
use crate::WasiNnCtx;
use thiserror::Error;
use wiggle::GuestPtr;

const MAX_GUEST_MODEL_REGISTRATION_SIZE: usize = 20 * 1024 * 1024; //20M

#[derive(Debug, Error)]
pub enum UsageError {
    #[error("Invalid context; has the load function been called?")]
    InvalidContext,
    #[error("Only OpenVINO's IR is currently supported, passed encoding: {0:?}")]
    InvalidEncoding(GraphEncoding),
    #[error("OpenVINO expects only two buffers (i.e. [ir, weights]), passed: {0}")]
    InvalidNumberOfBuilders(u32),
    #[error("Invalid graph handle; has it been loaded?")]
    InvalidGraphHandle,
    #[error("Invalid execution context handle; has it been initialized?")]
    InvalidExecutionContextHandle,
    #[error("Not enough memory to copy tensor data of size: {0}")]
    NotEnoughMemory(u32),
    #[error("Model size {0} exceeds allowed quota of {1}")]
    ModelTooLarge(usize, usize),
}

impl WasiNnCtx {
    fn build_graph(
        &mut self,
        model_bytes: &Vec<Vec<u8>>,
        encoding: GraphEncoding,
        target: ExecutionTarget,
    ) -> Result<Graph> {
        let encoding_id: u8 = encoding.into();
        let graph = if let Some(backend) = self.backends.get_mut(&encoding_id) {
            backend.load_from_bytes(model_bytes, target)?
        } else {
            return Err(UsageError::InvalidEncoding(encoding).into());
        };

        let graph_id = self.graphs.insert(graph);
        Ok(graph_id)
    }
}

impl<'a> WasiEphemeralNn for WasiNnCtx {
    fn load<'b>(
        &mut self,
        builders: &GraphBuilderArray<'_>,
        encoding: GraphEncoding,
        target: ExecutionTarget,
    ) -> Result<Graph> {
        let encoding_id: u8 = encoding.into();
        let graph = if let Some(backend) = self.backends.get_mut(&encoding_id) {
            backend.load(builders, target)?
        } else {
            return Err(UsageError::InvalidEncoding(encoding).into());
        };
        let graph_id = self.graphs.insert(graph);
        Ok(graph_id)
    }

    fn load_by_name<'b>(&mut self, model_name: &GuestPtr<'_, [u8]>) -> Result<Graph> {
        let model_name = String::from_utf8(model_name.to_vec().unwrap()).unwrap();
        let maybe_loaded_model = self.loaded_models.get(&model_name);

        match maybe_loaded_model {
            Some(model) => Ok(model.graph),
            None => {
                let registered_model = self.model_registry.get(&model_name).unwrap();
                let model_bytes = &registered_model.model_bytes;
                let encoding: GraphEncoding = registered_model.encoding;
                let target: ExecutionTarget = registered_model.target;

                let encoding_id: u8 = encoding.into();
                let graph = if let Some(backend) = self.backends.get_mut(&encoding_id) {
                    backend.load_from_bytes(model_bytes, target)?
                } else {
                    return Err(UsageError::InvalidEncoding(encoding).into());
                };
                let graph_id = self.graphs.insert(graph);

                Ok(graph_id)
            }
        }
    }

    fn register_model_bytes(
        &mut self,
        model_name: &GuestPtr<'_, [u8]>,
        model_bytes: &GraphBuilderArray<'_>,
        encoding: GraphEncoding,
        target: ExecutionTarget,
    ) -> Result<()> {
        let length: usize = model_bytes.len().try_into().unwrap();
        if length > MAX_GUEST_MODEL_REGISTRATION_SIZE {
            return Err(
                UsageError::ModelTooLarge(length, MAX_GUEST_MODEL_REGISTRATION_SIZE).into(),
            );
        }
        let model_name_bytes = model_name.to_vec().unwrap();
        let mut model_bytes_vec: Vec<Vec<u8>> = Vec::with_capacity(length.try_into().unwrap());
        let mut model_bytes = model_bytes.as_ptr();
        for i in 0..length {
            let v = model_bytes
                .read()?
                .as_slice()?
                .expect("cannot use with shared memories; see https://github.com/bytecodealliance/wasmtime/issues/5235 (TODO)")
                .to_vec();
            model_bytes_vec.push(v);
            model_bytes = model_bytes.add(1)?;
        }
        let model_name_key = String::from_utf8(model_name_bytes).unwrap();
        match target {
            ExecutionTarget::Cpu => {
                let graph = self.build_graph(&model_bytes_vec, encoding, target)?;
                self.loaded_models
                    .insert(model_name_key, LoadedModel { graph });
            }
            _ => {
                self.model_registry.insert(
                    model_name_key,
                    RegisteredModel {
                        model_bytes: model_bytes_vec,
                        encoding,
                        target,
                    },
                );
            }
        };
        Ok(())
    }

    fn unregister(&mut self, model_name: &GuestPtr<'_, [u8]>) -> Result<()> {
        let model_name_bytes = model_name.to_vec().unwrap();
        self.model_registry
            .remove(&String::from_utf8(model_name_bytes).unwrap());
        Ok(())
    }

    fn is_registered(&mut self, model_name: &GuestPtr<'_, [u8]>) -> Result<u32> {
        let model_name_bytes = model_name.to_vec().unwrap();
        if self
            .model_registry
            .contains_key(&String::from_utf8(model_name_bytes).unwrap())
        {
            Ok(1)
        } else {
            Ok(0)
        }
    }

    fn register_model_uri(
        &mut self,
        url: &GuestPtr<'_, [u8]>,
        model_name: &GuestPtr<'_, [u8]>,
        encoding: GraphEncoding,
        target: ExecutionTarget,
    ) -> Result<()> {
        unimplemented!()
    }

    fn init_execution_context(&mut self, graph_id: Graph) -> Result<GraphExecutionContext> {
        let exec_context = if let Some(graph) = self.graphs.get_mut(graph_id) {
            graph.init_execution_context()?
        } else {
            return Err(UsageError::InvalidGraphHandle.into());
        };

        let exec_context_id = self.executions.insert(exec_context);
        Ok(exec_context_id)
    }

    fn set_input<'b>(
        &mut self,
        exec_context_id: GraphExecutionContext,
        index: u32,
        tensor: &Tensor<'b>,
    ) -> Result<()> {
        if let Some(exec_context) = self.executions.get_mut(exec_context_id) {
            Ok(exec_context.set_input(index, tensor)?)
        } else {
            Err(UsageError::InvalidGraphHandle.into())
        }
    }

    fn compute(&mut self, exec_context_id: GraphExecutionContext) -> Result<()> {
        if let Some(exec_context) = self.executions.get_mut(exec_context_id) {
            Ok(exec_context.compute()?)
        } else {
            Err(UsageError::InvalidExecutionContextHandle.into())
        }
    }

    fn get_output<'b>(
        &mut self,
        exec_context_id: GraphExecutionContext,
        index: u32,
        out_buffer: &GuestPtr<'_, u8>,
        out_buffer_max_size: u32,
    ) -> Result<u32> {
        if let Some(exec_context) = self.executions.get_mut(exec_context_id) {
            let mut destination = out_buffer
                .as_array(out_buffer_max_size)
                .as_slice_mut()?
                .expect("cannot use with shared memories; see https://github.com/bytecodealliance/wasmtime/issues/5235 (TODO)");
            Ok(exec_context.get_output(index, &mut destination)?)
        } else {
            Err(UsageError::InvalidGraphHandle.into())
        }
    }
}
