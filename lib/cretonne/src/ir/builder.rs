//! Cretonne instruction builder.
//!
//! A `Builder` provides a convenient interface for inserting instructions into a Cretonne
//! function. Many of its methods are generated from the meta language instruction definitions.

use ir::{types, instructions};
use ir::{InstructionData, DataFlowGraph, Cursor};
use ir::{Opcode, Type, Inst, Value, Ebb, JumpTable, VariableArgs, SigRef, FuncRef};
use ir::immediates::{Imm64, Uimm8, Ieee32, Ieee64, ImmVector};
use ir::condcodes::{IntCC, FloatCC};

/// Base trait for instruction builders.
///
/// The `InstBuilderBase` trait provides the basic functionality required by the methods of the
/// generated `InstBuilder` trait. These methods should not normally be used directly. Use the
/// methods in the `InstBuilder trait instead.
///
/// Any data type that implements `InstBuilderBase` also gets all the methods of the `InstBuilder`
/// trait.
pub trait InstBuilderBase<'f>: Sized {
    /// Get an immutable reference to the data flow graph that will hold the constructed
    /// instructions.
    fn data_flow_graph(&self) -> &DataFlowGraph;

    /// Insert a simple instruction and return a reference to it.
    ///
    /// A 'simple' instruction has at most one result, and the `data.ty` field must contain the
    /// result type or `VOID` for an instruction with no result values.
    fn simple_instruction(self, data: InstructionData) -> (Inst, &'f mut DataFlowGraph);

    /// Insert a simple instruction and return a reference to it.
    ///
    /// A 'complex' instruction may produce multiple results, and the result types may depend on a
    /// controlling type variable. For non-polymorphic instructions with multiple results, pass
    /// `VOID` for the `ctrl_typevar` argument.
    fn complex_instruction(self,
                           data: InstructionData,
                           ctrl_typevar: Type)
                           -> (Inst, &'f mut DataFlowGraph);
}

// Include trait code generated by `meta/gen_instr.py`.
//
// This file defines the `InstBuilder` trait as an extension of `InstBuilderBase` with methods per
// instruction format and per opcode.
include!(concat!(env!("OUT_DIR"), "/builder.rs"));

/// Any type implementing `InstBuilderBase` gets all the `InstBuilder` methods for free.
impl<'f, T: InstBuilderBase<'f>> InstBuilder<'f> for T {}

/// Instruction builder.
///
/// A `Builder` holds mutable references to a data flow graph and a layout cursor. It provides
/// convenience method for creating and inserting instructions at the current cursor position.
pub struct Builder<'c, 'fc: 'c, 'fd> {
    pub pos: &'c mut Cursor<'fc>,
    pub dfg: &'fd mut DataFlowGraph,
}

impl<'c, 'fc, 'fd> Builder<'c, 'fc, 'fd> {
    /// Create a new builder which inserts instructions at `pos`.
    /// The `dfg` and `pos.layout` references should be from the same `Function`.
    pub fn new(dfg: &'fd mut DataFlowGraph, pos: &'c mut Cursor<'fc>) -> Builder<'c, 'fc, 'fd> {
        Builder {
            dfg: dfg,
            pos: pos,
        }
    }

    /// Create and insert an EBB. Further instructions will be inserted into the new EBB.
    pub fn ebb(&mut self) -> Ebb {
        let ebb = self.dfg.make_ebb();
        self.insert_ebb(ebb);
        ebb
    }

    /// Insert an existing EBB at the current position. Further instructions will be inserted into
    /// the new EBB.
    pub fn insert_ebb(&mut self, ebb: Ebb) {
        self.pos.insert_ebb(ebb);
    }
}

impl<'c, 'fc, 'fd> InstBuilderBase<'fd> for Builder<'c, 'fc, 'fd> {
    fn data_flow_graph(&self) -> &DataFlowGraph {
        self.dfg
    }

    fn simple_instruction(self, data: InstructionData) -> (Inst, &'fd mut DataFlowGraph) {
        let inst = self.dfg.make_inst(data);
        self.pos.insert_inst(inst);
        (inst, self.dfg)
    }

    fn complex_instruction(self,
                           data: InstructionData,
                           ctrl_typevar: Type)
                           -> (Inst, &'fd mut DataFlowGraph) {
        let inst = self.dfg.make_inst(data);
        self.dfg.make_inst_results(inst, ctrl_typevar);
        self.pos.insert_inst(inst);
        (inst, self.dfg)
    }
}
