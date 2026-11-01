import solvers.«81.py.lean.d».Execution
import data.«project-euler-statements».data.py.project_euler_statements.«p81.py»

namespace solvers.«81»
open FormalSW.PyAst ProjectEulerSolutions.RectangularMatrixCalls
set_option Elab.async false

theorem solve.«Returns» (heap : Mutable.Heap) (address : Nat) (_valid : heap.Valid)
    (typed : Mutable.Value.HasType heap (.ref address) "list[list[int]]")
    (requires : HeapCalls.Condition program (matrixEnv address) Solvers.P81Ast.requires_solve_0 heap) :
    (HeapCalls.call program "solve" [.ref address] heap).Returns := by
  obtain ⟨rows, found, reads, shape, checked⟩ := domain_of_condition shape_source heap address typed requires
  obtain ⟨_, returned⟩ := solve.result heap address rows found reads shape checked
  exact returned.returns

theorem solve.«Equiv(naive)» (heap : Mutable.Heap) (address : Nat) (valid : heap.Valid)
    (typed : Mutable.Value.HasType heap (.ref address) "list[list[int]]")
    (requires : HeapCalls.Condition program (matrixEnv address) Solvers.P81Ast.requires_solve_0 heap) :
    (HeapCalls.call program "solve" [.ref address] heap).Equiv
      (HeapCalls.call P81Ast.module "naive" [.ref address] heap) := by
  obtain ⟨rows, found, reads, shape, checked⟩ := domain_of_condition shape_source heap address typed requires
  obtain ⟨extra, optimized⟩ := solve.result heap address rows found reads shape checked
  obtain ⟨referenceExtra, reference⟩ :=
    data.«project-euler-statements».data.py.project_euler_statements.p81.naive.result heap address rows found reads shape
  exact Mutable.Call.equiv_of_finishes optimized.finishes reference.finishes
    (Mutable.Completion.equivalent_allocations heap extra referenceExtra
      (.scalar (.int (ProjectEulerStatements.P81.naive (rows.map Prod.snd)))) [] valid trivial (by simp))

theorem solve.«Supported» :
    (∀ (heap : Mutable.Heap) (address : Nat), heap.Valid →
      Mutable.Value.HasType heap (.ref address) "list[list[int]]" →
      HeapCalls.Condition program (matrixEnv address) Solvers.P81Ast.requires_solve_0 heap →
      (HeapCalls.call program "solve" [.ref address] heap).Supported) ∧
    (∀ (heap : Mutable.Heap) (address : Nat), heap.Valid →
      Mutable.Value.HasType heap (.ref address) "list[list[int]]" →
      HeapCalls.Condition program (matrixEnv address) Solvers.P81Ast.requires_solve_0 heap →
      (HeapCalls.call P81Ast.module "naive" [.ref address] heap).Supported) := by
  constructor
  · intro heap address valid typed requires
    exact (solve.«Returns» heap address valid typed requires).supported
  · intro heap address _ typed requires
    obtain ⟨rows, found, reads, shape, _⟩ := domain_of_condition shape_source heap address typed requires
    obtain ⟨_, returned⟩ :=
      data.«project-euler-statements».data.py.project_euler_statements.p81.naive.result heap address rows found reads shape
    exact returned.returns.supported

end solvers.«81»
