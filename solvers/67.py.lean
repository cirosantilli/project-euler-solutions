import solvers.«67.source»
import data.«project-euler-statements».data.py.project_euler_statements.«p18.py»

namespace solvers.«67»
open FormalSW.PyAst
open ProjectEulerStatements.P18 (reduceRowValues)
set_option Elab.async false

theorem solve.«Returns» (heap : Mutable.Heap) (address : Nat) (_valid : heap.Valid)
    (typed : Mutable.Value.HasType heap (.ref address) "list[list[int]]")
    (requires : HeapCalls.Condition program (triangleEnv address) Solvers.P67Ast.requires_solve_0 heap) :
    (HeapCalls.call program "solve" [.ref address] heap).Returns := by
  obtain ⟨rows, found, reads, shape⟩ := triangle_domain_of_condition heap address typed requires
  obtain ⟨_, returned⟩ := solve.result heap address rows found reads shape
  exact returned.returns

theorem solve.«Equiv(naive)» (heap : Mutable.Heap) (address : Nat) (valid : heap.Valid)
    (typed : Mutable.Value.HasType heap (.ref address) "list[list[int]]")
    (requires : HeapCalls.Condition program (triangleEnv address) Solvers.P67Ast.requires_solve_0 heap) :
    (HeapCalls.call program "solve" [.ref address] heap).Equiv
      (HeapCalls.call P18Ast.module "naive" [.ref address] heap) := by
  obtain ⟨rows, found, reads, shape⟩ := triangle_domain_of_condition heap address typed requires
  obtain ⟨extra, optimized⟩ := solve.result heap address rows found reads shape
  obtain ⟨referenceExtra, reference⟩ :=
    data.«project-euler-statements».data.py.project_euler_statements.p18.naive.result heap address rows found reads
  exact Mutable.Call.equiv_of_finishes optimized.finishes reference.finishes
    (Mutable.Completion.equivalent_allocations heap extra referenceExtra
      (.scalar (.int ((reduceRowValues (rows.map Prod.snd)).headD 0))) [] valid trivial (by simp))

theorem solve.«Supported» :
    (∀ (heap : Mutable.Heap) (address : Nat), heap.Valid →
      Mutable.Value.HasType heap (.ref address) "list[list[int]]" →
      HeapCalls.Condition program (triangleEnv address) Solvers.P67Ast.requires_solve_0 heap →
      (HeapCalls.call program "solve" [.ref address] heap).Supported) ∧
    (∀ (heap : Mutable.Heap) (address : Nat), heap.Valid →
      Mutable.Value.HasType heap (.ref address) "list[list[int]]" →
      HeapCalls.Condition program (triangleEnv address) Solvers.P67Ast.requires_solve_0 heap →
      (HeapCalls.call P18Ast.module "naive" [.ref address] heap).Supported) := by
  constructor
  · intro heap address valid typed requires
    exact (solve.«Returns» heap address valid typed requires).supported
  · intro heap address valid typed _
    exact data.«project-euler-statements».data.py.project_euler_statements.p18.naive.«Supported» heap address valid typed

end solvers.«67»
