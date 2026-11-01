import solvers.«83.py.lean.d».Execution
import data.«project-euler-statements».data.py.project_euler_statements.«p83.py»

namespace solvers.«83»
open FormalSW.PyAst ProjectEulerSolutions.RectangularMatrixCalls
set_option Elab.async false

private theorem nonnegative_values (heap : Mutable.Heap) (address : Nat) (rows : MatrixRows)
    (found : heap[address]? = some (.list (rows.map fun pair => .ref pair.1)))
    (reads : ∀ pair ∈ rows, heap[pair.1]? = some (.list (encode pair.2)))
    (costs : HeapCalls.Condition program (matrixEnv address) Solvers.P83Ast.requires_solve_1 heap) :
    ∀ row ∈ rows.map Prod.snd, ∀ value ∈ row, 0 ≤ value := by
  intro row member value present
  obtain ⟨pair, inRows, rfl⟩ := List.mem_map.mp member
  exact nonnegative_domain heap address rows found reads costs pair inRows value present

theorem solve.«Returns» (heap : Mutable.Heap) (address : Nat) (_valid : heap.Valid)
    (typed : Mutable.Value.HasType heap (.ref address) "list[list[int]]")
    (shape : HeapCalls.Condition program (matrixEnv address) Solvers.P83Ast.requires_solve_0 heap)
    (costs : HeapCalls.Condition program (matrixEnv address) Solvers.P83Ast.requires_solve_1 heap) :
    (HeapCalls.call program "solve" [.ref address] heap).Returns := by
  obtain ⟨rows, found, reads, rectangular, checked⟩ := domain_of_condition shape_source heap address typed shape
  obtain ⟨_, returned⟩ := solve.result heap address rows found reads rectangular
    (nonnegative_values heap address rows found reads costs) checked
  exact returned.returns

theorem solve.«Equiv(naive)» (heap : Mutable.Heap) (address : Nat) (valid : heap.Valid)
    (typed : Mutable.Value.HasType heap (.ref address) "list[list[int]]")
    (shape : HeapCalls.Condition program (matrixEnv address) Solvers.P83Ast.requires_solve_0 heap)
    (costs : HeapCalls.Condition program (matrixEnv address) Solvers.P83Ast.requires_solve_1 heap) :
    (HeapCalls.call program "solve" [.ref address] heap).Equiv
      (HeapCalls.call P83Ast.module "naive" [.ref address] heap) := by
  obtain ⟨rows, found, reads, rectangular, checked⟩ := domain_of_condition shape_source heap address typed shape
  obtain ⟨extra, optimized⟩ := solve.result heap address rows found reads rectangular
    (nonnegative_values heap address rows found reads costs) checked
  obtain ⟨referenceExtra, reference⟩ :=
    data.«project-euler-statements».data.py.project_euler_statements.p83.naive.result heap address rows found reads
  exact Mutable.Call.equiv_of_finishes optimized.finishes reference.finishes
    (Mutable.Completion.equivalent_allocations heap extra referenceExtra
      (.scalar (.int (ProjectEulerStatements.P83.naive (rows.map Prod.snd)))) [] valid trivial (by simp))

theorem solve.«Supported» :
    (∀ (heap : Mutable.Heap) (address : Nat), heap.Valid →
      Mutable.Value.HasType heap (.ref address) "list[list[int]]" →
      HeapCalls.Condition program (matrixEnv address) Solvers.P83Ast.requires_solve_0 heap →
      HeapCalls.Condition program (matrixEnv address) Solvers.P83Ast.requires_solve_1 heap →
      (HeapCalls.call program "solve" [.ref address] heap).Supported) ∧
    (∀ (heap : Mutable.Heap) (address : Nat), heap.Valid →
      Mutable.Value.HasType heap (.ref address) "list[list[int]]" →
      HeapCalls.Condition program (matrixEnv address) Solvers.P83Ast.requires_solve_0 heap →
      HeapCalls.Condition program (matrixEnv address) Solvers.P83Ast.requires_solve_1 heap →
      (HeapCalls.call P83Ast.module "naive" [.ref address] heap).Supported) := by
  constructor
  · intro heap address valid typed shape costs
    exact (solve.«Returns» heap address valid typed shape costs).supported
  · intro heap address _ typed _ _
    obtain ⟨rows, found, reads⟩ := decode_rows heap address typed
    obtain ⟨_, returned⟩ :=
      data.«project-euler-statements».data.py.project_euler_statements.p83.naive.result heap address rows found reads
    exact returned.returns.supported

end solvers.«83»
