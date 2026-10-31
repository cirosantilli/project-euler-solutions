import solvers.«88.py.lean.d».Search
import lang.py.FormalSW.SequenceArithmeticRelocation

namespace solvers.«88».solve
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.ProductSumNumbers
set_option Elab.async false

abbrev code : List Stmt := executableBody ((findFunctionIn program.body "solve").getD ([], [])).2
def context (heap : Mutable.Heap) (maxSize : Nat) : Context where
  base := heap ++ [.list [number maxSize], .list [number (2 * maxSize)], .list [number (2 * maxSize + 1)]]
  tail := [.list [.ref (heap.length + 3)]]
  bestCell := heap.length + 4
  maxCell := heap.length
  limitCell := heap.length + 1

theorem context_cells (heap : Mutable.Heap) (maxSize : Nat) :
    Cells (context heap maxSize) maxSize (2 * maxSize) := by
  constructor <;> intro best <;>
    simp [context, working, ProjectEulerSolutions.HeapCollection.working, List.append_assoc]

def preparedEnv (env : HeapCalls.Env) (heap : Mutable.Heap) (maxSize : Nat) : HeapCalls.Env :=
  setLocal (setLocal (setLocal (setLocal env "__formalsw_cell_solve_k_max" (.ref heap.length))
    "__formalsw_cell_solve_limit" (.ref (heap.length + 1))) "sentinel" (number (2 * maxSize + 1)))
    "__formalsw_cell_solve_best" (.ref (heap.length + 4))

/-- Allocate the sentinel singleton, repeated table, and its shared capture cell in source order. -/
theorem allocate_best (maxSize : Nat) (heap : Mutable.Heap) (cell : Nat) (env : HeapCalls.Env)
    (getMax : env "__formalsw_cell_solve_k_max" = some (.ref cell))
    (foundMax : heap[cell]? = some (.list [number maxSize]))
    (getSentinel : env "sentinel" = some (number (2 * maxSize + 1))) :
    Evaluates program env
      (.list [.binOp (.list [.name "sentinel"]) .mult
        (.binOp (.subscript (.name "__formalsw_cell_solve_k_max") (.constant (.int 0))) .add (.constant (.int 1)))])
      heap (.ref (heap.length + 2))
      (heap ++ [.list [number (2 * maxSize + 1)],
        .list (encode (List.replicate (maxSize + 1) (2 * maxSize + 1))), .list [.ref (heap.length + 1)]]) := by
  let singletonHeap := heap ++ [.list [number (2 * maxSize + 1)]]
  have repeated : Evaluates program env
      (.binOp (.list [.name "sentinel"]) .mult
        (.binOp (.subscript (.name "__formalsw_cell_solve_k_max") (.constant (.int 0))) .add (.constant (.int 1))))
      heap (.ref (heap.length + 1))
      (heap ++ [.list [number (2 * maxSize + 1)], .list (encode (List.replicate (maxSize + 1) (2 * maxSize + 1)))]) := by
    apply Evaluates.binary (av := .ref heap.length) (bv := number (maxSize + 1))
      (middle := singletonHeap) (afterRight := singletonHeap)
    · exact Evaluates.list (.cons (Evaluates.lookup getSentinel) .nil)
    · exact Evaluates.binary (av := number maxSize) (bv := number 1)
        (captured_number singletonHeap cell maxSize "__formalsw_cell_solve_k_max" env getMax
          (ProjectEulerSolutions.HeapCollection.prefix_found heap _ cell _ foundMax)) Evaluates.constant
        (by simp [Mutable.binary, scalarBinary, number])
    · have multiplied := Mutable.Relocation.binary_mult_list_eval singletonHeap heap.length
        [number (2 * maxSize + 1)] (maxSize + 1) (by simp [singletonHeap])
      simpa [Mutable.allocate, singletonHeap, encode, List.map_replicate, List.append_assoc] using multiplied
  simpa [List.append_assoc] using (Evaluates.list (.cons repeated .nil))

end solvers.«88».solve
