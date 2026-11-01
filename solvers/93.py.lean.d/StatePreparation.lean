import solvers.«93.py.lean.d».Accumulation
import ProjectEulerSolutions.Lib.RationalSorting
import lang.py.FormalSW.HeapSortingCalls

namespace solvers.«93».all_results
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.RationalPair
open ProjectEulerSolutions.HeapCollection
set_option Elab.async false

/-- Build the recursive tuple by the source's list display, concatenation, sorting and conversion. -/
theorem recursive_state (heap : Mutable.Heap) (rest : Nat) (items : List Mutable.Value)
    (pairs : List Pair) (item : Mutable.Value) (pair : Pair) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (found : heap[rest]? = some (.list items)) (rows : Heap.Rows heap items pairs) (row : Heap.Row heap item pair)
    (getRest : env "rest" = some (.ref rest)) (getValue : env "value" = some item) :
    ∃ extra state sortedItems sortedPairs,
      Evaluates program env recursiveStateExpr heap (.ref state) (heap ++ extra) globals ∧
      (heap ++ extra)[state]? = some (.tuple sortedItems) ∧
      Heap.Rows (heap ++ extra) sortedItems sortedPairs ∧
      sortedPairs.Perm (pairs ++ [pair]) ∧ sortedItems.Perm (items ++ [item]) := by
  let displayed := heap ++ [Mutable.Object.list [item]]
  let joined := displayed ++ [Mutable.Object.list (items ++ [item])]
  have displayedRest : displayed[rest]? = some (.list items) :=
    prefix_found heap [Mutable.Object.list [item]] rest _ found
  have displayedItem : displayed[heap.length]? = some (.list [item]) := by simp [displayed]
  have joinedFound : joined[displayed.length]? = some (.list (items ++ [item])) := by simp [joined]
  have combinedRows : Heap.Rows heap (items ++ [item]) (pairs ++ [pair]) := List.rel_append rows (.cons row .nil)
  have joinedRows : Heap.Rows joined (items ++ [item]) (pairs ++ [pair]) :=
    (combinedRows.append [.list [item]]).append [.list (items ++ [item])]
  obtain ⟨sortedItems, sortedPairs, sorted, sortedRows, pairPermutation, itemPermutation⟩ := joinedRows.sort
  let ordered := joined ++ [Mutable.Object.list sortedItems]
  let final := ordered ++ [Mutable.Object.tuple sortedItems]
  have orderedFound : ordered[joined.length]? = some (.list sortedItems) := by simp [ordered]
  have nonempty : sortedItems ≠ [] := by
    intro empty
    have length := itemPermutation.length_eq
    simp [empty] at length
  have concatenated : Evaluates program env (.binOp (.name "rest") .add (.list [.name "value"]))
      heap (.ref displayed.length) joined globals := by
    apply Evaluates.binary (Evaluates.lookup getRest) (Evaluates.list (.cons (Evaluates.lookup getValue) .nil))
    change Mutable.binary .add (.ref rest) (.ref heap.length) displayed = some (.ok (.ref displayed.length), joined)
    simp [Mutable.binary, show (BinOp.add == BinOp.add) = true from rfl,
      displayedRest, displayedItem, Mutable.array, Mutable.allocate, joined, Except.map]
  have sortExpression : Evaluates program env
      (.builtinCall .sorted [.binOp (.name "rest") .add (.list [.name "value"])])
      heap (.ref joined.length) ordered globals := by
    exact Evaluates.builtin (.cons concatenated .nil)
      (primitive_sorted_values joined displayed.length _ sortedItems joinedFound sorted)
  have executed : Evaluates program env recursiveStateExpr heap (.ref ordered.length) final globals := by
    apply Evaluates.builtin (.cons sortExpression .nil)
    simp [primitive, orderedFound, Mutable.sequence, Mutable.allocateTuple, nonempty, final, Except.map]
  refine ⟨[.list [item], .list (items ++ [item]), .list sortedItems, .tuple sortedItems],
    ordered.length, sortedItems, sortedPairs, ?_, ?_, ?_, pairPermutation, itemPermutation⟩
  · simpa only [final, ordered, joined, displayed, List.append_assoc, List.cons_append, List.nil_append] using executed
  · have finalFound : final[ordered.length]? = some (.tuple sortedItems) := by simp [final]
    simpa only [final, ordered, joined, displayed, List.append_assoc, List.cons_append, List.nil_append] using finalFound
  · simpa only [ordered, joined, displayed, List.append_assoc, List.cons_append, List.nil_append] using
      (sortedRows.append [.list sortedItems]).append [.tuple sortedItems]

end solvers.«93».all_results
