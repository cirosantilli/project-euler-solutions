import solvers.«93.py.lean.d».Remainder
import ProjectEulerSolutions.Lib.RationalReduction

namespace solvers.«93».all_results
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.RationalPair
open ProjectEulerSolutions.IndexedFilter ProjectEulerSolutions.ArithmeticExpressions
set_option Elab.async false

/-- Each actual recursive argument is a smaller canonical state with only valid arithmetic results. -/
theorem replacement_state (heap : Mutable.Heap) (rest : Nat) (items : List Mutable.Value)
    (pairs : List Pair) (i j : Nat) (ordered : i < j) (upper : j < pairs.length)
    (canonical : ∀ pair ∈ pairs, Canonical pair) (item : Mutable.Value) (combined : Pair)
    (chosen : combined ∈ choices (pairs[i]'(by omega)) pairs[j])
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (found : heap[rest]? = some (.list items))
    (rows : Heap.Rows heap items (select (keepOperand i j) 0 pairs)) (row : Heap.Row heap item combined)
    (getRest : env "rest" = some (.ref rest)) (getValue : env "value" = some item) :
    ∃ extra state sortedItems sortedPairs,
      Evaluates program env recursiveStateExpr heap (.ref state) (heap ++ extra) globals ∧
      (heap ++ extra)[state]? = some (.tuple sortedItems) ∧
      Heap.Rows (heap ++ extra) sortedItems sortedPairs ∧
      sortedPairs.Perm (combined :: select (keepOperand i j) 0 pairs) ∧
      sortedPairs.length < pairs.length ∧
      (∀ pair ∈ sortedPairs, Canonical pair) ∧
      (∀ answer, Reduces (sortedPairs.map value) answer → Reduces (pairs.map value) answer) := by
  obtain ⟨extra, state, sortedItems, sortedPairs, executed, finalState, finalRows, pairPermutation, _⟩ :=
    recursive_state heap rest items (select (keepOperand i j) 0 pairs) item combined env globals found rows row getRest getValue
  have permutation : sortedPairs.Perm (combined :: select (keepOperand i j) 0 pairs) :=
    pairPermutation.trans (by simp)
  have positive : ∀ pair ∈ pairs, 0 < pair.2 := fun pair member => (canonical pair member).positive
  have inside : i < pairs.length := by omega
  have combinedCanonical := choices_canonical pairs[i] pairs[j]
    (positive pairs[i] (List.getElem_mem inside)) (positive pairs[j] (List.getElem_mem upper)) combined chosen
  refine ⟨extra, state, sortedItems, sortedPairs, executed, finalState, finalRows, permutation, ?_, ?_, ?_⟩
  · calc
      sortedPairs.length = (select (keepOperand i j) 0 pairs).length + 1 := permutation.length_eq
      _ < pairs.length := pair_replacement_shorter pairs i j ordered upper
  · intro pair member
    rcases List.mem_cons.mp (permutation.mem_iff.mp member) with rfl | remaining
    · exact combinedCanonical
    · exact canonical pair ((sublist (keepOperand i j) 0 pairs).subset remaining)
  · intro answer derived
    exact (reduces_iff_index_pair pairs answer positive (by omega)).mpr
      ⟨i, j, ordered, upper, combined, chosen, derived.perm (permutation.map value)⟩

end solvers.«93».all_results
