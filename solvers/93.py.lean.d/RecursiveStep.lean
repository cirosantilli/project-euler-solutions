import solvers.«93.py.lean.d».ReductionInput
import solvers.«93.py.lean.d».MemoAccumulation
import ProjectEulerSolutions.Lib.HeapPreservesOutside

namespace solvers.«93».all_results
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.RationalPair
open ProjectEulerSolutions ProjectEulerSolutions.IndexedFilter
set_option Elab.async false

/-- The induction hypothesis ranges over every state with fewer operands than the current state. -/
def SmallerCalls (size : Nat) (globals : Mutable.Globals) : Prop :=
  ∀ (heap : Mutable.Heap) (cell memo state : Nat) (values : List Mutable.Value) (pairs : List Pair) (entries : List Memo.Entry),
    pairs.length < size → heap[cell]? = some (.list [.ref memo]) → MemoValid heap memo entries →
    heap[state]? = some (.tuple values) → Heap.Rows heap values pairs → (∀ pair ∈ pairs, Canonical pair) →
    RecursiveVerified heap cell memo state pairs entries globals

/-- The complete innermost source statement constructs, recursively solves, and unions one smaller state. -/
theorem recursive_update (heap : Mutable.Heap) (cell memo collector rest : Nat)
    (items : List Mutable.Value) (pairs : List Pair) (i j : Nat) (ordered : i < j) (upper : j < pairs.length)
    (canonical : ∀ pair ∈ pairs, Canonical pair) (item : Mutable.Value) (combined : Pair)
    (chosen : combined ∈ choices (pairs[i]'(by omega)) pairs[j])
    (entries : List Memo.Entry) (collected : List Heap.SetEntry) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (smaller : SmallerCalls pairs.length globals) (cellFound : heap[cell]? = some (.list [.ref memo]))
    (valid : MemoValid heap memo entries) (found : heap[rest]? = some (.list items))
    (rows : Heap.Rows heap items (select (keepOperand i j) 0 pairs)) (row : Heap.Row heap item combined)
    (original : Heap.SetAt heap collector collected) (uncached : collector ∉ entries.map Memo.Entry.result)
    (getCell : env cellName = some (.ref cell)) (getRest : env "rest" = some (.ref rest))
    (getValue : env "value" = some item) (getResult : env "results" = some (.ref collector)) :
    ∃ after final finalEntries,
      BlockContinues program env [.expr updateExpr] heap env after globals ∧
      Heap.SetAt after collector final ∧
      (∀ pair, pair ∈ final.map Prod.fst ↔ pair ∈ collected.map Prod.fst ∨
        Expected (combined :: select (keepOperand i j) 0 pairs) pair) ∧
      MemoValid after memo finalEntries ∧ Memo.Origins entries heap.length finalEntries ∧
      collector ∉ finalEntries.map Memo.Entry.result ∧ PreservesOutside [memo, collector] heap after := by
  obtain ⟨extra, state, sortedItems, sortedPairs, prepared, stateFound, stateRows, permutation, shorter, stateCanonical, _⟩ :=
    replacement_state heap rest items pairs i j ordered upper canonical item combined chosen env globals found rows row getRest getValue
  have preparedCell := HeapCollection.prefix_found heap extra cell _ cellFound
  have preparedMemo := valid.append extra
  obtain ⟨middle, result, finalEntries, returned, post⟩ :=
    smaller (heap ++ extra) cell memo state sortedItems sortedPairs entries shorter preparedCell preparedMemo stateFound stateRows stateCanonical
  have recursive : Evaluates program env recursiveResultExpr heap (.ref result) middle globals :=
    Evaluates.call (.cons prepared (.cons (Evaluates.lookup getCell) .nil)) returned
  have retained : Heap.SetAt middle collector collected :=
    (original.append extra).preserved post.frame (Memo.encode entries) preparedMemo.stored
  have absent := post.collector_uncached uncached (by
    have inside := original.bound
    simp only [List.length_append]
    omega)
  obtain ⟨after, final, executed, finalSet, members, finalMemo, unionFrame⟩ :=
    update_valid heap middle collector result memo collected sortedPairs finalEntries env globals
      original getResult recursive retained post.complete post.valid absent
  have callFrame := (PreservesExcept.append memo heap extra).trans post.frame
  refine ⟨after, final, finalEntries, executed, finalSet, ?_, finalMemo, ?_, absent, ?_⟩
  · intro pair
    exact (members pair).trans (or_congr Iff.rfl (expected_perm permutation pair))
  · exact (Memo.Origins.refl entries heap.length).trans post.origins (by simp)
  · exact (PreservesOutside.of_except callFrame (by simp)).trans (PreservesOutside.of_except unionFrame (by simp))

end solvers.«93».all_results
