import solvers.«93.py.lean.d».RecursiveStep
import ProjectEulerSolutions.Lib.RationalHeapFrames

namespace solvers.«93».all_results
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.RationalPair
open ProjectEulerSolutions ProjectEulerSolutions.IndexedFilter
set_option Elab.async false

/-- Traverse every operation result and union the complete recursively obtained result sets. -/
theorem value_loop (pairs : List Pair) (i j : Nat) (ordered : i < j) (upper : j < pairs.length)
    (canonical : ∀ pair ∈ pairs, Canonical pair) (globals : Mutable.Globals) (smaller : SmallerCalls pairs.length globals)
    (cell memo collector rest operations : Nat) (items operationValues : List Mutable.Value)
    (remaining : List Pair) (pending : List Mutable.Value) (position : Nat)
    (heap : Mutable.Heap) (entries : List Memo.Entry) (collected : List Heap.SetEntry) (env : HeapCalls.Env)
    (cellFound : heap[cell]? = some (.list [.ref memo])) (valid : MemoValid heap memo entries)
    (restFound : heap[rest]? = some (.list items)) (rows : Heap.Rows heap items (select (keepOperand i j) 0 pairs))
    (operationsFound : heap[operations]? = some (.list operationValues)) (suffix : operationValues.drop position = pending)
    (todo : Heap.Rows heap pending remaining) (choicesLeft : ∀ combined ∈ remaining, combined ∈ choices (pairs[i]'(by omega)) pairs[j])
    (original : Heap.SetAt heap collector collected) (uncached : collector ∉ entries.map Memo.Entry.result)
    (getCell : env cellName = some (.ref cell)) (getRest : env "rest" = some (.ref rest))
    (getResult : env "results" = some (.ref collector)) :
    ∃ after finalEnv final finalEntries,
      BlockContinues program env [.sequenceStepLoop "value" (.ref operations) position [.expr updateExpr] []]
        heap finalEnv after globals ∧
      Heap.SetAt after collector final ∧
      (∀ pair, pair ∈ final.map Prod.fst ↔ pair ∈ collected.map Prod.fst ∨
        ∃ combined ∈ remaining, Expected (combined :: select (keepOperand i j) 0 pairs) pair) ∧
      MemoValid after memo finalEntries ∧ Memo.Origins entries heap.length finalEntries ∧
      collector ∉ finalEntries.map Memo.Entry.result ∧ PreservesOutside [memo, collector] heap after ∧
      (∀ name, name ≠ "value" → finalEnv name = env name) := by
  induction remaining generalizing heap entries collected env pending position with
  | nil =>
    cases todo
    refine ⟨heap, env, collected, entries, ?_, original, ?_, valid, Memo.Origins.refl entries heap.length,
      uncached, PreservesOutside.refl _ heap, fun _ _ => rfl⟩
    · apply BlockContinues.sequence_done (items := operationValues)
        (by simp [sequenceValue, Mutable.sequence, operationsFound]) (rest := .nil)
      have atHead := congrArg (fun values : List Mutable.Value => values[0]?) suffix
      simpa using atHead
    · intro pair
      simp
  | cons combined remaining ih =>
    obtain ⟨item, pending, row, tail, rfl⟩ := List.forall₂_cons_right_iff.mp todo
    let bound := setLocal env "value" item
    obtain ⟨middle, next, middleEntries, executed, nextSet, firstMembers, middleMemo, firstOrigins, absent, firstFrame⟩ :=
      recursive_update heap cell memo collector rest items pairs i j ordered upper canonical item combined
        (choicesLeft combined (List.mem_cons_self ..)) entries collected bound globals smaller cellFound valid restFound rows row
        original uncached (by simpa [bound, setLocal, cellName] using getCell)
        (by simpa [bound, setLocal] using getRest) (by simp [bound, setLocal]) (by simpa [bound, setLocal] using getResult)
    have nextCell := firstFrame.ordinary (Memo.encode entries) (collected.map Heap.encodeEntry)
      valid.stored original.keyEntries cell _ cellFound rfl rfl
    have nextRest := firstFrame.ordinary (Memo.encode entries) (collected.map Heap.encodeEntry)
      valid.stored original.keyEntries rest _ restFound rfl rfl
    have nextOperations := firstFrame.ordinary (Memo.encode entries) (collected.map Heap.encodeEntry)
      valid.stored original.keyEntries operations _ operationsFound rfl rfl
    have nextRows := rows.preserved_outside firstFrame (Memo.encode entries) (collected.map Heap.encodeEntry) valid.stored original.keyEntries
    have nextTodo := Heap.Rows.preserved_outside tail firstFrame (Memo.encode entries) (collected.map Heap.encodeEntry) valid.stored original.keyEntries
    have nextSuffix : operationValues.drop (position + 1) = pending := by
      simpa only [List.tail_drop, List.tail_cons] using congrArg List.tail suffix
    obtain ⟨after, finalEnv, final, finalEntries, continued, finalSet, finalMembers, finalMemo, finalOrigins, finalAbsent, finalFrame, localFrame⟩ :=
      ih pending (position + 1) middle middleEntries next bound nextCell middleMemo nextRest nextRows nextOperations nextSuffix nextTodo
        (fun pair member => choicesLeft pair (List.mem_cons_of_mem combined member)) nextSet absent
        (by simpa [bound, setLocal, cellName] using getCell) (by simpa [bound, setLocal] using getRest)
        (by simpa [bound, setLocal] using getResult)
    refine ⟨after, finalEnv, final, finalEntries, ?_, finalSet, ?_, finalMemo,
      firstOrigins.trans finalOrigins firstFrame.size, finalAbsent, firstFrame.trans finalFrame, ?_⟩
    · apply BlockContinues.sequence_step (items := operationValues) (value := item)
        (by simp [sequenceValue, Mutable.sequence, operationsFound])
      · have atHead := congrArg (fun values : List Mutable.Value => values[0]?) suffix
        simpa using atHead
      · simpa only [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
          Bool.false_or, Bool.false_eq_true, if_false] using executed.append continued
    · intro pair
      rw [finalMembers, firstMembers]
      simp only [List.mem_cons, or_and_right, exists_or, exists_eq_left]
      tauto
    · intro name different
      rw [localFrame name different]
      simp [bound, setLocal, different]

end solvers.«93».all_results
