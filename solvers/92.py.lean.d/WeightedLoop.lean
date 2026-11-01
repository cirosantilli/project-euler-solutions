import solvers.«92.py.lean.d».WeightedRow
import ProjectEulerSolutions.Lib.DigitSquareDistribution

namespace solvers.«92».weighted
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.DigitSquareChains
set_option Elab.async false

/-- The sum generator's live dictionary cursor runs the original filter and nested binding. -/
def cursor (address : Nat) (entries : table.Entries) (position : Nat) : Stmt :=
  .dictionaryItemsStepLoop "$formalsw.unpack" address (traversal.keys entries) position body []

def weight (entries : table.Entries) : Nat :=
  Prefix.weighted (table.view entries) (fun state => if endsAt89 state.1 then 1 else 0)

theorem weight_cons (entry : table.Entry) (rest : table.Entries) :
    weight (entry :: rest) = contribution entry.1 entry.2.2 + weight rest := by
  cases hit : endsAt89 entry.1.1 <;>
    simp [weight, Prefix.weighted, table.view, contribution, hit]

/-- Every remaining dictionary entry is visited once; cache writes preserve all counts and key tuples. -/
theorem scan (heap : Mutable.Heap) (before : HeapCalls.Env) (globals : Mutable.Globals)
    (sourceAddress cache accumulated position : Nat) (original : table.Entries) (cached : store.Entries)
    (known : before "known" = some (.ref cache))
    (sum : before sumAccumulator = some (number accumulated))
    (sourceFound : table.Found heap sourceAddress original) (sourceKeys : table.Keys heap original)
    (found : store.Found heap cache cached) (valid : Memo.Valid (store.lookup cached))
    (different : sourceAddress ≠ cache) :
    ∃ result finalHeap finalLocals,
      BlockContinues program before [cursor sourceAddress original position] heap finalLocals finalHeap globals ∧
      store.Found finalHeap cache result ∧ Memo.Valid (store.lookup result) ∧
      finalLocals "known" = some (.ref cache) ∧
      finalLocals sumAccumulator = some (number (accumulated + weight (original.drop position))) ∧
      rows.HeapFrame heap finalHeap cache := by
  by_cases active : position < original.length
  · let entry := original[position]
    have present : original[position]? = some entry := List.getElem?_eq_getElem active
    obtain ⟨updated, middle, locals, rowRan, middleFound, middleValid, middleKnown, middleSum, rowFrame⟩ :=
      row_result heap before globals cache accumulated entry cached known sum
        (sourceKeys entry (List.getElem_mem active)) found valid
    obtain ⟨result, finalHeap, finalLocals, rest, finalFound, finalValid, finalKnown, finalSum, frame⟩ :=
      scan middle locals globals sourceAddress cache (accumulated + contribution entry.1 entry.2.2)
        (position + 1) original updated middleKnown middleSum
        (traversal.found_frame heap middle sourceAddress cache original sourceFound different rowFrame)
        (keys_frame heap middle cache original cached sourceKeys found rowFrame) middleFound middleValid different
    refine ⟨result, finalHeap, finalLocals, ?_, finalFound, finalValid, finalKnown, ?_, ?_⟩
    · apply BlockContinues.dictionary_items_step
        (key := .ref entry.2.1) (value := number entry.2.2)
        (middle := heap) (afterTuple := traversal.pairHeap heap entry)
        (by simpa only [present, Option.map_some] using traversal.next heap sourceAddress position original sourceFound)
        (traversal.allocated heap entry)
      simpa only [show Calls.loopContinuation body (cursor sourceAddress original (position + 1)) =
        cursor sourceAddress original (position + 1) from rfl] using rowRan.append rest
    · rw [List.drop_eq_getElem_cons active, weight_cons]
      simpa only [entry, Nat.add_assoc] using finalSum
    · intro other object excluded initial
      exact frame other object excluded (rowFrame other object excluded initial)
  · have exhausted : original.length ≤ position := by omega
    refine ⟨cached, heap, before, ?_, found, valid, known, ?_, ?_⟩
    · exact BlockContinues.dictionary_items_done
        (by simpa only [List.getElem?_eq_none exhausted, Option.map_none] using
          traversal.next heap sourceAddress position original sourceFound) BlockContinues.nil
    · simpa [List.drop_eq_nil_of_le exhausted, weight, table.view, Prefix.weighted] using sum
    · intro other object _ present
      exact present
termination_by original.length - position
decreasing_by omega

/-- With an exact prefix distribution, the completed cursor returns the reference count. -/
theorem result (heap : Mutable.Heap) (before : HeapCalls.Env) (globals : Mutable.Globals)
    (bound sourceAddress cache : Nat) (original : table.Entries) (cached : store.Entries)
    (known : before "known" = some (.ref cache))
    (sum : before sumAccumulator = some (number 0))
    (sourceFound : table.Found heap sourceAddress original) (sourceKeys : table.Keys heap original)
    (found : store.Found heap cache cached) (valid : Memo.Valid (store.lookup cached))
    (different : sourceAddress ≠ cache) (exact : Prefix.Exact bound (table.view original)) :
    ∃ result finalHeap,
      BlockReturns program before [cursor sourceAddress original 0, .return (.name sumAccumulator)]
        heap (number (count (bound + 1 : Nat))) finalHeap globals ∧
      store.Found finalHeap cache result ∧ Memo.Valid (store.lookup result) ∧
      rows.HeapFrame heap finalHeap cache := by
  obtain ⟨result, finalHeap, finalLocals, ran, finalFound, finalValid, _, finalSum, frame⟩ :=
    scan heap before globals sourceAddress cache 0 0 original cached known sum
      sourceFound sourceKeys found valid different
  refine ⟨result, finalHeap, ?_, finalFound, finalValid, frame⟩
  apply ran.returns
  apply BlockReturns.return_value
  apply Evaluates.lookup
  simpa [weight, exact.answer] using finalSum

end solvers.«92».weighted
