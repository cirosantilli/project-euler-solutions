import solvers.«84.py.lean.d».TransitionItemsEntry

namespace solvers.«84».transition_row.items
open FormalSW.PyAst HeapCalls ProjectEulerSolutions NaturalDictionary resolve_landing
set_option Elab.async false

/-- Traverse the complete live dictionary without a fuel bound or an assumed execution result. -/
theorem scan (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (source root doubles count position : Nat) (isDouble : Bool) (original : List weights.Entry) (entries : Entries)
    (getRow : env "row" = some (.ref root)) (getDoubles : env "doubles" = some (number doubles))
    (getCount : env "count" = some (number count)) (getDouble : env "is_double" = some (.scalar (.bool isDouble)))
    (sourceFound : heap[source]? = some (.keyDict (weights.encode original)))
    (valid : weights.Valid heap original) (found : Found heap root entries)
    (nonnegative : ∀ entry ∈ original, 0 ≤ entry.landing.1 ∧ 0 ≤ entry.weight) :
    ∃ finalHeap finalEnv,
      BlockContinues program env [cursor source original position] heap finalEnv finalHeap globals ∧
      Found finalHeap root ((original.drop position).foldl (applyEntry doubles count isDouble) entries) ∧
      HeapFrame heap finalHeap root ∧ LocalsFrame env finalEnv := by
  by_cases active : position < original.length
  · let entry := original[position]
    have present : original[position]? = some entry := List.getElem?_eq_getElem active
    have member : entry ∈ original := List.getElem_mem active
    obtain ⟨middle, middleEnv, ran, middleFound, frame, localFrame⟩ :=
      entry_result heap env globals root doubles count isDouble entry entries found getRow getDoubles getCount getDouble
        (valid entry member) (nonnegative entry member).1 (nonnegative entry member).2
    have different : source ≠ root := by
      intro same
      subst source
      change heap[root]? = some (.dict (encode entries)) at found
      rw [sourceFound] at found
      cases found
    obtain ⟨finalHeap, finalEnv, continued, finalFound, finalFrame, finalLocals⟩ :=
      scan middle middleEnv globals source root doubles count (position + 1) isDouble original
        (applyEntry doubles count isDouble entries entry)
        ((localFrame _ (by decide)).trans getRow) ((localFrame _ (by decide)).trans getDoubles)
        ((localFrame _ (by decide)).trans getCount) ((localFrame _ (by decide)).trans getDouble)
        (frame _ _ different sourceFound) (keys_frame frame entries found original valid) middleFound nonnegative
    refine ⟨finalHeap, finalEnv, ?_, ?_, ?_, ?_⟩
    · apply BlockContinues.dictionary_items_step
        (key := .ref entry.address) (value := .scalar (.int entry.weight))
        (middle := heap) (afterTuple := pairHeap heap entry)
        (by simpa only [present, Option.map_some] using next heap source position original sourceFound)
        (allocated heap entry)
      simpa only [show Calls.loopContinuation body (cursor source original (position + 1)) =
        cursor source original (position + 1) from rfl] using ran.append continued
    · simpa only [List.drop_eq_getElem_cons active, List.foldl_cons, entry] using finalFound
    · intro address object excluded initial
      exact finalFrame address object excluded (frame address object excluded initial)
    · intro name keep
      exact (finalLocals name keep).trans (localFrame name keep)
  · have exhausted : original.length ≤ position := by omega
    refine ⟨heap, env, ?_, ?_, ?_, ?_⟩
    · exact BlockContinues.dictionary_items_done
        (by simpa only [List.getElem?_eq_none exhausted, Option.map_none] using next heap source position original sourceFound)
        BlockContinues.nil
    · simpa only [List.drop_eq_nil_of_le exhausted, List.foldl_nil] using found
    · intro address object _ present
      exact present
    · intro name _
      rfl
termination_by original.length - position
decreasing_by omega

theorem result (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (source root doubles count : Nat) (isDouble : Bool) (original : List weights.Entry) (entries : Entries)
    (getRow : env "row" = some (.ref root)) (getDoubles : env "doubles" = some (number doubles))
    (getCount : env "count" = some (number count)) (getDouble : env "is_double" = some (.scalar (.bool isDouble)))
    (evaluated : Evaluates program env receiver heap (.ref source) heap globals)
    (sourceFound : heap[source]? = some (.keyDict (weights.encode original)))
    (valid : weights.Valid heap original) (found : Found heap root entries)
    (nonnegative : ∀ entry ∈ original, 0 ≤ entry.landing.1 ∧ 0 ≤ entry.weight) :
    ∃ finalHeap finalEnv,
      BlockContinues program env landingCode heap finalEnv finalHeap globals ∧
      Found finalHeap root (original.foldl (applyEntry doubles count isDouble) entries) ∧
      HeapFrame heap finalHeap root ∧ LocalsFrame env finalEnv := by
  obtain ⟨finalHeap, finalEnv, ran, finalFound, frame, localFrame⟩ :=
    scan heap env globals source root doubles count 0 isDouble original entries getRow getDoubles getCount getDouble
      sourceFound valid found nonnegative
  refine ⟨finalHeap, finalEnv, ?_, by simpa only [List.drop_zero] using finalFound, frame, localFrame⟩
  · rw [items.source]
    apply BlockContinues.for_dictionary_items evaluated sourceFound (represented original)
    simpa [cursor, keys, fields, List.map_map, Function.comp_def, sequenceSource,
      comprehensionTargetName, comprehensionTargetBody, bindings.target, body, bindings.body] using ran

end solvers.«84».transition_row.items
