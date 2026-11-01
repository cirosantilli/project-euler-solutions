import solvers.«92.py.lean.d».TraversalRow

namespace solvers.«92».traversal
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.DigitSquareChains
set_option Elab.async false

/-- Consume the remaining dictionary entries, collecting every row's contributions. -/
theorem scan (heap : Mutable.Heap) (before : HeapCalls.Env) (globals : Mutable.Globals)
    (sourceAddress destination ceiling position : Nat) (original accumulated : table.Entries)
    (following : before "following" = some (.ref destination))
    (limit : before "ceiling" = some (table.number ceiling))
    (sourceFound : table.Found heap sourceAddress original) (sourceKeys : table.Keys heap original)
    (found : table.Found heap destination accumulated) (valid : table.Keys heap accumulated)
    (different : sourceAddress ≠ destination) :
    ∃ result finalHeap finalLocals,
      BlockContinues program before [cursor sourceAddress original position] heap finalLocals finalHeap globals ∧
      table.Found finalHeap destination result ∧ table.Keys finalHeap result ∧
      table.view result = Prefix.collect (table.view accumulated)
        (Prefix.expand ceiling (table.view (original.drop position))) ∧
      rows.HeapFrame heap finalHeap destination ∧ LocalsFrame before finalLocals := by
  by_cases active : position < original.length
  · let entry := original[position]
    have present : original[position]? = some entry := List.getElem?_eq_getElem active
    obtain ⟨updated, middle, locals, rowRan, middleFound, middleKeys, rowView, rowFrame, rowLocals⟩ :=
      row_result heap before globals destination ceiling entry accumulated following limit
        (sourceKeys entry (List.getElem_mem active)) found valid
    obtain ⟨result, finalHeap, finalLocals, rest, finalFound, finalKeys, restView, heapFrame, localFrame⟩ :=
      scan middle locals globals sourceAddress destination ceiling (position + 1) original updated
        ((local_following before locals rowLocals).trans following)
        ((local_ceiling before locals rowLocals).trans limit)
        (found_frame heap middle sourceAddress destination original sourceFound different rowFrame)
        (keys_frame heap middle destination original accumulated sourceKeys found rowFrame)
        middleFound middleKeys different
    refine ⟨result, finalHeap, finalLocals, ?_, finalFound, finalKeys, ?_, ?_, ?_⟩
    · apply BlockContinues.dictionary_items_step
        (key := .ref entry.2.1) (value := table.number entry.2.2)
        (middle := heap) (afterTuple := pairHeap heap entry)
        (by simpa only [present, Option.map_some] using next heap sourceAddress position original sourceFound)
        (allocated heap entry)
      simpa only [show Calls.loopContinuation body (cursor sourceAddress original (position + 1)) =
        cursor sourceAddress original (position + 1) from rfl] using rowRan.append rest
    · rw [restView, rowView, ← Prefix.collect_append]
      congr 1
      rw [List.drop_eq_getElem_cons active]
      simp only [table.view, List.map_cons, Prefix.expand, List.flatMap_cons, entry]
    · intro other object excluded initial
      exact heapFrame other object excluded (rowFrame other object excluded initial)
    · intro name keep
      exact (localFrame name keep).trans (rowLocals name keep)
  · have exhausted : original.length ≤ position := by omega
    refine ⟨accumulated, heap, before, ?_, found, valid, ?_, ?_, ?_⟩
    · exact BlockContinues.dictionary_items_done
        (by simpa only [List.getElem?_eq_none exhausted, Option.map_none] using
          next heap sourceAddress position original sourceFound) BlockContinues.nil
    · simp [List.drop_eq_nil_of_le exhausted, table.view, Prefix.expand, Prefix.collect]
    · intro other object _ present
      exact present
    · intro name _
      rfl
termination_by original.length - position
decreasing_by omega

/-- The source loop realizes the mathematical expansion and merge, with no cutoff. -/
theorem result (heap : Mutable.Heap) (before : HeapCalls.Env) (globals : Mutable.Globals)
    (sourceAddress destination ceiling : Nat) (original accumulated : table.Entries)
    (counts : before "counts" = some (.ref sourceAddress))
    (following : before "following" = some (.ref destination))
    (limit : before "ceiling" = some (table.number ceiling))
    (sourceFound : table.Found heap sourceAddress original) (sourceKeys : table.Keys heap original)
    (found : table.Found heap destination accumulated) (valid : table.Keys heap accumulated)
    (different : sourceAddress ≠ destination) :
    ∃ result finalHeap finalLocals,
      BlockContinues program before [loop] heap finalLocals finalHeap globals ∧
      table.Found finalHeap destination result ∧ table.Keys finalHeap result ∧
      table.view result = Prefix.collect (table.view accumulated) (Prefix.expand ceiling (table.view original)) ∧
      rows.HeapFrame heap finalHeap destination ∧ LocalsFrame before finalLocals := by
  obtain ⟨result, finalHeap, finalLocals, ran, finalFound, finalKeys, view, heapFrame, localFrame⟩ :=
    scan heap before globals sourceAddress destination ceiling 0 original accumulated following limit
      sourceFound sourceKeys found valid different
  refine ⟨result, finalHeap, finalLocals, ?_, finalFound, finalKeys, ?_, heapFrame, localFrame⟩
  · obtain ⟨object, present, represented⟩ := fields_found heap sourceAddress original sourceFound
    apply BlockContinues.for_dictionary_items (Evaluates.lookup counts) present represented
    simpa [cursor, keys, fields, List.map_map, Function.comp_def, sequenceSource,
      comprehensionTargetName, comprehensionTargetBody, bindings.target, body, bindings.body] using ran
  · simpa using view

end solvers.«92».traversal
