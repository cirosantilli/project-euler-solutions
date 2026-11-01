import ProjectEulerSolutions.Lib.RationalSetStorage

/-! Multi-element union updates of rational result sets preserve their shared tuple objects. -/
namespace ProjectEulerSolutions.RationalPair.Heap
open FormalSW.PyAst
set_option Elab.async false

theorem mem_fold_insertEntry (incoming before : List SetEntry) (pair : Pair) :
    pair ∈ (incoming.foldl insertEntry before).map Prod.fst ↔
      pair ∈ before.map Prod.fst ∨ pair ∈ incoming.map Prod.fst := by
  induction incoming generalizing before with
  | nil => simp
  | cons entry rest ih =>
    simp only [List.foldl_cons, ih, mem_insertEntry, List.map_cons, List.mem_cons]
    tauto

/-- Read-only caller objects are preserved, even when source and destination refer to the same set. -/
theorem SetAt.add_all (incoming : List SetEntry) {heap : Mutable.Heap} {address : Nat} {entries : List SetEntry}
    (found : SetAt heap address entries) (rows : EntryRows heap incoming) :
    ∃ after,
      (incoming.map (fun entry => Mutable.Value.ref entry.2)).mapM (Mutable.addItem address) heap =
        some (.ok (List.replicate incoming.length (.scalar .none)), after) ∧
      SetAt after address (incoming.foldl insertEntry entries) ∧
      (∀ other, other ≠ address → after[other]? = heap[other]?) ∧ after.length = heap.length := by
  induction incoming generalizing heap entries with
  | nil => exact ⟨heap, rfl, found, fun _ _ => rfl, rfl⟩
  | cons entry rest ih =>
    let inserted := insertEntry entries entry
    let middle := heap.set address (setObject inserted)
    have row := rows entry (List.mem_cons_self ..)
    have next : SetAt middle address inserted := found.write inserted (entryRows_insert heap entries entry found.rows row)
    have nextRows : EntryRows middle rest := by
      intro item member
      exact found.row_write item (rows item (List.mem_cons_of_mem entry member)) inserted
    obtain ⟨after, continued, finalSet, unchanged, size⟩ := ih next nextRows
    refine ⟨after, ?_, finalSet, ?_, ?_⟩
    · have first := found.add entry row
      simp only [List.mapM_map] at continued
      simp [List.mapM_cons, Mutable.eval_bind, first, continued, middle, inserted, Except.map, List.replicate_succ]
    · intro other different
      rw [unchanged other different]
      simp [middle, Ne.symm different]
    · simpa only [middle, List.length_set] using size

/-- Updating from a result set unions its values without copying any rational tuple. -/
theorem SetAt.update {heap : Mutable.Heap} {address source : Nat} {entries incoming : List SetEntry}
    (found : SetAt heap address entries) (input : SetAt heap source incoming) :
    ∃ after,
      HeapCalls.primitive (.updateSet address) [.ref source] heap = some (.ok (.scalar .none), after) ∧
      SetAt after address (incoming.foldl insertEntry entries) ∧
      (∀ other, other ≠ address → after[other]? = heap[other]?) ∧ after.length = heap.length := by
  obtain ⟨after, inserted, finalSet, unchanged, size⟩ := found.add_all incoming input.rows
  refine ⟨after, ?_, finalSet, unchanged, size⟩
  simp [HeapCalls.primitive, Mutable.updateSet, input.read, Mutable.eval_bind, inserted, Except.map]

/-- The union description does not depend on insertion order or duplicate representatives. -/
theorem SetAt.update_values {heap : Mutable.Heap} {address source : Nat} {entries incoming : List SetEntry}
    (found : SetAt heap address entries) (input : SetAt heap source incoming) :
    ∃ after final,
      HeapCalls.primitive (.updateSet address) [.ref source] heap = some (.ok (.scalar .none), after) ∧
      SetAt after address final ∧
      (∀ pair, pair ∈ final.map Prod.fst ↔ pair ∈ entries.map Prod.fst ∨ pair ∈ incoming.map Prod.fst) ∧
      (∀ other, other ≠ address → after[other]? = heap[other]?) ∧ after.length = heap.length := by
  obtain ⟨after, executed, finalSet, unchanged, size⟩ := found.update input
  exact ⟨after, incoming.foldl insertEntry entries, executed, finalSet,
    mem_fold_insertEntry incoming entries, unchanged, size⟩

end ProjectEulerSolutions.RationalPair.Heap
