import lang.py.FormalSW.HeapSort

namespace ProjectEulerSolutions.HeapKeySorting
open FormalSW.PyAst
set_option Elab.async false

theorem insert_map (heap : Mutable.Heap) (encode : α → Mutable.Value × Mutable.Value)
    (less : α → α → Bool) (value : α) (items : List α)
    (compared : ∀ item ∈ items, HeapSort.compare (HeapSort.comparisonFuel heap) heap false
      (encode item).2 (encode value).2 = some (.ok (less item value))) :
    HeapSort.keyedInsert heap (encode value) (items.map encode) =
      some (.ok ((HeapSort.insertBy less value items).map encode)) := by
  induction items with
  | nil => rfl
  | cons first rest ih =>
      have firstCompared := compared first List.mem_cons_self
      have restCompared := ih (fun item member => compared item (List.mem_cons_of_mem _ member))
      simp only [List.map_cons, HeapSort.keyedInsert, firstCompared, HeapSort.insertBy]
      cases less first value <;> simp [restCompared, Except.map]

/-- The source's cached-key sort follows any total comparison proved for its input keys. -/
theorem sort_map (heap : Mutable.Heap) (encode : α → Mutable.Value × Mutable.Value)
    (less : α → α → Bool) (items : List α)
    (compared : ∀ left ∈ items, ∀ right ∈ items,
      HeapSort.compare (HeapSort.comparisonFuel heap) heap false (encode left).2 (encode right).2 =
        some (.ok (less left right))) :
    HeapSort.keyedSort heap (items.map encode) = some (.ok ((HeapSort.sortBy less items).map encode)) := by
  induction items with
  | nil => rfl
  | cons first rest ih =>
      have restCompared := ih (fun left hl right hr =>
        compared left (List.mem_cons_of_mem _ hl) right (List.mem_cons_of_mem _ hr))
      simp only [List.map_cons, HeapSort.keyedSort, restCompared, HeapSort.sortBy]
      apply insert_map
      intro item member
      exact compared item (List.mem_cons_of_mem _ ((HeapSort.sortBy_perm less rest).mem_iff.mp member))
        first List.mem_cons_self

end ProjectEulerSolutions.HeapKeySorting
