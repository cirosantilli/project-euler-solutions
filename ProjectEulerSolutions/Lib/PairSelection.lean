import ProjectEulerSolutions.Lib.IndexedFilterPairs

/-! Increasing index pairs cover every choice of two occurrences in a multiset. -/
namespace ProjectEulerSolutions.IndexedFilter
set_option Elab.async false

private theorem parts_perm (before middle after : List α) (a b : α) :
    (before ++ a :: (middle ++ b :: after)).Perm (a :: b :: (before ++ middle ++ after)) := by
  apply List.perm_middle.trans
  simpa only [List.append_assoc] using
    (List.perm_middle (a := b) (l₁ := before ++ middle) (l₂ := after)).cons a

theorem pair_parts {values rest : List α} {a b : α} (permutation : values.Perm (a :: b :: rest)) :
    ∃ before middle after,
      (values = before ++ a :: (middle ++ b :: after) ∨ values = before ++ b :: (middle ++ a :: after)) ∧
      (before ++ middle ++ after).Perm rest := by
  have memberA : a ∈ values := permutation.mem_iff.mpr (by simp)
  obtain ⟨before, after, rfl⟩ := List.mem_iff_append.mp memberA
  have remaining : (before ++ after).Perm (b :: rest) :=
    (List.perm_middle.symm.trans permutation).cons_inv
  have memberB : b ∈ before ++ after := remaining.mem_iff.mpr (by simp)
  rcases List.mem_append.mp memberB with earlier | later
  · obtain ⟨leading, middle, rfl⟩ := List.mem_iff_append.mp earlier
    have rearranged : ((leading ++ b :: middle) ++ a :: after).Perm (b :: a :: (leading ++ middle ++ after)) := by
      simpa only [List.append_assoc, List.cons_append] using parts_perm leading middle after b a
    have swapped : ((leading ++ b :: middle) ++ a :: after).Perm (b :: a :: rest) :=
      permutation.trans (List.Perm.swap _ _ _)
    refine ⟨leading, middle, after, .inr ?_, (rearranged.symm.trans swapped).cons_inv.cons_inv⟩
    simp only [List.append_assoc, List.cons_append]
  · obtain ⟨middle, trailing, rfl⟩ := List.mem_iff_append.mp later
    exact ⟨before, middle, trailing, .inl rfl,
      ((parts_perm before middle trailing a b).symm.trans permutation).cons_inv.cons_inv⟩

private theorem parts_indices (before middle after : List α) (a b : α) :
    (before ++ a :: (middle ++ b :: after))[before.length]? = some a ∧
      (before ++ a :: (middle ++ b :: after))[before.length + 1 + middle.length]? = some b := by
  constructor
  · simp
  · rw [List.getElem?_append_right (by omega)]
    have offset : before.length + 1 + middle.length - before.length = middle.length + 1 := by omega
    simp [offset]

/-- The two possible operand orientations account for choosing only indices with i < j. -/
theorem pair_indices {values rest : List α} {a b : α} (permutation : values.Perm (a :: b :: rest)) :
    ∃ i j, i < j ∧ j < values.length ∧
      ((values[i]? = some a ∧ values[j]? = some b) ∨ (values[i]? = some b ∧ values[j]? = some a)) ∧
      (select (exceptTwo i j) 0 values).Perm rest := by
  obtain ⟨before, middle, after, split, remainder⟩ := pair_parts permutation
  refine ⟨before.length, before.length + 1 + middle.length, by omega, ?_, ?_, ?_⟩
  · rcases split with rfl | rfl <;> simp only [List.length_append, List.length_cons] <;> omega
  · rcases split with rfl | rfl
    · exact .inl (parts_indices before middle after a b)
    · exact .inr (parts_indices before middle after b a)
  · rcases split with rfl | rfl <;> rwa [select_two_parts]

end ProjectEulerSolutions.IndexedFilter
