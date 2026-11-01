import ProjectEulerSolutions.Lib.ProductSumMinima

/-! The mutable best-value table is a pointwise minimum of the visited factor trace. -/
namespace ProjectEulerSolutions.ProductSumNumbers
set_option Elab.async false

def relax (best : List Nat) (entry : Nat × Nat) : List Nat :=
  best.set entry.1 (min entry.2 (best[entry.1]?.getD 0))

def accumulate (entries : List (Nat × Nat)) (best : List Nat) : List Nat :=
  entries.foldl relax best

theorem relax_length (best : List Nat) (entry : Nat × Nat) :
    (relax best entry).length = best.length := by simp [relax]

theorem relax_lookup (best : List Nat) (entry : Nat × Nat) (index : Nat)
    (inside : index < best.length) :
    (relax best entry)[index]?.getD 0 =
      if entry.1 = index then min entry.2 (best[index]?.getD 0) else best[index]?.getD 0 := by
  obtain ⟨size, value⟩ := entry
  by_cases same : size = index
  · subst size
    simp [relax, List.getElem?_set_self inside]
  · simp [relax, List.getElem?_set_ne same, same]

/-- Omitting a non-improving write has exactly the same effect on the table. -/
theorem relax_guarded (best : List Nat) (size value : Nat) (inside : size < best.length) :
    relax best (size, value) =
      if value < best[size]?.getD 0 then best.set size value else best := by
  have found : best[size]?.getD 0 = best[size] := by simp [inside]
  simp only [relax, found]
  by_cases lower : value < best[size]
  · rw [if_pos lower, min_eq_left (by omega)]
  · rw [if_neg lower, min_eq_right (by omega), List.set_getElem_self inside]

theorem leastFor_min (size first second : Nat) (entries : List (Nat × Nat)) :
    leastFor size (min first second) entries = min first (leastFor size second entries) := by
  induction entries with
  | nil => rfl
  | cons entry rest ih =>
      obtain ⟨candidateSize, value⟩ := entry
      simp only [leastFor]
      split_ifs <;> simp [ih, min_left_comm]

theorem accumulate_length (entries : List (Nat × Nat)) (best : List Nat) :
    (accumulate entries best).length = best.length := by
  induction entries generalizing best with
  | nil => rfl
  | cons entry rest ih =>
      exact (ih (relax best entry)).trans (relax_length best entry)

theorem accumulate_lookup (entries : List (Nat × Nat)) (best : List Nat) (index : Nat)
    (inside : index < best.length) :
    (accumulate entries best)[index]?.getD 0 = leastFor index (best[index]?.getD 0) entries := by
  induction entries generalizing best with
  | nil => rfl
  | cons entry rest ih =>
      obtain ⟨size, value⟩ := entry
      change (accumulate rest (relax best (size, value)))[index]?.getD 0 = _
      rw [ih (relax best (size, value)) (by simpa only [relax_length] using inside),
        relax_lookup best (size, value) index inside]
      simp only [leastFor]
      split_ifs <;> simp only [leastFor_min]

theorem accumulate_append (before after : List (Nat × Nat)) (best : List Nat) :
    accumulate (before ++ after) best = accumulate after (accumulate before best) := by
  exact List.foldl_append

def table (maxSize : Nat) : List Nat :=
  accumulate (search maxSize (2 * maxSize) 2 1 0 0) (List.replicate (maxSize + 1) (2 * maxSize + 1))

theorem table_length (maxSize : Nat) : (table maxSize).length = maxSize + 1 := by
  simp [table, accumulate_length]

theorem table_lookup (maxSize size : Nat) (inside : size ≤ maxSize) :
    (table maxSize)[size]?.getD 0 = minimum maxSize size := by
  unfold table minimum
  rw [accumulate_lookup _ _ size (by simp; omega)]
  simp [show size < maxSize + 1 by omega]

theorem table_correct (maxSize size : Nat) (lower : 2 ≤ size) (upper : size ≤ maxSize) :
    Represents size ((table maxSize)[size]?.getD 0) ∧
      ∀ value, Represents size value → (table maxSize)[size]?.getD 0 ≤ value := by
  rw [table_lookup maxSize size upper]
  exact minimum_correct maxSize size lower upper

end ProjectEulerSolutions.ProductSumNumbers
