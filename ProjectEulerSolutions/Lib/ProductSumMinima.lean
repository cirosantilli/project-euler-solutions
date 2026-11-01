import ProjectEulerSolutions.Lib.ProductSumSearchCorrect

/-! Exact minima from the factor-search trace and the complete product bound. -/
namespace ProjectEulerSolutions.ProductSumNumbers
set_option Elab.async false

def leastFor (size sentinel : Nat) : List (Nat × Nat) → Nat
  | [] => sentinel
  | (candidateSize, value) :: rest =>
      if candidateSize = size then min value (leastFor size sentinel rest)
      else leastFor size sentinel rest

theorem leastFor_le_sentinel (size sentinel : Nat) (entries : List (Nat × Nat)) :
    leastFor size sentinel entries ≤ sentinel := by
  induction entries with
  | nil => rfl
  | cons entry rest ih =>
      obtain ⟨candidateSize, value⟩ := entry
      simp only [leastFor]
      split_ifs <;> omega

theorem leastFor_le_entry (size sentinel value : Nat) (entries : List (Nat × Nat))
    (member : (size, value) ∈ entries) : leastFor size sentinel entries ≤ value := by
  induction entries with
  | nil => simp at member
  | cons entry rest ih =>
      obtain ⟨candidateSize, candidateValue⟩ := entry
      rcases List.mem_cons.mp member with same | later
      · cases same
        simp [leastFor]
      · have bound := ih later
        simp only [leastFor]
        split_ifs <;> omega

theorem leastFor_selected (size sentinel : Nat) (entries : List (Nat × Nat)) :
    leastFor size sentinel entries = sentinel ∨
      (size, leastFor size sentinel entries) ∈ entries := by
  induction entries with
  | nil => exact Or.inl rfl
  | cons entry rest ih =>
      obtain ⟨candidateSize, value⟩ := entry
      by_cases same : candidateSize = size
      · subst candidateSize
        by_cases lower : value ≤ leastFor size sentinel rest
        · right
          simp [leastFor, min_eq_left lower]
        · simpa only [leastFor, if_pos rfl, min_eq_right (by omega : leastFor size sentinel rest ≤ value)]
            using ih.imp_right (List.mem_cons_of_mem _)
      · simpa only [leastFor, if_neg same] using ih.imp_right (List.mem_cons_of_mem _)

/-- The sentinel is just above the complete bound; it is never a minimum for size at least two. -/
def minimum (maxSize size : Nat) : Nat :=
  leastFor size (2 * maxSize + 1) (search maxSize (2 * maxSize) 2 1 0 0)

theorem minimum_le_witness (maxSize size : Nat) (lower : 2 ≤ size) (upper : size ≤ maxSize) :
    minimum maxSize size ≤ 2 * size := by
  apply leastFor_le_entry
  exact root_complete maxSize (2 * maxSize) size (2 * size) upper lower (by omega)
    (bound_witness size lower)

theorem minimum_correct (maxSize size : Nat) (lower : 2 ≤ size) (upper : size ≤ maxSize) :
    Represents size (minimum maxSize size) ∧
      ∀ value, Represents size value → minimum maxSize size ≤ value := by
  have bound := minimum_le_witness maxSize size lower upper
  constructor
  · have selected := leastFor_selected size (2 * maxSize + 1) (search maxSize (2 * maxSize) 2 1 0 0)
    change minimum maxSize size = 2 * maxSize + 1 ∨
      (size, minimum maxSize size) ∈ search maxSize (2 * maxSize) 2 1 0 0 at selected
    rcases selected with impossible | member
    · omega
    · exact ((root_iff maxSize (2 * maxSize) size (minimum maxSize size) lower).mp member).2.2
  · intro value represented
    by_cases inside : value ≤ 2 * maxSize
    · exact leastFor_le_entry size (2 * maxSize + 1) value _
        (root_complete maxSize (2 * maxSize) size value upper lower inside represented)
    · omega

/-- Increasing the requested range cannot change a minimum already covered. -/
theorem minimum_stable (left right size : Nat) (lower : 2 ≤ size)
    (leftBound : size ≤ left) (rightBound : size ≤ right) :
    minimum left size = minimum right size := by
  have a := minimum_correct left size lower leftBound
  have b := minimum_correct right size lower rightBound
  exact Nat.le_antisymm (a.2 _ b.1) (b.2 _ a.1)

def answer (maxSize : Nat) : Nat :=
  ((List.range' 2 (maxSize - 1)).map (minimum maxSize)).eraseDups.sum

end ProjectEulerSolutions.ProductSumNumbers
