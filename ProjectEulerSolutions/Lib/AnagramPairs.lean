import ProjectEulerSolutions.Lib.AnagramKeys

namespace ProjectEulerSolutions.AnagramPairs
open AnagramKeys
set_option Elab.async false

abbrev Pair := String × String

def partners (word : String) : List String → List Pair
  | [] => []
  | other :: rest =>
      (if word ≠ other ∧ key word = key other then [(word, other)] else []) ++ partners word rest

def pairs : List String → List Pair
  | [] => []
  | word :: rest => partners word rest ++ pairs rest

theorem mem_partners (word other : String) (words : List String) :
    (word, other) ∈ partners word words ↔ other ∈ words ∧ word ≠ other ∧ key word = key other := by
  induction words with
  | nil => simp [partners]
  | cons first rest ih =>
      by_cases accepted : word ≠ first ∧ key word = key first
      · simp only [partners, if_pos accepted, List.singleton_append, List.mem_cons, Prod.mk.injEq,
          true_and, ih]
        constructor
        · rintro (same | ⟨member, different, equal⟩)
          · subst other
            exact ⟨by simp, accepted⟩
          · exact ⟨by simp [member], different, equal⟩
        · rintro ⟨member, different, equal⟩
          rcases member with same | member
          · exact Or.inl same
          · exact Or.inr ⟨member, different, equal⟩
      · simp only [partners, if_neg accepted, List.nil_append, ih, List.mem_cons]
        constructor
        · rintro ⟨member, different, equal⟩
          exact ⟨Or.inr member, different, equal⟩
        · rintro ⟨same | member, different, equal⟩
          · subst other
            exact (accepted ⟨different, equal⟩).elim
          · exact ⟨member, different, equal⟩

theorem partners_as_filter (word : String) (words : List String) :
    partners word words = (words.filter fun other => decide (word ≠ other ∧ key word = key other)).map (word, ·) := by
  induction words with
  | nil => rfl
  | cons other rest ih =>
      by_cases accepted : word ≠ other ∧ key word = key other <;> simp [partners, accepted, ih]

theorem mem_partners_iff (word left right : String) (words : List String) :
    (left, right) ∈ partners word words ↔ left = word ∧ right ∈ words ∧ word ≠ right ∧ key word = key right := by
  rw [partners_as_filter]
  simp only [List.mem_map, List.mem_filter, decide_eq_true_eq, Prod.mk.injEq]
  constructor
  · rintro ⟨other, ⟨member, different, equal⟩, same, rfl⟩
    exact ⟨same.symm, member, different, equal⟩
  · rintro ⟨rfl, member, different, equal⟩
    exact ⟨right, ⟨member, different, equal⟩, rfl, rfl⟩

theorem mem_pairs_sound (words : List String) (left right : String) (member : (left, right) ∈ pairs words) :
    left ∈ words ∧ right ∈ words ∧ left ≠ right ∧ key left = key right := by
  induction words with
  | nil => simp [pairs] at member
  | cons word rest ih =>
      rcases List.mem_append.mp member with first | later
      · obtain ⟨rfl, other, different, equal⟩ := (mem_partners_iff word left right rest).mp first
        exact ⟨by simp, by simp [other], different, equal⟩
      · obtain ⟨first, second, different, equal⟩ := ih later
        exact ⟨by simp [first], by simp [second], different, equal⟩

/-- Every distinct pair of anagram words occurs in one input-order orientation. -/
theorem mem_pairs_complete (words : List String) (left right : String)
    (first : left ∈ words) (second : right ∈ words) (different : left ≠ right) (equal : key left = key right) :
    (left, right) ∈ pairs words ∨ (right, left) ∈ pairs words := by
  induction words with
  | nil => simp at first
  | cons word rest ih =>
      rcases List.mem_cons.mp first with rfl | inRestLeft
      · have second : right ∈ rest := (List.mem_cons.mp second).resolve_left (Ne.symm different)
        exact Or.inl (List.mem_append_left _ ((mem_partners left right rest).mpr ⟨second, different, equal⟩))
      · rcases List.mem_cons.mp second with rfl | inRestRight
        · exact Or.inr (List.mem_append_left _
            ((mem_partners right left rest).mpr ⟨inRestLeft, Ne.symm different, equal.symm⟩))
        · exact (ih inRestLeft inRestRight).imp (List.mem_append_right _) (List.mem_append_right _)

theorem mem_pairs_nonempty (words : List String) (left right : String) (member : (left, right) ∈ pairs words) :
    left.toList ≠ [] ∧ right.toList ≠ [] := by
  obtain ⟨_, _, different, equal⟩ := mem_pairs_sound words left right member
  exact distinct_nonempty left right different equal

end ProjectEulerSolutions.AnagramPairs
