import ProjectEulerSolutions.Lib.AnagramSquares

/-! Canonical pattern equality is exactly an injective substitution between the words. -/
namespace ProjectEulerSolutions.AnagramSquares
set_option Elab.async false

variable {α β : Type*} [DecidableEq α]

/-- Corresponding positions have exactly the same equality relation. -/
def Compatible (word : List α) (digits : List β) : Prop :=
  ∃ length : word.length = digits.length,
    ∀ i j (hi : i < word.length) (hj : j < word.length),
      word[i] = word[j] ↔ digits[i]'(by omega) = digits[j]'(by omega)

theorem compatible_of_labels_eq [DecidableEq β] (word : List α) (digits : List β)
    (equal : labels [] word = labels [] digits) : Compatible word digits := by
  have length : word.length = digits.length := by
    simpa using congrArg List.length equal
  refine ⟨length, ?_⟩
  intro i j hi hj
  rw [← labels_equal_iff [] word i j hi hj,
      ← labels_equal_iff [] digits i j (by omega) (by omega)]
  simp only [equal]

/-- Read the digit at the first occurrence of a letter. The fallback is never used on the word. -/
def substitution (fallback : β) (word : List α) (digits : List β) (letter : α) : β :=
  (digits[word.idxOf letter]?).getD fallback

theorem substitution_at (fallback : β) (word : List α) (digits : List β)
    (compatible : Compatible word digits) (i : Nat) (inside : i < word.length) :
    substitution fallback word digits word[i] = digits[i]'(by rcases compatible with ⟨length, _⟩; omega) := by
  obtain ⟨length, matching⟩ := compatible
  have first : word.idxOf word[i] < word.length := List.idxOf_lt_length_of_mem (List.getElem_mem inside)
  have firstDigit : word.idxOf word[i] < digits.length := by omega
  unfold substitution
  rw [List.getElem?_eq_getElem firstDigit, Option.getD_some]
  exact (matching _ _ first inside).mp (List.getElem_idxOf first)

theorem substitution_map (fallback : β) (word : List α) (digits : List β)
    (compatible : Compatible word digits) :
    word.map (substitution fallback word digits) = digits := by
  apply List.ext_getElem
  · simpa using compatible.choose
  · intro i left right
    simp only [List.getElem_map]
    exact substitution_at fallback word digits compatible i (by simpa using left)

theorem substitution_injective (fallback : β) (word : List α) (digits : List β)
    (compatible : Compatible word digits) :
    Set.InjOn (substitution fallback word digits) {letter | letter ∈ word} := by
  obtain ⟨length, matching⟩ := compatible
  intro a ha b hb equal
  have ia : word.idxOf a < word.length := List.idxOf_lt_length_of_mem ha
  have ib : word.idxOf b < word.length := List.idxOf_lt_length_of_mem hb
  have da : word.idxOf a < digits.length := by omega
  have db : word.idxOf b < digits.length := by omega
  simp only [substitution, List.getElem?_eq_getElem da, List.getElem?_eq_getElem db, Option.getD_some] at equal
  have letters := (matching _ _ ia ib).mpr equal
  simpa only [List.getElem_idxOf] using letters

/-- Pattern equality is a complete filter: it rejects no bijective assignment and accepts no collision. -/
theorem labels_eq_iff_substitution [DecidableEq β] (fallback : β) (word : List α) (digits : List β) :
    labels [] word = labels [] digits ↔
      ∃ f : α → β, word.map f = digits ∧ Set.InjOn f {letter | letter ∈ word} := by
  constructor
  · intro equal
    have compatible := compatible_of_labels_eq word digits equal
    exact ⟨substitution fallback word digits, substitution_map fallback word digits compatible,
      substitution_injective fallback word digits compatible⟩
  · rintro ⟨f, mapped, injective⟩
    have preserved := labels_map f [] word (by simpa using injective)
    simpa only [List.map_nil, mapped] using preserved.symm

end ProjectEulerSolutions.AnagramSquares
