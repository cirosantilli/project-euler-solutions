import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Sqrt
import Mathlib.Data.Set.Function

/-! Canonical equality patterns and the complete square interval used by problem 98. -/
namespace ProjectEulerSolutions.AnagramSquares
set_option Elab.async false

variable {α β : Type*} [DecidableEq α] [DecidableEq β]

def extend (known : List α) (letter : α) : List α :=
  if letter ∈ known then known else known ++ [letter]

def labels (known : List α) : List α → List Nat
  | [] => []
  | letter :: rest => known.idxOf letter :: labels (extend known letter) rest

def alphabet (known : List α) : List α → List α
  | [] => known
  | letter :: rest => alphabet (extend known letter) rest

@[simp] theorem labels_length (known word : List α) : (labels known word).length = word.length := by
  induction word generalizing known with
  | nil => rfl
  | cons letter rest ih => simp [labels, ih]

theorem extend_prefix (known : List α) (letter : α) : known <+: extend known letter := by
  by_cases member : letter ∈ known <;> simp [extend, member]

@[simp] theorem mem_extend (known : List α) (letter : α) : letter ∈ extend known letter := by
  by_cases member : letter ∈ known <;> simp [extend, member]

theorem alphabet_prefix (known word : List α) : known <+: alphabet known word := by
  induction word generalizing known with
  | nil => exact List.prefix_refl _
  | cons letter rest ih => exact (extend_prefix known letter).trans (ih _)

theorem mem_alphabet (known word : List α) (letter : α) (member : letter ∈ word) :
    letter ∈ alphabet known word := by
  induction word generalizing known with
  | nil => simp at member
  | cons first rest ih =>
      rcases List.mem_cons.mp member with rfl | later
      · exact (alphabet_prefix (extend known letter) rest).subset (mem_extend known letter)
      · exact ih _ later

theorem extend_idxOf (known : List α) (letter : α) : (extend known letter).idxOf letter = known.idxOf letter := by
  by_cases member : letter ∈ known
  · simp [extend, member]
  · simp [extend, member, List.idxOf_append_of_notMem member]

/-- Each emitted label is the position of its letter in the final first-occurrence alphabet. -/
theorem labels_eq_map (known word : List α) :
    labels known word = word.map (fun letter => (alphabet known word).idxOf letter) := by
  induction word generalizing known with
  | nil => rfl
  | cons letter rest ih =>
      simp only [labels, alphabet, List.map_cons, ih]
      rw [← (alphabet_prefix (extend known letter) rest).idxOf_eq_of_mem (mem_extend known letter), extend_idxOf]

/-- Equal labels express precisely equal letters; distinct letters cannot share a label. -/
theorem labels_equal_iff (known word : List α) (i j : Nat) (hi : i < word.length) (hj : j < word.length) :
    (labels known word)[i]'(by simpa using hi) = (labels known word)[j]'(by simpa using hj) ↔ word[i] = word[j] := by
  simp only [labels_eq_map, List.getElem_map]
  exact List.idxOf_inj (mem_alphabet known word word[i] (List.getElem_mem hi))

theorem idxOf_map (f : α → β) (known : List α) (letter : α)
    (reflects : ∀ other ∈ known, f other = f letter ↔ other = letter) :
    (known.map f).idxOf (f letter) = known.idxOf letter := by
  induction known with
  | nil => rfl
  | cons first rest ih =>
      simp only [List.map_cons, List.idxOf_cons, Bool.cond_eq_ite, beq_iff_eq]
      simp only [reflects first (by simp)]
      by_cases same : first = letter
      · simp [same]
      · simp only [if_neg same]
        rw [ih (fun other member => reflects other (by simp [member]))]

/-- Any injective letter-to-digit substitution preserves the pattern cache key. -/
theorem labels_map (f : α → β) (known word : List α)
    (injective : Set.InjOn f {letter | letter ∈ known ++ word}) :
    labels (known.map f) (word.map f) = labels known word := by
  induction word generalizing known with
  | nil => rfl
  | cons letter rest ih =>
      have reflects (other : α) (member : other ∈ known) : f other = f letter ↔ other = letter :=
        ⟨fun equal => injective (by simp [member]) (by simp) equal, fun equal => congrArg f equal⟩
      have membership : f letter ∈ known.map f ↔ letter ∈ known := by
        constructor
        · rintro member
          obtain ⟨other, member, equal⟩ := List.mem_map.mp member
          exact (reflects other member).mp equal ▸ member
        · intro member
          exact List.mem_map_of_mem member
      have extended : extend (known.map f) (f letter) = (extend known letter).map f := by
        simp only [extend, membership]
        split <;> simp
      have included (value : α) (member : value ∈ extend known letter ++ rest) :
          value ∈ known ++ letter :: rest := by
        by_cases present : letter ∈ known
        · simp only [extend, if_pos present, List.mem_append] at member
          rcases member with old | later
          · exact List.mem_append_left _ old
          · exact List.mem_append_right _ (List.mem_cons_of_mem _ later)
        · simpa only [extend, if_neg present, List.mem_append, List.mem_cons,
            List.not_mem_nil, false_or, or_assoc] using member
      have continued : Set.InjOn f {value | value ∈ extend known letter ++ rest} := by
        intro left leftMember right rightMember equal
        exact injective (included left leftMember) (included right rightMember) equal
      simp only [List.map_cons, labels, idxOf_map f known letter reflects, extended]
      rw [ih _ continued]

def firstRoot (width : Nat) : Nat := (10 ^ (width - 1) - 1).sqrt + 1
def lastRoot (width : Nat) : Nat := (10 ^ width - 1).sqrt

theorem root_interval (width root : Nat) :
    firstRoot width ≤ root ∧ root ≤ lastRoot width ↔
      10 ^ (width - 1) ≤ root * root ∧ root * root < 10 ^ width := by
  have lowerPositive : 0 < 10 ^ (width - 1) := Nat.pow_pos (by decide)
  have upperPositive : 0 < 10 ^ width := Nat.pow_pos (by decide)
  simp only [firstRoot, lastRoot, Nat.add_one_le_iff, Nat.sqrt_lt, Nat.le_sqrt]
  omega

end ProjectEulerSolutions.AnagramSquares
