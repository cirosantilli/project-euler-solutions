import Mathlib.Data.Nat.Digits.Lemmas
import Mathlib.Data.List.Perm.Basic
import Mathlib.Data.List.Range
import Lean.Elab.Tactic.Omega

/-! Decimal digit counts, their in-place updates, and equality of digit multisets. -/
namespace ProjectEulerSolutions.DigitHistogram

set_option Elab.async false

/-- Ten counters in digit order, including the counter for zero. -/
def counts (digits : List Nat) : List Nat := (List.range 10).map (fun digit => digits.count digit)

def signature (n : Nat) : List Nat := counts (Nat.digits 10 n)

@[simp] theorem length_counts (digits : List Nat) : (counts digits).length = 10 := by simp [counts]

@[simp] theorem get_counts (digits : List Nat) (digit : Nat) (bounded : digit < 10) :
    (counts digits)[digit] = digits.count digit := by simp [counts]

@[simp] theorem get_counts? (digits : List Nat) (digit : Nat) (bounded : digit < 10) :
    (counts digits)[digit]? = some (digits.count digit) := by simp [counts, bounded]

@[simp] theorem counts_nil : counts [] = List.replicate 10 0 := by decide

/-- Appending one decimal digit increments exactly its own counter. -/
theorem append_digit (digits : List Nat) (digit : Nat) :
    counts (digits ++ [digit]) = (counts digits).set digit (digits.count digit + 1) := by
  apply List.ext_getElem
  · simp
  · intro index leftBound rightBound
    have within : index < 10 := by simpa using leftBound
    by_cases same : index = digit
    · subst index
      simp
    · simp [Ne.symm same]

/-- A histogram determines a decimal digit multiset completely. -/
theorem counts_eq_iff (left right : List Nat) (leftBounded : ∀ digit ∈ left, digit < 10)
    (rightBounded : ∀ digit ∈ right, digit < 10) : counts left = counts right ↔ left.Perm right := by
  constructor
  · intro equal
    apply List.perm_iff_count.mpr
    intro digit
    by_cases within : digit < 10
    · have lookup := congrArg (fun items : List Nat => items[digit]?) equal
      simpa only [get_counts? left digit within, get_counts? right digit within, Option.some.injEq] using lookup
    · have absentLeft : digit ∉ left := fun member => within (leftBounded digit member)
      have absentRight : digit ∉ right := fun member => within (rightBounded digit member)
      simp [List.count_eq_zero.mpr absentLeft, List.count_eq_zero.mpr absentRight]
  · intro permutation
    apply List.map_congr_left
    intro digit _
    exact permutation.count_eq digit

/-- The ten-counter signature is equivalent to permutation of the original decimal digits. -/
theorem signature_eq_iff (a b : Nat) : signature a = signature b ↔ (Nat.digits 10 a).Perm (Nat.digits 10 b) :=
  counts_eq_iff _ _ (fun _ member => Nat.digits_lt_base (by decide) member)
    (fun _ member => Nat.digits_lt_base (by decide) member)

end ProjectEulerSolutions.DigitHistogram
