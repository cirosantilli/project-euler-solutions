import Mathlib.Data.Nat.Sqrt
import Mathlib.Tactic.Linarith
import Init.Data.Nat.ToString
import Init.Data.Char.Lemmas

/-! Canonical decimal character lists and their positional integer values. -/
namespace ProjectEulerSolutions.DecimalCharacters
set_option Elab.async false

def value (digits : List Char) : Nat := Nat.ofDigitChars 10 digits 0

theorem cons_value (first : Char) (rest : List Char) :
    value (first :: rest) = 10 ^ rest.length * (first.toNat - 48) + value rest := by
  simp only [value, Nat.ofDigitChars_cons, Nat.mul_zero, Nat.zero_add]
  exact Nat.ofDigitChars_eq_ofDigitChars_zero

theorem upper (digits : List Char) (decimal : ∀ digit ∈ digits, digit.isDigit) : value digits < 10 ^ digits.length := by
  induction digits with
  | nil => simp [value]
  | cons first rest ih =>
      have firstBounds := Char.isDigit_iff_toNat.mp (decimal first (by simp))
      have restBound := ih (fun digit member => decimal digit (by simp [member]))
      have digitBound : first.toNat - 48 ≤ 9 := by
        have upper : first.toNat ≤ 57 := firstBounds.2
        omega
      have multiplied := Nat.mul_le_mul_left (10 ^ rest.length) digitBound
      rw [cons_value, List.length_cons, Nat.pow_succ]
      nlinarith

theorem lower (first : Char) (rest : List Char) (decimal : first.isDigit) (nonzero : first ≠ '0') :
    10 ^ rest.length ≤ value (first :: rest) := by
  have bounds := Char.isDigit_iff_toNat.mp decimal
  have different : first.toNat ≠ 48 := fun same => nonzero (Char.toNat_inj.mp same)
  have positive : 1 ≤ first.toNat - 48 := by
    have lower : 48 ≤ first.toNat := bounds.1
    omega
  have multiplied := Nat.mul_le_mul_left (10 ^ rest.length) positive
  rw [cons_value]
  omega

/-- Decimal positional evaluation is injective when the character lists have equal length. -/
theorem injective (left right : List Char) (leftDigits : ∀ digit ∈ left, digit.isDigit)
    (rightDigits : ∀ digit ∈ right, digit.isDigit) (lengths : left.length = right.length)
    (equal : value left = value right) : left = right := by
  induction left generalizing right with
  | nil => exact (List.eq_nil_of_length_eq_zero (by simpa using lengths.symm)).symm
  | cons first rest ih =>
      cases right with
      | nil => simp at lengths
      | cons second tail =>
          have restLength : rest.length = tail.length := by simpa using lengths
          have firstBounds := Char.isDigit_iff_toNat.mp (leftDigits first (by simp))
          have secondBounds := Char.isDigit_iff_toNat.mp (rightDigits second (by simp))
          simp only [Char.reduceToNat] at firstBounds secondBounds
          have restDigits : ∀ digit ∈ rest, digit.isDigit := fun digit member => leftDigits digit (by simp [member])
          have tailDigits : ∀ digit ∈ tail, digit.isDigit := fun digit member => rightDigits digit (by simp [member])
          have restBound := upper rest restDigits
          have tailBound := upper tail tailDigits
          rw [cons_value, cons_value, ← restLength] at equal
          rw [← restLength] at tailBound
          have same : first.toNat - 48 = second.toNat - 48 := by
            rcases lt_trichotomy (first.toNat - 48) (second.toNat - 48) with smaller | same | larger
            · have multiplied := Nat.mul_le_mul_left (10 ^ rest.length) (Nat.succ_le_of_lt smaller)
              nlinarith
            · exact same
            · have multiplied := Nat.mul_le_mul_left (10 ^ rest.length) (Nat.succ_le_of_lt larger)
              nlinarith
          have head : first = second := Char.toNat_inj.mp (by omega)
          have tails : value rest = value tail := by rw [same] at equal; omega
          rw [head]
          exact congrArg (second :: ·) (ih tail restDigits tailDigits restLength tails)

/-- A nonempty decimal string without a leading zero is exactly the printed form of its value. -/
theorem canonical (first : Char) (rest : List Char) (decimal : ∀ digit ∈ first :: rest, digit.isDigit)
    (nonzero : first ≠ '0') : (toString (value (first :: rest))).toList = first :: rest := by
  let n := value (first :: rest)
  have upperBound := upper (first :: rest) decimal
  have lowerBound := lower first rest (decimal first (by simp)) nonzero
  have lengthUpper : n.repr.length ≤ (first :: rest).length :=
    (Nat.length_repr_le_iff (by simp)).mpr upperBound
  have lengthLower : (first :: rest).length ≤ n.repr.length := by
    by_contra smaller
    by_cases empty : rest = []
    · have positive := Nat.length_repr_pos (n := n)
      simp only [empty, List.length_cons, List.length_nil] at smaller
      omega
    · have positive : 0 < rest.length := List.length_pos_iff.mpr empty
      have bound := (Nat.length_repr_le_iff positive).mp (show n.repr.length ≤ rest.length by simpa using Nat.le_of_lt_succ (Nat.lt_of_not_ge smaller))
      omega
  apply injective
  · intro digit member
    change digit ∈ n.repr.toList at member
    rw [Nat.toList_repr] at member
    exact Nat.isDigit_of_mem_toDigits (by decide) (by decide) member
  · exact decimal
  · change n.repr.length = (first :: rest).length
    omega
  · simp [value, Nat.toList_repr]

end ProjectEulerSolutions.DecimalCharacters
