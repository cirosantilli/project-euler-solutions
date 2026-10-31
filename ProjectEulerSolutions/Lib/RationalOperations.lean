import ProjectEulerSolutions.Lib.RationalPair
import ProjectEulerSolutions.Lib.ArithmeticExpressions

/-! Exact arithmetic on unreduced positive-denominator pairs used by the reference. -/
namespace ProjectEulerSolutions.RationalPair
set_option Elab.async false

def add (a b : Pair) : Pair := (a.1 * b.2 + b.1 * a.2, a.2 * b.2)
def sub (a b : Pair) : Pair := (a.1 * b.2 - b.1 * a.2, a.2 * b.2)
def mul (a b : Pair) : Pair := (a.1 * b.1, a.2 * b.2)
def quotient (a b : Pair) : Pair :=
  if 0 < b.1 then (a.1 * b.2, a.2 * b.1)
  else (-a.1 * b.2, -a.2 * b.1)

def reference (a b : Pair) : List Pair :=
  [add a b, sub a b, mul a b] ++ if b.1 = 0 then [] else [quotient a b]

theorem add_value (a b : Pair) (ha : a.2 ≠ 0) (hb : b.2 ≠ 0) :
    value (add a b) = value a + value b := by
  simp only [value, add]
  exact (Rat.divInt_add_divInt _ _ ha hb).symm

theorem sub_value (a b : Pair) (ha : a.2 ≠ 0) (hb : b.2 ≠ 0) :
    value (sub a b) = value a - value b := by
  simp only [value, sub]
  simp only [sub_eq_add_neg, Rat.neg_divInt, Rat.divInt_add_divInt _ _ ha hb, neg_mul]

theorem mul_value (a b : Pair) : value (mul a b) = value a * value b := by
  simp [value, mul, Rat.divInt_mul_divInt]

theorem quotient_value (a b : Pair) : value (quotient a b) = value a / value b := by
  simp only [quotient]
  split
  · simp [value, Rat.divInt_div_divInt]
  · simp [value, Rat.divInt_div_divInt, Rat.divInt_neg]

theorem quotient_positive (a b : Pair) (ha : 0 < a.2) (nonzero : b.1 ≠ 0) :
    0 < (quotient a b).2 := by
  simp only [quotient]
  split_ifs with positive
  · exact mul_pos ha positive
  · exact mul_pos_of_neg_of_neg (by omega) (by omega)

/-- Every emitted rational retains a positive denominator; zero division emits no candidate. -/
theorem reference_positive (a b pair : Pair) (ha : 0 < a.2) (hb : 0 < b.2)
    (member : pair ∈ reference a b) : 0 < pair.2 := by
  simp only [reference, List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with (rfl | rfl | rfl) | extra
  · exact mul_pos ha hb
  · exact mul_pos ha hb
  · exact mul_pos ha hb
  · split_ifs at extra with zero
    · simp at extra
    · obtain rfl := List.mem_singleton.mp extra
      exact quotient_positive a b ha zero

/-- The exact value list agrees with all four defined ordered arithmetic operations. -/
theorem reference_values (a b : Pair) (ha : 0 < a.2) (hb : 0 < b.2) :
    (reference a b).map value = ArithmeticExpressions.operations (value a) (value b) := by
  have zero : value b = 0 ↔ b.1 = 0 := Rat.divInt_eq_zero hb.ne'
  by_cases zeroNumerator : b.1 = 0 <;>
    simp [reference, ArithmeticExpressions.operations, zero, zeroNumerator, add_value a b ha.ne' hb.ne',
    sub_value a b ha.ne' hb.ne', mul_value, quotient_value]

end ProjectEulerSolutions.RationalPair
