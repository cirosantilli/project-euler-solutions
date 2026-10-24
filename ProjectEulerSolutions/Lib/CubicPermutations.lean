import ProjectEulerSolutions.Lib.DigitHistogram
import ProjectEulerSolutions.Lib.PrimePermutations
import ProjectEulerStatements.Lib.DecimalDigits
import lang.py.FormalSW.HashKeys

/-! Decimal character counters used by the cubic-permutation source solver. -/
namespace ProjectEulerSolutions.CubicPermutations
open FormalSW.PyAst
set_option Elab.async false

/-- The string representation contains one zero digit even when the positional digit list is empty. -/
def decimal (n : Nat) : List Nat := if n = 0 then [0] else (Nat.digits 10 n).reverse
def signature (n : Nat) : List Nat := DigitHistogram.counts (decimal n)

theorem decimal_bounded (n digit : Nat) (member : digit ∈ decimal n) : digit < 10 := by
  by_cases zero : n = 0
  · simp only [decimal, zero, if_pos, List.mem_singleton] at member
    omega
  · exact Nat.digits_lt_base (by decide) (by simpa [decimal, zero] using member)

theorem text_decimal (n : Nat) :
    (toString (n : Int)).toList = (decimal n).map Nat.digitChar := by
  by_cases zero : n = 0
  · subst n
    decide
  · simpa [decimal, zero, toString, Int.repr, Nat.repr] using
      ProjectEulerStatements.DecimalDigits.toDigits_eq_digitChars n (by omega)

theorem digit_ordinal (digit : Nat) (bounded : digit < 10) :
    ((Nat.digitChar digit).toNat : Int) - 48 = digit := by
  interval_cases digit <;> decide

theorem signature_positive (n : Nat) (positive : 0 < n) : signature n = DigitHistogram.signature n := by
  apply (DigitHistogram.counts_eq_iff (decimal n) (Nat.digits 10 n)
    (decimal_bounded n) (fun _ member => Nat.digits_lt_base (by decide) member)).mpr
  simpa only [decimal, if_neg (show n ≠ 0 by omega)] using List.reverse_perm (Nat.digits 10 n)

/-- Counter keys and the independent sorted-digit reference identify exactly the same positive cubes. -/
theorem signature_eq_sorted (left right : Nat) (leftPositive : 0 < left) (rightPositive : 0 < right) :
    signature left = signature right ↔ PrimePermutations.sortedDigits left = PrimePermutations.sortedDigits right := by
  rw [signature_positive left leftPositive, signature_positive right rightPositive,
    DigitHistogram.signature_eq_iff, PrimePermutations.sortedDigits_eq_iff]

def counterKey (values : List Nat) : HashKey := .tuple (values.map (fun value : Nat => .scalar (.int value)))
def key (n : Nat) : HashKey := counterKey (signature n)

private theorem integer_keys_equal (fuel : Nat) (positive : 0 < fuel) (left right : List Nat) :
    (left.map (fun value : Nat => HashKey.scalar (.int value))).isEqv
      (right.map (fun value : Nat => HashKey.scalar (.int value))) (HashKey.equalFuel fuel) = decide (left = right) := by
  cases fuel with
  | zero => omega
  | succ fuel =>
      induction left generalizing right with
      | nil => cases right <;> rfl
      | cons value rest ih =>
          cases right with
          | nil => rfl
          | cons other tail =>
              simp [List.isEqv, HashKey.equalFuel, scalarCompare, evalCmpOp, ih]

/-- Integer-only tuple hashing compares the full counter contents, independent of tuple addresses. -/
theorem counterKey_equal (left right : List Nat) :
    HashKey.equal (counterKey left) (counterKey right) = decide (left = right) := by
  unfold HashKey.equal counterKey
  change (left.map (fun value : Nat => HashKey.scalar (.int value))).isEqv
    (right.map (fun value : Nat => HashKey.scalar (.int value)))
    (HashKey.equalFuel ((HashKey.listDepth (left.map (fun value : Nat => .scalar (.int value))) + 1) +
      (HashKey.listDepth (right.map (fun value : Nat => .scalar (.int value))) + 1))) = _
  exact integer_keys_equal _ (by omega) left right

theorem key_equal_sorted (left right : Nat) (leftPositive : 0 < left) (rightPositive : 0 < right) :
    HashKey.equal (key left) (key right) = true ↔
      PrimePermutations.sortedDigits left = PrimePermutations.sortedDigits right := by
  rw [key, key, counterKey_equal, decide_eq_true_eq]
  exact signature_eq_sorted left right leftPositive rightPositive

end ProjectEulerSolutions.CubicPermutations
