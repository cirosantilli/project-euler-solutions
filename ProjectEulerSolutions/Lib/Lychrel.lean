import ProjectEulerStatements.Lib.DecimalDigits
import ProjectEulerSolutions.Equiv.P4
import lang.py.FormalSW.IntegerParsing
import Init.Data.List.Range

/-! Structurally terminating reverse-and-add mathematics, independent of old P55 proof files. -/
namespace ProjectEulerSolutions.Lychrel
open FormalSW.PyAst
set_option Elab.async false

/-- Nat.digits uses little-endian digits; reversing it before ofDigits reverses the decimal number. -/
def reverse (n : Nat) : Nat := Nat.ofDigits 10 (Nat.digits 10 n).reverse

def palindrome (n : Nat) : Bool := decide (Nat.digits 10 n = (Nat.digits 10 n).reverse)
def step (n : Nat) : Nat := n + reverse n

def iterate : Nat → Nat → Nat
  | 0, n => n
  | fuel+1, n => iterate fuel (step n)

def candidate : Nat → Nat → Bool
  | _, 0 => true
  | n, fuel+1 => if palindrome (step n) then false else candidate (step n) fuel

def count (limit fuel : Nat) : Nat :=
  ((List.range' 1 (limit-1)).filter (fun n => candidate n fuel)).length

theorem iterate_zero (n : Nat) : iterate 0 n = n := rfl
theorem iterate_succ (fuel n : Nat) : iterate (fuel+1) n = iterate fuel (step n) := rfl

theorem iterate_step (fuel n : Nat) : iterate fuel (step n) = step (iterate fuel n) := by
  induction fuel generalizing n with
  | zero => rfl
  | succ fuel ih => exact ih (step n)

theorem iterate_add (first second n : Nat) : iterate (first+second) n = iterate second (iterate first n) := by
  induction first generalizing n with
  | zero => simp only [Nat.zero_add, iterate_zero]
  | succ first ih => simpa only [Nat.succ_add, iterate_succ] using ih (step n)

theorem candidate_zero (n : Nat) : candidate n 0 = true := rfl
theorem candidate_succ (n fuel : Nat) :
    candidate n (fuel+1) = if palindrome (step n) then false else candidate (step n) fuel := rfl

private theorem shifted_any (n fuel : Nat) :
    (List.range' 2 fuel).any (fun k => palindrome (iterate k n)) =
      (List.range' 1 fuel).any (fun k => palindrome (iterate k (step n))) := by
  rw [show List.range' 2 fuel = (List.range' 1 fuel).map (fun k => k+1) from List.range'_succ_left]
  simp only [List.any_map, Function.comp_def, iterate_succ]

/-- Testing the positive steps 1 through fuel is exactly the recursive candidate predicate. -/
theorem candidate_eq_not_any (n fuel : Nat) :
    candidate n fuel = !((List.range' 1 fuel).any (fun k => palindrome (iterate k n))) := by
  induction fuel generalizing n with
  | zero => rfl
  | succ fuel ih =>
      rw [candidate_succ, List.range'_succ, List.any_cons, shifted_any, ih (step n)]
      simp only [iterate_succ, iterate_zero]
      cases palindrome (step n) <;>
        cases (List.range' 1 fuel).any (fun k => palindrome (iterate k (step n))) <;> rfl

private theorem filtered_length_sum (values : List Nat) (predicate : Nat → Bool) :
    ((values.filter predicate).length : Int) =
      (values.map fun n => if predicate n then (1 : Int) else 0).sum := by
  induction values with
  | nil => rfl
  | cons n rest ih =>
      cases tested : predicate n with
      | false => simp [tested, ih]
      | true => simp [tested, ih, Int.add_comm]

theorem count_eq_sum (limit fuel : Nat) :
    (count limit fuel : Int) =
      ((List.range' 1 (limit-1)).map fun n => if candidate n fuel then (1 : Int) else 0).sum :=
  filtered_length_sum _ _

/-- Decimal parser accumulation agrees with the positional value of arbitrary digit lists, including leading zeroes. -/
theorem ofDigitChars_digitChars (digits : List Nat) (bounded : ∀ digit ∈ digits, digit < 10) :
    Nat.ofDigitChars 10 (digits.map Nat.digitChar) 0 = Nat.ofDigits 10 digits.reverse := by
  rw [Nat.ofDigitChars_eq_foldl]
  simpa only [Nat.mul_comm] using ProjectEulerStatements.DecimalDigits.foldl_digitChars digits bounded

/-- Reversing the printed nonnegative integer parses as the reversed positional number.
The zero case and leading zeroes of the reversed text are both included. -/
theorem parse_reversed_decimal (n : Nat) :
    FormalSW.Py.Builtin.Int.parseText (String.ofList (toString (n : Int)).toList.reverse) = .ok (reverse n : Int) := by
  by_cases zero : n = 0
  · subst n
    change FormalSW.Py.Builtin.Int.parseText (toString (0 : Nat)) = .ok (0 : Int)
    exact FormalSW.Py.Builtin.Int.parseText_nat 0
  · have positive : 0 < n := Nat.pos_of_ne_zero zero
    have printed : (toString (n : Int)).toList = Nat.toDigits 10 n := by
      change n.repr.toList = Nat.toDigits 10 n
      exact Nat.toList_repr
    have characters : (toString (n : Int)).toList.reverse = (Nat.digits 10 n).map Nat.digitChar := by
      rw [printed, ProjectEulerStatements.DecimalDigits.toDigits_eq_digitChars n positive,
        ← List.map_reverse, List.reverse_reverse]
    have bounded : ∀ digit ∈ Nat.digits 10 n, digit < 10 := by
      intro digit member
      exact Nat.digits_lt_base (b := 10) (m := n) (d := digit) (by decide) member
    have nonempty : (Nat.digits 10 n).map Nat.digitChar ≠ [] := by
      simpa only [ne_eq, List.map_eq_nil_iff] using (Nat.digits_ne_nil_iff_ne_zero (b := 10) (n := n)).mpr zero
    have ascii : ∀ character ∈ (Nat.digits 10 n).map Nat.digitChar, character.isDigit := by
      intro character member
      obtain ⟨digit, present, rfl⟩ := List.mem_map.mp member
      simpa only [Nat.isDigit_digitChar, decide_eq_true_eq] using bounded digit present
    rw [characters, FormalSW.Py.Builtin.Int.parseText_digits _ nonempty ascii, ofDigitChars_digitChars _ bounded]
    rfl

/-- The actual scalar int conversion accepts the whole reversed printed string. -/
theorem integerConversion_reverse (n : Nat) :
    Calls.integerConversion [.string (String.ofList (toString (n : Int)).toList.reverse)] =
      some (.ok (.int (reverse n))) := by
  rw [Calls.integerConversion, parse_reversed_decimal]
  rfl

/-- Reuse the genuine P4 proof to relate decimal-string and decimal-digit palindrome tests. -/
theorem palindrome_string (n : Nat) :
    decide (toString (n : Int) = String.ofList (toString (n : Int)).toList.reverse) = palindrome n := by
  change ProjectEulerSolutions.P4.isPalindrome n = palindrome n
  apply Bool.eq_iff_iff.mpr
  simpa only [palindrome, decide_eq_true_eq, ProjectEulerStatements.P4.isPalindrome] using
    ProjectEulerSolutions.P4.Equiv.palindrome_bool_iff n

end ProjectEulerSolutions.Lychrel
