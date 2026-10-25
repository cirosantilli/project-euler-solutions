import ProjectEulerStatements.Lib.DecimalDigitSum
import Mathlib.Data.List.Range

/-! Finite nested searches for the maximum decimal digit sum of a power. -/
namespace ProjectEulerSolutions.PowerDigitSum

set_option Elab.async false

def digitSum (n : Nat) : Nat := (Nat.digits 10 n).sum

def score (a b : Nat) : Int := digitSum (a ^ b)

def row (a : Nat) (bs : List Nat) (best : Int) : Int :=
  bs.foldl (fun best b => max best (score a b)) best

def scan (bs as : List Nat) (best : Int) : Int :=
  as.foldl (fun best a => row a bs best) best

def maximum (limit : Nat) : Int := scan (List.range limit) (List.range limit) 0

@[simp] theorem row_nil (a : Nat) (best : Int) : row a [] best = best := rfl
@[simp] theorem row_cons (a b : Nat) (bs : List Nat) (best : Int) :
    row a (b :: bs) best = row a bs (max best (score a b)) := rfl
@[simp] theorem scan_nil (bs : List Nat) (best : Int) : scan bs [] best = best := rfl
@[simp] theorem scan_cons (bs : List Nat) (a : Nat) (as : List Nat) (best : Int) :
    scan bs (a :: as) best = scan bs as (row a bs best) := rfl

end ProjectEulerSolutions.PowerDigitSum


namespace ProjectEulerSolutions.PowerDigitSum
set_option Elab.async false

/-- The actual ord-minus-48 expression agrees with natural digit subtraction on ASCII digits. -/
theorem ordinal_digit (character : Char) (digit : character.isDigit) :
    (character.toNat : Int) - 48 = ((character.toNat - 48 : Nat) : Int) := by
  have bounded : 48 ≤ character.toNat := (Char.isDigit_iff_toNat.mp digit).1
  exact (Int.natCast_sub bounded).symm

/-- Integer and natural digit accumulators coincide because every printed character is a digit. -/
theorem fold_ordinal_digits (characters : List Char) (initial : Nat)
    (digits : ∀ character ∈ characters, character.isDigit) :
    characters.foldl (fun (total : Int) character => total + ((character.toNat : Int) - 48)) initial =
      ((characters.foldl (fun (total : Nat) character => total + (character.toNat - 48)) initial : Nat) : Int) := by
  induction characters generalizing initial with
  | nil => rfl
  | cons character rest ih =>
      have head := digits character List.mem_cons_self
      have tail : ∀ item ∈ rest, item.isDigit := fun item member => digits item (List.mem_cons_of_mem character member)
      simpa only [List.foldl_cons, ordinal_digit character head, Int.natCast_add] using
        ih (initial + (character.toNat - 48)) tail

theorem printed_characters (n : Nat) : (toString (n : Int)).toList = Nat.toDigits 10 n := by
  change n.repr.toList = Nat.toDigits 10 n
  exact Nat.toList_repr

theorem printed_ascii (n : Nat) : ∀ character ∈ (toString (n : Int)).toList, character.isDigit := by
  intro character member
  rw [printed_characters] at member
  exact Nat.isDigit_of_mem_toDigits (by decide) (by decide) member

/-- Summing the real printed characters with integer ord-minus-48 computes the positional digit sum, including zero. -/
theorem printed_digit_sum (n : Nat) :
    (toString (n : Int)).toList.foldl
        (fun (total : Int) character => total + ((character.toNat : Int) - 48)) 0 = (digitSum n : Int) := by
  calc
    _ = (((toString (n : Int)).toList.foldl
        (fun (total : Nat) character => total + (character.toNat - 48)) 0 : Nat) : Int) := by
      simpa only [Int.natCast_zero] using fold_ordinal_digits _ 0 (printed_ascii n)
    _ = _ := by
      change (((toString n).toList.foldl
        (fun (total : Nat) character => total + (character.toNat - '0'.toNat)) 0 : Nat) : Int) = _
      exact congrArg (fun value : Nat => (value : Int)) (ProjectEulerStatements.DecimalDigits.sum_string_digits n)

private theorem fold_cast (digits : List Nat) (initial : Nat) :
    (digits.map (fun (digit : Nat) => (digit : Int))).foldl Int.add initial =
      ((digits.foldl Nat.add initial : Nat) : Int) := by
  induction digits generalizing initial with
  | nil => rfl
  | cons digit rest ih =>
      simpa only [List.map_cons, List.foldl_cons, Int.natCast_add] using ih (initial+digit)

/-- The reference source's integer list summation agrees with the natural digit sum. -/
theorem integer_digits_sum (digits : List Nat) :
    (digits.map (fun (digit : Nat) => (digit : Int))).foldl Int.add 0 = (digits.sum : Int) := by
  calc
    _ = ((digits.foldl Nat.add 0 : Nat) : Int) := by
      simpa only [Int.natCast_zero] using fold_cast digits 0
    _ = _ := congrArg (fun value : Nat => (value : Int))
      (ProjectEulerStatements.DecimalDigits.foldl_add_eq_sum digits)

end ProjectEulerSolutions.PowerDigitSum
