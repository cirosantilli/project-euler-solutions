import ProjectEulerSolutions.Lib.AnagramPairs
import ProjectEulerSolutions.Lib.AnagramAssignment
import Init.Data.Nat.ToString

/-! Exhaustive square assignments for distinct anagram pairs. -/
namespace ProjectEulerSolutions.AnagramSearch
open AnagramSquares
set_option Elab.async false

def squareText (root : Nat) : String := toString (root * root)

def transformed (word digits other : List Char) : Nat :=
  Nat.ofDigitChars 10 (other.map (substitution '0' word digits)) 0

def step (word other : String) (root : Nat) (best : Int) : Int :=
  let digits := (squareText root).toList
  match other.toList with
  | [] => best
  | first :: _ =>
      if labels [] word.toList = labels [] digits then
        if substitution '0' word.toList digits first = '0' then best
        else
          let n := transformed word.toList digits other.toList
          if n.sqrt ^ 2 = n then max (max best (root * root)) n else best
      else best

def scanRoots (word other : String) (first : Nat) : Nat → Int → Int
  | 0, best => best
  | count + 1, best => scanRoots word other (first + 1) count (step word other first best)

def scanPairs : List AnagramPairs.Pair → Int → Int
  | [], best => best
  | (word, other) :: rest, best =>
      scanPairs rest (scanRoots word other (firstRoot word.length)
        (lastRoot word.length + 1 - firstRoot word.length) best)

def solve (words : List String) : Int := scanPairs (AnagramPairs.pairs words) 0

theorem ordered_roots (width : Nat) : firstRoot width ≤ lastRoot width + 1 := by
  have powers : 10 ^ (width - 1) ≤ 10 ^ width := Nat.pow_le_pow_right (by decide) (by omega)
  have roots := Nat.sqrt_le_sqrt (Nat.sub_le_sub_right powers 1)
  exact Nat.add_le_add_right roots 1

/-- Every square in the source interval prints with exactly the requested number of digits. -/
theorem square_width (width root : Nat) (positive : 0 < width)
    (lower : firstRoot width ≤ root) (upper : root ≤ lastRoot width) :
    (squareText root).toList.length = width := by
  obtain ⟨lo, hi⟩ := (root_interval width root).mp ⟨lower, upper⟩
  have printed : (squareText root).toList.length = (root * root).repr.length := by
    simp [squareText, Nat.repr_eq_ofList_toDigits]
  rw [printed]
  have atMost := (Nat.length_repr_le_iff positive).mpr hi
  have atLeast : width ≤ (root * root).repr.length := by
    by_contra short
    by_cases one : width = 1
    · have nonempty := Nat.length_repr_pos (n := root * root)
      omega
    · have shorter := (Nat.length_repr_le_iff (show 0 < width - 1 by omega)).mp
        (show (root * root).repr.length ≤ width - 1 by omega)
      omega
  omega

theorem square_decimal (root : Nat) (digit : Char) (member : digit ∈ (squareText root).toList) : digit.isDigit := by
  have printed : (squareText root).toList = Nat.toDigits 10 (root * root) := Nat.toList_repr
  rw [printed] at member
  exact Nat.isDigit_of_mem_toDigits (by decide) (by decide) member

end ProjectEulerSolutions.AnagramSearch
