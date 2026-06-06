import ProjectEulerStatements.P36
import ProjectEulerSolutions.Termination.P36
namespace ProjectEulerSolutions.P36

def dropLastList (xs : List Char) : List Char :=
  match xs with
  | [] => []
  | [_] => []
  | x :: rest => x :: dropLastList rest

termination_by 0
decreasing_by all_goals exact Termination.decreases
def generateDecimalPalindromes (limit : Nat) : List Nat :=
  let rec loop (i : Nat) (acc : List Nat) : List Nat :=
    if i >= 1000 then
      acc.reverse
    else
      let s := toString i
      let rev := String.mk s.data.reverse
      let oddStr := s ++ String.mk (dropLastList s.data |>.reverse)
      let evenStr := s ++ rev
      let odd := oddStr.toNat!
      let even := evenStr.toNat!
      let acc := if odd < limit then odd :: acc else acc
      let acc := if even < limit then even :: acc else acc
      loop (i + 1) acc
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  loop 1 []

termination_by 0
decreasing_by all_goals exact Termination.decreases
def isPalindromeBase2 (n : Nat) : Bool :=
  let rec bits (m : Nat) (acc : List Nat) : List Nat :=
    if m == 0 then acc else bits (m / 2) ((m % 2) :: acc)
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  let b := bits n []
  b == b.reverse

termination_by 0
decreasing_by all_goals exact Termination.decreases
def solveCore (limit : Nat) : Nat :=
  let pals := generateDecimalPalindromes limit
  let rec loop (lst : List Nat) (total : Nat) : Nat :=
    match lst with
    | [] => total
    | n :: ns =>
        if n % 2 == 0 then
          loop ns total
        else if isPalindromeBase2 n then
          loop ns (total + n)
        else
          loop ns total
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  loop pals 0


termination_by 0
decreasing_by all_goals exact Termination.decreases
example : isPalindromeBase2 585 = true := by
  native_decide


def solve (limit : Nat) :=
  solveCore limit
end ProjectEulerSolutions.P36
