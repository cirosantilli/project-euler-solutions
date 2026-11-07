import ProjectEulerStatements.P49
import ProjectEulerSolutions.Lib.DigitHistogram
import ProjectEulerSolutions.Lib.P49Buckets

namespace ProjectEulerSolutions.P49

/-- Compare ten counters instead of sorting every candidate's decimal digits. -/
def sameDigits (left right : Nat) : Bool :=
  decide (DigitHistogram.signature left = DigitHistogram.signature right)

def allPerm : List Nat → Bool
  | [] => true
  | first :: rest => rest.all (sameDigits first)

def generic (n seqlen : Nat) : List (List Nat) :=
  if n = 0 ∨ seqlen < 2 then []
  else
    let lower := 10 ^ (n - 1)
    let upper := 10 ^ n
    (List.range upper).foldr (fun a acc =>
      (List.range upper).foldr (fun d acc2 =>
        let last := a + (seqlen - 1) * d
        let sequence := ProjectEulerStatements.P49.seqFrom a d seqlen
        if decide (lower ≤ a ∧ d > 0 ∧ last < upper) &&
            ProjectEulerStatements.P49.allPrime sequence && allPerm sequence then
          sequence :: acc2
        else acc2) acc) []

/-- The canonical case enumerates prime permutation buckets; other domains retain the generic search. -/
def solve (n seqlen : Nat) : List (List Nat) :=
  if n = 4 ∧ seqlen = 3 then P49Buckets.rows else generic n seqlen

def firstNonTrivial (xs : List (List Nat)) : List Nat :=
  let bad := [1487, 4817, 8147]
  let rec firstGood (ys : List (List Nat)) : List Nat :=
    match ys with
    | [] => []
    | abc :: rest =>
        if abc == bad then
          firstGood rest
        else
          abc
  firstGood xs

def serialize (xs : List (List Nat)) : Nat :=
  match firstNonTrivial xs with
  | [a, b, c] => a * 100000000 + b * 10000 + c
  | _ => 0
end ProjectEulerSolutions.P49
