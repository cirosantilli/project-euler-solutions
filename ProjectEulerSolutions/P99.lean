import ProjectEulerStatements.P99
namespace ProjectEulerSolutions.P99
open ProjectEulerSolutions.PowerIntervals

/-- Sign normalization and common-exponent cancellation before adaptive interval refinement. -/
def comparePositive (a : Nat) (m : Int) (b : Nat) (n : Int)
    (ha : 0 < a) (hb : 0 < b) : Int :=
  let m := if a = 1 then 0 else m
  let n := if b = 1 then 0 else n
  if m = 0 ∧ n = 0 then 0
  else if m ≤ 0 ∧ 0 ≤ n then -1
  else if 0 ≤ m ∧ n ≤ 0 then 1
  else if a = b then integerComparison m n
  else
    let left := (if m < 0 then -m else m).toNat
    let right := (if m < 0 then -n else n).toNat
    (if m < 0 then -1 else 1) *
      refine a (left / left.gcd right) b (right / left.gcd right) ha hb 16 (by decide)

def comparePowers (a : Nat) (m : Int) (b : Nat) (n : Int) : Int :=
  if ha : 0 < a then if hb : 0 < b then comparePositive a m b n ha hb else 0 else 0

def scan : List (Nat × Int) → Nat → Nat → (Nat × Int) → Nat
  | [], _, best, _ => best
  | current :: rest, next, best, incumbent =>
      if best = 0 ∨ 0 < comparePowers current.1 current.2 incumbent.1 incumbent.2 then
        scan rest (next + 1) next current
      else scan rest (next + 1) best incumbent

def solve (pairs : List (Nat × Int)) : Nat := scan pairs 1 0 (1, 0)

def parsePairs (text : String) : List (Nat × Int) :=
  (text.splitOn "\n" |>.filter (· != "")).map fun line =>
    let parts := line.splitOn ","
    ((parts[0]?.getD "0").toNat?.getD 0, (parts[1]?.getD "0").toInt?.getD 0)
end ProjectEulerSolutions.P99
