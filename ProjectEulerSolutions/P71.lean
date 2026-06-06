import ProjectEulerStatements.P71
import ProjectEulerSolutions.Termination.P71
namespace ProjectEulerSolutions.P71

def gcd (a b : Nat) : Nat :=
  if b == 0 then a else gcd b (a % b)

termination_by 0
decreasing_by all_goals exact Termination.decreases
def bestFractionLeftOf (targetN targetD maxD : Nat) : Nat × Nat :=
  let rec loop (d : Nat) (bestN bestD : Nat) : Nat × Nat :=
    if d > maxD then
      (bestN, bestD)
    else
      let n := (targetN * d - 1) / targetD
      if n == 0 then
        loop (d + 1) bestN bestD
      else if gcd n d != 1 then
        loop (d + 1) bestN bestD
      else if n * bestD > bestN * d then
        loop (d + 1) n d
      else
        loop (d + 1) bestN bestD
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  loop 1 0 1


termination_by 0
decreasing_by all_goals exact Termination.decreases
example : bestFractionLeftOf 3 7 8 = (2, 5) := by
  native_decide


def solve (_n : Nat) :=
  (bestFractionLeftOf 3 7 1000000).1
end ProjectEulerSolutions.P71
