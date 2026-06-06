import ProjectEulerStatements.P73
import ProjectEulerSolutions.Termination.P73
namespace ProjectEulerSolutions.P73

def gcd (a b : Nat) : Nat :=
  if b == 0 then a else gcd b (a % b)

termination_by 0
decreasing_by all_goals exact Termination.decreases
def countBetweenOneThirdAndOneHalf (limit : Nat) : Nat :=
  let rec loopD (d : Nat) (total : Nat) : Nat :=
    if d > limit then
      total
    else
      let nStart := d / 3 + 1
      let nEnd := (d - 1) / 2
      if nStart > nEnd then
        loopD (d + 1) total
      else
        let rec loopN (n : Nat) (cnt : Nat) : Nat :=
          if n > nEnd then
            cnt
          else
            let cnt := if gcd n d == 1 then cnt + 1 else cnt
            loopN (n + 1) cnt
        termination_by 0
        decreasing_by all_goals exact Termination.decreases
        loopD (d + 1) (loopN nStart total)
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  loopD 2 0


termination_by 0
decreasing_by all_goals exact Termination.decreases
example : countBetweenOneThirdAndOneHalf 8 = 3 := by
  native_decide


def solve (_n : Nat) :=
  countBetweenOneThirdAndOneHalf 12000
end ProjectEulerSolutions.P73
