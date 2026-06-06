import ProjectEulerStatements.P47
import ProjectEulerSolutions.Termination.P47
namespace ProjectEulerSolutions.P47

def distinctPrimeFactorCounts (limit : Nat) : Array Nat :=
  let counts := Array.replicate (limit + 1) 0
  let rec loopP (p : Nat) (counts : Array Nat) : Array Nat :=
    if p > limit then
      counts
    else
      if counts[p]! == 0 then
        let rec loopM (m : Nat) (counts : Array Nat) : Array Nat :=
          if m > limit then
            counts
          else
            loopM (m + p) (counts.set! m (counts[m]! + 1))
        termination_by 0
        decreasing_by all_goals exact Termination.decreases
        loopP (p + 1) (loopM p counts)
      else
        loopP (p + 1) counts
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  loopP 2 counts

termination_by 0
decreasing_by all_goals exact Termination.decreases
def firstConsecutiveWithKFactors (k runLen startLimit : Nat) : Nat :=
  let rec loopLimit (limit : Nat) : Nat :=
    let counts := distinctPrimeFactorCounts limit
    let rec loopN (n streak : Nat) : Nat :=
      if n > limit then
        0
      else if counts[n]! == k then
        if streak + 1 == runLen then
          n - runLen + 1
        else
          loopN (n + 1) (streak + 1)
      else
        loopN (n + 1) 0
    termination_by 0
    decreasing_by all_goals exact Termination.decreases
    let found := loopN 2 0
    if found != 0 then found else loopLimit (limit * 2)
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  loopLimit startLimit


termination_by 0
decreasing_by all_goals exact Termination.decreases
example : firstConsecutiveWithKFactors 2 2 100 = 14 := by
  native_decide

example : firstConsecutiveWithKFactors 3 3 2000 = 644 := by
  native_decide


def solve (n : Nat) :=
  let _ := ProjectEulerStatements.P47.exists_consecutive_with_n_factors n
  firstConsecutiveWithKFactors n n (n + 10)
end ProjectEulerSolutions.P47
