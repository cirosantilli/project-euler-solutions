import ProjectEulerStatements.P76
import ProjectEulerSolutions.Termination.P76
namespace ProjectEulerSolutions.P76

def countSummations (n : Nat) : Nat :=
  let dp0 := Array.replicate (n + 1) 0
  let dp0 := dp0.set! 0 1
  let rec loopPart (part : Nat) (dp : Array Nat) : Array Nat :=
    if part >= n then
      dp
    else
      let rec loopS (s : Nat) (dp : Array Nat) : Array Nat :=
        if s > n then
          dp
        else
          let dp := dp.set! s (dp[s]! + dp[s - part]!)
          loopS (s + 1) dp
      termination_by 0
      decreasing_by all_goals exact Termination.decreases
      loopPart (part + 1) (loopS part dp)
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  let dp := loopPart 1 dp0
  dp[n]!


termination_by 0
decreasing_by all_goals exact Termination.decreases
example : countSummations 5 = 6 := by
  native_decide


def solve (_n : Nat) :=
  countSummations 100
end ProjectEulerSolutions.P76
