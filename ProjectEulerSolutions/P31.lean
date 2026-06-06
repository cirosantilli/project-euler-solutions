import ProjectEulerStatements.P31
import ProjectEulerSolutions.Termination.P31
namespace ProjectEulerSolutions.P31

def countWays (target : Nat) (coins : List Nat) : Nat :=
  let dp0 := (Array.replicate (target + 1) 0).set! 0 1
  let rec loopCoins (cs : List Nat) (dp : Array Nat) : Array Nat :=
    match cs with
    | [] => dp
    | c :: rest =>
        let rec loopSum (s : Nat) (dp : Array Nat) : Array Nat :=
          if s > target then
            dp
          else
            let dp' := dp.set! s (dp[s]! + dp[s - c]!)
            loopSum (s + 1) dp'
        termination_by 0
        decreasing_by all_goals exact Termination.decreases
        loopCoins rest (loopSum c dp)
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  let dp := loopCoins coins dp0
  dp[target]!


termination_by 0
decreasing_by all_goals exact Termination.decreases
example : countWays 0 [1, 2, 5, 10, 20, 50, 100, 200] = 1 := by
  native_decide

example : countWays 1 [1, 2, 5, 10, 20, 50, 100, 200] = 1 := by
  native_decide

example : countWays 2 [1, 2, 5, 10, 20, 50, 100, 200] = 2 := by
  native_decide

example : countWays 5 [1, 2, 5, 10, 20, 50, 100, 200] = 4 := by
  native_decide

example : countWays 10 [1, 2, 5, 10, 20, 50, 100, 200] = 11 := by
  native_decide


def solve (amt : Nat) :=
  let coins := [1, 2, 5, 10, 20, 50, 100, 200]
  countWays amt coins
end ProjectEulerSolutions.P31
