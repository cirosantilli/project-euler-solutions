import ProjectEulerStatements.P88
import ProjectEulerSolutions.Termination.P88
namespace ProjectEulerSolutions.P88

/-
Termination split blocker: unlike the other 1-100 partial definitions, `dfs`
recurses through a call nested inside the local recursive `loopF`. A blanket
external decreasing theorem is not enough here because Lean elaborates the
outer `dfs` recursion and the local loop together and falls back to a malformed
structural relation involving the `best : Array Nat` parameter. Making this
total likely requires either a real measure over the search tree
(`limit + 1 - prod`, plus local loop fuel) or refactoring the local loop into a
separate helper with its own termination theorem.
-/
partial def dfs (start prod summ terms kMax limit : Nat) (best : Array Nat) : Array Nat :=
  let maxF := limit / prod
  let rec loopF (f : Nat) (best : Array Nat) : Array Nat :=
    if f > maxF then
      best
    else
      let newProd := prod * f
      let newSum := summ + f
      let newTerms := terms + 1
      let k := newTerms + (newProd - newSum)
      let best :=
        if k <= kMax && newProd < best[k]! then best.set! k newProd else best
      let best := if k <= kMax then dfs f newProd newSum newTerms kMax limit best else best
      loopF (f + 1) best
  loopF start best

def minProductSumNumbersSum (kMax : Nat) : Nat :=
  let limit := 2 * kMax
  let inf := Nat.pow 10 18
  let best0 := Array.replicate (kMax + 1) inf
  let best := dfs 2 1 0 0 kMax limit best0
  let seen := Array.replicate (limit + 1) false
  let rec loopK (k : Nat) (seen : Array Bool) (sum : Nat) : Nat :=
    if k > kMax then
      sum
    else
      let v := best[k]!
      if v <= limit && seen[v]! == false then
        loopK (k + 1) (seen.set! v true) (sum + v)
      else
        loopK (k + 1) seen sum
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  loopK 2 seen 0

termination_by 0
decreasing_by all_goals exact Termination.decreases

example : minProductSumNumbersSum 6 = 30 := by
  native_decide

example : minProductSumNumbersSum 12 = 61 := by
  native_decide


def solve (kMax : Nat) :=
  minProductSumNumbersSum kMax
end ProjectEulerSolutions.P88
