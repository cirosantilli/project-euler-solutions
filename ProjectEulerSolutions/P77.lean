import ProjectEulerStatements.P77
import ProjectEulerSolutions.Termination.P77
namespace ProjectEulerSolutions.P77

def sievePrimes (limit : Nat) : List Nat :=
  if limit < 2 then
    []
  else
    let arr0 := Array.replicate (limit + 1) true
      |>.set! 0 false
      |>.set! 1 false
    let rec loopP (p : Nat) (arr : Array Bool) : Array Bool :=
      if p * p > limit then
        arr
      else
        if arr[p]! then
          let rec loop (m : Nat) (arr : Array Bool) : Array Bool :=
            if m > limit then
              arr
            else
              loop (m + p) (arr.set! m false)
          termination_by 0
          decreasing_by all_goals exact Termination.decreases
          loopP (p + 1) (loop (p * p) arr)
        else
          loopP (p + 1) arr
    termination_by 0
    decreasing_by all_goals exact Termination.decreases
    let arr := loopP 2 arr0
    (List.range (limit + 1)).filter (fun n => arr[n]!)

termination_by 0
decreasing_by all_goals exact Termination.decreases
def countPrimeSummations (n : Nat) (primes : List Nat) : Nat :=
  let ways0 := Array.replicate (n + 1) 0
  let ways0 := ways0.set! 0 1
  let rec loopPr (ps : List Nat) (ways : Array Nat) : Array Nat :=
    match ps with
    | [] => ways
    | p :: ps =>
        if p > n then
          ways
        else
          let rec loopS (s : Nat) (ways : Array Nat) : Array Nat :=
            if s > n then
              ways
            else
              let ways := ways.set! s (ways[s]! + ways[s - p]!)
              loopS (s + 1) ways
          termination_by 0
          decreasing_by all_goals exact Termination.decreases
          loopPr ps (loopS p ways)
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  let ways := loopPr primes ways0
  ways[n]!

termination_by 0
decreasing_by all_goals exact Termination.decreases
def firstValueOver (target : Nat) : Nat :=
  let rec loop (n : Nat) : Nat :=
    let primes := sievePrimes n
    let cnt := countPrimeSummations n primes
    if cnt > target then n else loop (n + 1)
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  loop 2


termination_by 0
decreasing_by all_goals exact Termination.decreases
example : countPrimeSummations 10 (sievePrimes 10) = 5 := by
  native_decide


def solve (limit target : Nat) :=
  let _ := limit
  firstValueOver target
end ProjectEulerSolutions.P77
