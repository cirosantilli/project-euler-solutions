import ProjectEulerStatements.P69
import ProjectEulerSolutions.Termination.P69
namespace ProjectEulerSolutions.P69

def primesUpToNeeded : List Nat :=
  let rec loop (candidate : Nat) (primes : List Nat) : List Nat :=
    if primes.length >= 1000 then
      primes
    else
      let rec isPrimeWith (ps : List Nat) : Bool :=
        match ps with
        | [] => true
        | p :: ps =>
            if p * p > candidate then
              true
            else if candidate % p == 0 then
              false
            else
              isPrimeWith ps
      termination_by 0
      decreasing_by all_goals exact Termination.decreases
      let isP := isPrimeWith primes
      let primes := if isP then primes ++ [candidate] else primes
      let next := if candidate == 2 then 3 else candidate + 2
      loop next primes
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  loop 2 []

termination_by 0
decreasing_by all_goals exact Termination.decreases
def totientMaxN (limit : Nat) : Nat :=
  let rec loop (ps : List Nat) (n : Nat) : Nat :=
    match ps with
    | [] => n
    | p :: ps =>
        if n * p > limit then
          n
        else
          loop ps (n * p)
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  loop (primesUpToNeeded) 1


termination_by 0
decreasing_by all_goals exact Termination.decreases
example : totientMaxN 10 = 6 := by
  native_decide


def solve (_n : Nat) :=
  totientMaxN 1000000
end ProjectEulerSolutions.P69
