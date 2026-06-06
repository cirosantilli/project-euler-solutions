import ProjectEulerStatements.P72
import ProjectEulerSolutions.Termination.P72
namespace ProjectEulerSolutions.P72

def computePhiLinear (limit : Nat) : Array Nat :=
  let phi0 := Array.replicate (limit + 1) 0
  let comp0 := Array.replicate (limit + 1) false
  let phi0 := if limit >= 1 then phi0.set! 1 1 else phi0
  let rec loop (i : Nat) (phi : Array Nat) (isComp : Array Bool) (primes : Array Nat)
      : Array Nat :=
    if i > limit then
      phi
    else
      let (phi, isComp, primes) :=
        if isComp[i]! then
          (phi, isComp, primes)
        else
          (phi.set! i (i - 1), isComp, primes.push i)
      let rec loopP (j : Nat) (phi : Array Nat) (isComp : Array Bool)
          : Array Nat × Array Bool :=
        if j >= primes.size then
          (phi, isComp)
        else
          let p := primes[j]!
          let ip := i * p
          if ip > limit then
            (phi, isComp)
          else
            let isComp := isComp.set! ip true
            if i % p == 0 then
              let phi := phi.set! ip (phi[i]! * p)
              (phi, isComp)
            else
              let phi := phi.set! ip (phi[i]! * (p - 1))
              loopP (j + 1) phi isComp
      termination_by 0
      decreasing_by all_goals exact Termination.decreases
      let (phi, isComp) := loopP 0 phi isComp
      loop (i + 1) phi isComp primes
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  loop 2 phi0 comp0 #[]

termination_by 0
decreasing_by all_goals exact Termination.decreases
def countReducedProperFractions (maxD : Nat) : Nat :=
  if maxD < 2 then
    0
  else
    let phi := computePhiLinear maxD
    let rec loop (d : Nat) (acc : Nat) : Nat :=
      if d > maxD then
        acc
      else
        loop (d + 1) (acc + phi[d]!)
    termination_by 0
    decreasing_by all_goals exact Termination.decreases
    loop 2 0


termination_by 0
decreasing_by all_goals exact Termination.decreases
example : countReducedProperFractions 8 = 21 := by
  native_decide

example : countReducedProperFractions 1 = 0 := by
  native_decide

example : countReducedProperFractions 2 = 1 := by
  native_decide


def solve (_n : Nat) :=
  countReducedProperFractions 1000000
end ProjectEulerSolutions.P72
