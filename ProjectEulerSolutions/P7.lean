import ProjectEulerStatements.P7
import ProjectEulerSolutions.Termination.P7
namespace ProjectEulerSolutions.P7

def isPrimeWithLoop (n : Nat) (primes : Array Nat) (i : Nat) : Bool :=
  if i >= primes.size then
    true
  else
    let p := primes[i]!
    if p * p > n then
      true
    else if n % p == 0 then
      false
    else
      isPrimeWithLoop n primes (i + 1)
termination_by primes.size - i
decreasing_by
  omega

def isPrimeWith (n : Nat) (primes : Array Nat) : Bool :=
  isPrimeWithLoop n primes 0

def solve (n : Nat) : Nat :=
  if n == 1 then
    2
  else
    let rec go (candidate count : Nat) (primes : Array Nat) : Nat :=
      if count == n then
        primes[primes.size - 1]!
      else
        if isPrimeWith candidate primes then
          let primes' := primes.push candidate
          go (candidate + 2) (count + 1) primes'
        else
          go (candidate + 2) count primes
    termination_by Termination.goMeasure n candidate count
    decreasing_by
      · exact Termination.go_decreases n candidate count (count + 1)
      · exact Termination.go_decreases n candidate count count
    go 3 1 #[2]

example : solve 1 = 2 := by
  native_decide

example : solve 6 = 13 := by
  native_decide
end ProjectEulerSolutions.P7
