import ProjectEulerStatements.P10
namespace ProjectEulerSolutions.P10

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

def go (limit candidate sum : Nat) (primes : Array Nat) : Nat :=
  if candidate >= limit then
    sum
  else
    if isPrimeWith candidate primes then
      go limit (candidate + 2) (sum + candidate) (primes.push candidate)
    else
      go limit (candidate + 2) sum primes
termination_by limit + 1 - candidate
decreasing_by
  all_goals omega

def solve (limit : Nat) : Nat :=
  if limit <= 2 then
    0
  else if limit <= 3 then
    2
  else
    go limit 3 2 #[2]

example : solve 10 = 17 := by
  native_decide
end ProjectEulerSolutions.P10
