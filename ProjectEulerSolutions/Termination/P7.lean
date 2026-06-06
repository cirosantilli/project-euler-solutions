namespace ProjectEulerSolutions.P7.Termination

/-
The difficult part is not showing that each recursive branch reduces some
syntactic counter; it is proving a useful bound for the prime search. The loop
increments `candidate` until `count = n`, and termination needs an external
fact that the first `n` primes are found below a bound such as `2 * n * n + 10`.
That bound ultimately depends on Euclid-style infinitude of primes plus enough
arithmetic to connect `isPrimeWith` and the accumulated prime array to
`Nat.Prime`. Until those invariants are proved, this module isolates the proof
gap away from the executable definition.
-/
def goMeasure (n candidate count : Nat) : Nat :=
  (n + 1 - count) * (2 * n * n + 11) + ((2 * n * n + 10) - candidate)

theorem go_decreases (n candidate count nextCount : Nat) :
    goMeasure n (candidate + 2) nextCount < goMeasure n candidate count := by
  sorry

end ProjectEulerSolutions.P7.Termination
