import ProjectEulerStatements.P50

namespace ProjectEulerSolutions.P50
open ConsecutivePrimeSums

/-- Search lengths in descending order and stop at the first prime window. -/
def solve (limit : Nat) : Nat :=
  let primes := ProjectEulerStatements.P50.primesBelow limit
  ((scanLengths primes limit primes.length).map fun chosen =>
    window primes chosen.first chosen.length).getD 0

end ProjectEulerSolutions.P50
