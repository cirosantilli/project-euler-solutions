import ProjectEulerSolutions.Lib.PythagoreanArraySieve
import ProjectEulerStatements.Lib.PythagoreanPerimeterCountingProof
namespace ProjectEulerSolutions.P39

/-- Sieve multiples of primitive Pythagorean perimeters, then take the first maximum. -/
def solve (limit : Nat) : Nat :=
  let counts := PythagoreanArraySieve.compute limit
  (List.range' 1 limit).foldl
    (ProjectEulerStatements.ArgMaximum.select (fun p => counts[p]!)) 0
end ProjectEulerSolutions.P39
