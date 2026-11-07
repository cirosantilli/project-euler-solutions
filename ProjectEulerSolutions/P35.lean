import ProjectEulerStatements.Lib.CircularPrimeCounting
namespace ProjectEulerSolutions.P35

/-- Count prime rotation classes, marking their members to avoid counting a class twice. -/
def solve (limit : Nat) : Nat :=
  (ProjectEulerStatements.CircularPrimes.algorithmFold limit).2
end ProjectEulerSolutions.P35
