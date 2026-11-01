import ProjectEulerSolutions.Lib.LatticeReference
import ProjectEulerSolutions.Lib.LatticeOptimized
namespace ProjectEulerSolutions.P91

theorem equiv (n : Nat) : ProjectEulerStatements.P91.naive n = solve n :=
  (LatticeRightTriangles.natural_reference_eq n).trans
    ((LatticeRightTriangles.referenceCount_eq n).trans (LatticeRightTriangles.natural_optimized_eq n).symm)
end ProjectEulerSolutions.P91
