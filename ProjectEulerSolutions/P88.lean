import ProjectEulerStatements.P88
import ProjectEulerStatements.Lib.ProductSumDistinct

namespace ProjectEulerSolutions.P88

/-- Run the factor search once, relaxing the table of all sizes together. -/
def solve (maxSize : Int) : Nat :=
  ((ProjectEulerSolutions.ProductSumNumbers.table maxSize.toNat).drop 2).eraseDups.sum

end ProjectEulerSolutions.P88
