import ProjectEulerSolutions.P88
namespace ProjectEulerSolutions.P88

theorem equiv (maxSize : Int) : ProjectEulerStatements.P88.naive maxSize = solve maxSize :=
  (ProjectEulerSolutions.ProductSumNumbers.answer_table maxSize.toNat).symm

end ProjectEulerSolutions.P88
