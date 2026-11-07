import ProjectEulerSolutions.Lib.PandigitalProductPermutations
namespace ProjectEulerSolutions.P32

/-- Restrict factor pairs by the possible product width, then deduplicate products. -/
def solve (width : Int) : Nat :=
  if width < 3 ∨ 9 < width then 0 else PandigitalProductSearch.answer width.toNat
end ProjectEulerSolutions.P32
