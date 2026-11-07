import ProjectEulerSolutions.Lib.P24PermutationRank
namespace ProjectEulerSolutions.P24

def serialize (digits : List Int) : String := String.intercalate "" (digits.map toString)

/-- Factoradic selection uses each chosen position once and preserves the remaining order. -/
def solve (digits : List Int) (index : Int) : String :=
  serialize (PermutationRank.unrank digits.length digits (index-1).toNat)
end ProjectEulerSolutions.P24
