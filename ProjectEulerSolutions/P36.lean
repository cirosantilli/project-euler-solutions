import ProjectEulerStatements.Lib.DoublePalindromeSearch
namespace ProjectEulerSolutions.P36

/-- Enumerate mirrored decimal prefixes until the next odd palindrome reaches the limit. -/
def generateDecimalPalindromes (limit : Nat) : List Nat :=
  ProjectEulerStatements.DoublePalindromeSearch.generate limit

def solve (limit : Nat) : Nat :=
  (generateDecimalPalindromes limit).foldl ProjectEulerStatements.DoublePalindromes.scanStep 0
end ProjectEulerSolutions.P36
