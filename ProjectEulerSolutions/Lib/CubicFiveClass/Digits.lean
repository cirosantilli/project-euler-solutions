import ProjectEulerSolutions.Lib.CubicClassWitness

/-! The digit histogram identifying the five-cube existence witness. -/
namespace ProjectEulerSolutions.CubicFiveClass
open CubicPermutations CubicSearch
set_option Elab.async false

def digits : List Nat := [1, 1, 1, 2, 1, 2, 1, 1, 1, 1]

theorem example_signature : signature (cube 5027) = digits := by decide

end ProjectEulerSolutions.CubicFiveClass
