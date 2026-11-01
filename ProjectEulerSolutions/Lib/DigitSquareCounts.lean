import ProjectEulerSolutions.Lib.DigitSquareReachability

namespace ProjectEulerSolutions.DigitSquareChains
set_option Elab.async false

/-- Count positive starts strictly below the signed input bound. -/
def count (limit : Int) : Nat :=
  ((List.range' 1 (limit - 1).toNat).filter endsAt89).length

theorem count_nonpositive (limit : Int) (small : limit ≤ 1) : count limit = 0 := by
  have empty : (limit - 1).toNat = 0 := by omega
  simp [count, empty]

end ProjectEulerSolutions.DigitSquareChains
