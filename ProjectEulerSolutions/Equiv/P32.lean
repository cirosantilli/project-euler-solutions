import ProjectEulerSolutions.P32
namespace ProjectEulerSolutions.P32

/-- The Python reference guards decimal width, then enumerates permutations and factor splits. -/
def reference (width : Int) : Nat :=
  if width < 3 ∨ 9 < width then 0
  else (PandigitalProductSearch.permutationProducts width.toNat).eraseDups.sum

theorem equiv (width : Int) : reference width = solve width := by
  unfold reference solve
  split
  · rfl
  · apply PandigitalProductSearch.permutation_answer
    omega
end ProjectEulerSolutions.P32
