import ProjectEulerSolutions.P100
namespace ProjectEulerSolutions.P100
open ProjectEulerSolutions.ArrangedProbability

private theorem search_eq (limit b t : Int) (hb : 0 < b) (ht : 1 < t) :
    search limit (2*t-1) (2*b-1) (by omega) (by omega) = firstOver limit b t hb ht := by
  rw [search, firstOver]
  have total : (2*t-1+1)/2 = t := by omega
  have blue : (2*b-1+1)/2 = b := by omega
  simp only [total, blue]
  split
  · rfl
  · have u : 3*(2*t-1)+4*(2*b-1) = 2*(4*b+3*t-3)-1 := by ring
    have v : 2*(2*t-1)+3*(2*b-1) = 2*(3*b+2*t-2)-1 := by ring
    simp only [u, v]
    exact search_eq limit _ _ (by omega) (by omega)
termination_by (limit+1-t).toNat
decreasing_by omega

theorem equiv (limit : Int) : ProjectEulerStatements.P100.naive limit = solve limit := by
  unfold ProjectEulerStatements.P100.naive answer solve blueRedForTotalLimit
  exact (congrArg Prod.fst (search_eq limit 3 4 (by omega) (by omega))).symm
end ProjectEulerSolutions.P100
