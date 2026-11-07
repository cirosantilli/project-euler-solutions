import ProjectEulerSolutions.P92
namespace ProjectEulerSolutions.P92
open ProjectEulerSolutions.DigitSquareChains

private theorem distribution_exact (bound : Nat) : Prefix.Exact bound (distribution bound) := by
  rw [distribution]
  split
  · rename_i zero
    subst bound
    exact Prefix.initial_exact
  · have step := (distribution_exact (bound / 10)).transition (bound % 10) (by omega)
    simpa [Nat.div_add_mod] using step
termination_by bound
decreasing_by omega

theorem equiv (limit : Int) : ProjectEulerStatements.P92.naive limit = solve limit := by
  unfold solve ProjectEulerStatements.P92.naive
  split
  · exact count_nonpositive limit (by assumption)
  · have bound : (((limit - 1).toNat + 1 : Nat) : Int) = limit := by omega
    simpa only [bound] using (distribution_exact (limit - 1).toNat).answer.symm
end ProjectEulerSolutions.P92
