import ProjectEulerSolutions.P31
namespace ProjectEulerSolutions.P31

theorem countWays_correct (target : Nat) (coins : List Nat)
    (positive : ∀ c ∈ coins, 0 < c) :
    countWays target coins = ProjectEulerStatements.P31.countWaysBound target coins := by
  unfold countWays
  rw [CoinChangeArray.compute_encode target coins positive]
  simp [CoinChangeArray.encode, ProjectEulerStatements.CoinChange.countWaysBound_reverse target coins positive]

theorem equiv (n : Nat) : ProjectEulerStatements.P31.naive n = solve n := by
  exact (countWays_correct n [1, 2, 5, 10, 20, 50, 100, 200] (by simp)).symm
end ProjectEulerSolutions.P31
