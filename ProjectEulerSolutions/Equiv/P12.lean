import ProjectEulerSolutions.P12
namespace ProjectEulerSolutions.P12
open ProjectEulerStatements.P12

theorem search_correct (k n : Nat) (before : n ≤ Nat.find (exists_triangle_divisorCount_gt k)) :
    search k n before = firstTriangleOver k := by
  rw [search]
  split
  · rename_i found
    have same : countTriangleDivisors n = divisorCount (triangle n) :=
      (divisorCount_triangle_split n).symm
    have least := Nat.find_min' (exists_triangle_divisorCount_gt k) (same ▸ found)
    have equal : n = Nat.find (exists_triangle_divisorCount_gt k) := by omega
    simp [firstTriangleOver, equal]
  · exact search_correct k (n+1) _
termination_by Nat.find (exists_triangle_divisorCount_gt k) - n
decreasing_by
  have failed : ¬ k < countTriangleDivisors n := by assumption
  have witness := Nat.find_spec (exists_triangle_divisorCount_gt k)
  have different : n ≠ Nat.find (exists_triangle_divisorCount_gt k) := by
    intro same
    rw [← same, divisorCount_triangle_split] at witness
    exact failed witness
  omega

theorem equiv (n : Nat) : ProjectEulerStatements.P12.naive n = solve n := by
  rw [naive_eq_firstTriangleOver]
  exact (search_correct n 1 _).symm
end ProjectEulerSolutions.P12
