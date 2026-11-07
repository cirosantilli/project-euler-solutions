import ProjectEulerStatements.Lib.TriangularDivisors
namespace ProjectEulerSolutions.P12
open ProjectEulerStatements.P12

/-- Coprime consecutive factors let us count divisors before multiplying the triangle. -/
def countTriangleDivisors (n : Nat) : Nat :=
  divisorCount (splitLeft n) * divisorCount (splitRight n)

def search (k n : Nat) (before : n ≤ Nat.find (exists_triangle_divisorCount_gt k)) : Nat :=
  if found : k < countTriangleDivisors n then triangle n
  else search k (n+1) (by
    have witness := Nat.find_spec (exists_triangle_divisorCount_gt k)
    have same : countTriangleDivisors n = divisorCount (triangle n) :=
      (divisorCount_triangle_split n).symm
    have : n ≠ Nat.find (exists_triangle_divisorCount_gt k) := by
      intro equal
      rw [← equal] at witness
      exact found (same.symm ▸ witness)
    omega)
termination_by Nat.find (exists_triangle_divisorCount_gt k) - n
decreasing_by
  have failed : ¬ k < countTriangleDivisors n := by assumption
  have witness := Nat.find_spec (exists_triangle_divisorCount_gt k)
  have different : n ≠ Nat.find (exists_triangle_divisorCount_gt k) := by
    intro same
    rw [← same, divisorCount_triangle_split] at witness
    exact failed witness
  omega

def solve (k : Nat) : Nat := search k 1 (firstIndex_positive k)
end ProjectEulerSolutions.P12
