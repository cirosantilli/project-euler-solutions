import ProjectEulerSolutions.P39
namespace ProjectEulerSolutions.P39
open ProjectEulerStatements.PythagoreanPerimeterCounting

theorem equiv (n : Nat) : ProjectEulerStatements.P39.naive n = solve n := by
  unfold solve
  rw [PythagoreanArraySieve.compute_encode]
  have fold (items : List Nat) (bounded : ∀ p ∈ items, p ≤ n) (best : Nat) (hb : best ≤ n) :
      items.foldl (ProjectEulerStatements.ArgMaximum.select
        (fun p => (PythagoreanArraySieve.encode (n+1) (table n))[p]!)) best =
      items.foldl (ProjectEulerStatements.ArgMaximum.select (table n)) best := by
    induction items generalizing best with
    | nil => rfl
    | cons head tail ih =>
      have hh := bounded head (by simp)
      have ht : ∀ p ∈ tail, p ≤ n := fun p hp => bounded p (by simp [hp])
      simp only [List.foldl_cons, ProjectEulerStatements.ArgMaximum.select]
      have get (p : Nat) (hp : p ≤ n) : (PythagoreanArraySieve.encode (n+1) (table n))[p]! = table n p := by
        simp [PythagoreanArraySieve.encode, show p < n+1 by omega]
      simp only [get best hb, get head hh]
      split
      · exact ih ht head hh
      · exact ih ht best hb
  rw [fold _ (by intro p hp; have := List.mem_range'.mp hp; omega) 0 (by omega)]
  exact (table_scan n).symm
end ProjectEulerSolutions.P39
