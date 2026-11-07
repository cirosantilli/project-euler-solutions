import ProjectEulerSolutions.P44
namespace ProjectEulerSolutions.P44
open PentagonalSearch PentagonalDiscriminant

theorem scanRow_correct (k count best : Nat) : scanRow k count best = row k count best := by
  induction count generalizing best with
  | zero => rfl
  | succ j ih =>
    rw [scanRow]
    split
    · rename_i cut
      exact (prune k (j+1) best cut).symm
    · rename_i improve
      simp only [correct, ih]
      rw [row_succ]
      have smaller : min best (difference k (j+1)) = difference k (j+1) := by omega
      simp only [accepted, smaller]

/-- `search` is the exhaustive bounded-pair reference, including Python's no-match sentinel. -/
theorem equiv (limit : Nat) : PentagonalSearch.search limit = solve limit := by
  simp only [solve, scanRow_correct, ← skip_first]
end ProjectEulerSolutions.P44
