import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot049Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root49Candidates32 : List Nat := []
set_option maxRecDepth 8192 in
theorem root49Candidates32_checked : candidates edge 4 239 6257 root49Tail33 winner.bound = root49Candidates32 := by
  unfold candidates
  have row : edge 6257 = (SortedNatTree.contains FastRows.row6257) := by
    change Data.row6257.contains = (SortedNatTree.contains FastRows.row6257)
    exact FastRows.row6257_checked.symm
  rw [row]
  decide
theorem root49Step32 : scan edge (search edge 3) 4 [239] 239 root49Tail32 winner =
    scan edge (search edge 3) 4 [239] 239 root49Tail33 winner := by
  rw [root49Tail32, scan, if_neg (by decide), root49Candidates32_checked]
  change scan edge (search edge 3) 4 [239] 239 root49Tail33
    (if 3 ≤ root49Candidates32.length then search edge 3 [239, 6257] root49Candidates32 6496 winner else winner) = _
  rw [if_neg (by decide)]
end ProjectEulerSolutions.PrimePairLarge.Search
