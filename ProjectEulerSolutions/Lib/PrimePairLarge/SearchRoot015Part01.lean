import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot015Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root15Candidates32 : List Nat := [5821, 7681, 8311, 16567]
set_option maxRecDepth 8192 in
theorem root15Candidates32_checked : candidates edge 4 61 5623 root15Tail33 winner.bound = root15Candidates32 := by
  unfold candidates
  have row : edge 5623 = (SortedNatTree.contains FastRows.row5623) := by
    change Data.row5623.contains = (SortedNatTree.contains FastRows.row5623)
    exact FastRows.row5623_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root15Child32 : search edge 3 [61, 5623] root15Candidates32 5684 winner = winner := by decide
theorem root15Step32 : scan edge (search edge 3) 4 [61] 61 root15Tail32 winner =
    scan edge (search edge 3) 4 [61] 61 root15Tail33 winner := by
  rw [root15Tail32, scan, if_neg (by decide), root15Candidates32_checked]
  change scan edge (search edge 3) 4 [61] 61 root15Tail33
    (if 3 ≤ root15Candidates32.length then search edge 3 [61, 5623] root15Candidates32 5684 winner else winner) = _
  rw [if_pos (by decide), root15Child32]
def root15Candidates33 : List Nat := []
set_option maxRecDepth 8192 in
theorem root15Candidates33_checked : candidates edge 4 61 5821 root15Tail34 winner.bound = root15Candidates33 := by
  unfold candidates
  have row : edge 5821 = (SortedNatTree.contains FastRows.row5821) := by
    change Data.row5821.contains = (SortedNatTree.contains FastRows.row5821)
    exact FastRows.row5821_checked.symm
  rw [row]
  decide
theorem root15Step33 : scan edge (search edge 3) 4 [61] 61 root15Tail33 winner =
    scan edge (search edge 3) 4 [61] 61 root15Tail34 winner := by
  rw [root15Tail33, scan, if_neg (by decide), root15Candidates33_checked]
  change scan edge (search edge 3) 4 [61] 61 root15Tail34
    (if 3 ≤ root15Candidates33.length then search edge 3 [61, 5821] root15Candidates33 5882 winner else winner) = _
  rw [if_neg (by decide)]
def root15Candidates34 : List Nat := []
set_option maxRecDepth 8192 in
theorem root15Candidates34_checked : candidates edge 4 61 5827 root15Tail35 winner.bound = root15Candidates34 := by
  unfold candidates
  have row : edge 5827 = (SortedNatTree.contains FastRows.row5827) := by
    change Data.row5827.contains = (SortedNatTree.contains FastRows.row5827)
    exact FastRows.row5827_checked.symm
  rw [row]
  decide
theorem root15Step34 : scan edge (search edge 3) 4 [61] 61 root15Tail34 winner =
    scan edge (search edge 3) 4 [61] 61 root15Tail35 winner := by
  rw [root15Tail34, scan, if_neg (by decide), root15Candidates34_checked]
  change scan edge (search edge 3) 4 [61] 61 root15Tail35
    (if 3 ≤ root15Candidates34.length then search edge 3 [61, 5827] root15Candidates34 5888 winner else winner) = _
  rw [if_neg (by decide)]
def root15Candidates35 : List Nat := [11839]
set_option maxRecDepth 8192 in
theorem root15Candidates35_checked : candidates edge 4 61 5869 root15Tail36 winner.bound = root15Candidates35 := by
  unfold candidates
  have row : edge 5869 = (SortedNatTree.contains FastRows.row5869) := by
    change Data.row5869.contains = (SortedNatTree.contains FastRows.row5869)
    exact FastRows.row5869_checked.symm
  rw [row]
  decide
theorem root15Step35 : scan edge (search edge 3) 4 [61] 61 root15Tail35 winner =
    scan edge (search edge 3) 4 [61] 61 root15Tail36 winner := by
  rw [root15Tail35, scan, if_neg (by decide), root15Candidates35_checked]
  change scan edge (search edge 3) 4 [61] 61 root15Tail36
    (if 3 ≤ root15Candidates35.length then search edge 3 [61, 5869] root15Candidates35 5930 winner else winner) = _
  rw [if_neg (by decide)]
def root15Candidates36 : List Nat := [11527, 13309]
set_option maxRecDepth 8192 in
theorem root15Candidates36_checked : candidates edge 4 61 6079 root15Tail37 winner.bound = root15Candidates36 := by
  unfold candidates
  have row : edge 6079 = (SortedNatTree.contains FastRows.row6079) := by
    change Data.row6079.contains = (SortedNatTree.contains FastRows.row6079)
    exact FastRows.row6079_checked.symm
  rw [row]
  decide
theorem root15Step36 : scan edge (search edge 3) 4 [61] 61 root15Tail36 winner =
    scan edge (search edge 3) 4 [61] 61 root15Tail37 winner := by
  rw [root15Tail36, scan, if_neg (by decide), root15Candidates36_checked]
  change scan edge (search edge 3) 4 [61] 61 root15Tail37
    (if 3 ≤ root15Candidates36.length then search edge 3 [61, 6079] root15Candidates36 6140 winner else winner) = _
  rw [if_neg (by decide)]
def root15Candidates37 : List Nat := [6793, 10861, 11587, 16111]
set_option maxRecDepth 8192 in
theorem root15Candidates37_checked : candidates edge 4 61 6367 root15Tail38 winner.bound = root15Candidates37 := by
  unfold candidates
  have row : edge 6367 = (SortedNatTree.contains FastRows.row6367) := by
    change Data.row6367.contains = (SortedNatTree.contains FastRows.row6367)
    exact FastRows.row6367_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root15Child37 : search edge 3 [61, 6367] root15Candidates37 6428 winner = winner := by decide
theorem root15Step37 : scan edge (search edge 3) 4 [61] 61 root15Tail37 winner =
    scan edge (search edge 3) 4 [61] 61 root15Tail38 winner := by
  rw [root15Tail37, scan, if_neg (by decide), root15Candidates37_checked]
  change scan edge (search edge 3) 4 [61] 61 root15Tail38
    (if 3 ≤ root15Candidates37.length then search edge 3 [61, 6367] root15Candidates37 6428 winner else winner) = _
  rw [if_pos (by decide), root15Child37]
end ProjectEulerSolutions.PrimePairLarge.Search
