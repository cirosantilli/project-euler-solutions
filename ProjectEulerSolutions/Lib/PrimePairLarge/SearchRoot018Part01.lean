import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot018Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root18Candidates32 : List Nat := [5953, 6121, 8677, 18757]
set_option maxRecDepth 8192 in
theorem root18Candidates32_checked : candidates edge 4 73 5659 root18Tail33 winner.bound = root18Candidates32 := by
  unfold candidates
  have row : edge 5659 = (SortedNatTree.contains FastRows.row5659) := by
    change Data.row5659.contains = (SortedNatTree.contains FastRows.row5659)
    exact FastRows.row5659_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root18Child32 : search edge 3 [73, 5659] root18Candidates32 5732 winner = winner := by decide
theorem root18Step32 : scan edge (search edge 3) 4 [73] 73 root18Tail32 winner =
    scan edge (search edge 3) 4 [73] 73 root18Tail33 winner := by
  rw [root18Tail32, scan, if_neg (by decide), root18Candidates32_checked]
  change scan edge (search edge 3) 4 [73] 73 root18Tail33
    (if 3 ≤ root18Candidates32.length then search edge 3 [73, 5659] root18Candidates32 5732 winner else winner) = _
  rw [if_pos (by decide), root18Child32]
def root18Candidates33 : List Nat := [6121, 9397]
set_option maxRecDepth 8192 in
theorem root18Candidates33_checked : candidates edge 4 73 5821 root18Tail34 winner.bound = root18Candidates33 := by
  unfold candidates
  have row : edge 5821 = (SortedNatTree.contains FastRows.row5821) := by
    change Data.row5821.contains = (SortedNatTree.contains FastRows.row5821)
    exact FastRows.row5821_checked.symm
  rw [row]
  decide
theorem root18Step33 : scan edge (search edge 3) 4 [73] 73 root18Tail33 winner =
    scan edge (search edge 3) 4 [73] 73 root18Tail34 winner := by
  rw [root18Tail33, scan, if_neg (by decide), root18Candidates33_checked]
  change scan edge (search edge 3) 4 [73] 73 root18Tail34
    (if 3 ≤ root18Candidates33.length then search edge 3 [73, 5821] root18Candidates33 5894 winner else winner) = _
  rw [if_neg (by decide)]
def root18Candidates34 : List Nat := [6277]
set_option maxRecDepth 8192 in
theorem root18Candidates34_checked : candidates edge 4 73 5953 root18Tail35 winner.bound = root18Candidates34 := by
  unfold candidates
  have row : edge 5953 = (SortedNatTree.contains FastRows.row5953) := by
    change Data.row5953.contains = (SortedNatTree.contains FastRows.row5953)
    exact FastRows.row5953_checked.symm
  rw [row]
  decide
theorem root18Step34 : scan edge (search edge 3) 4 [73] 73 root18Tail34 winner =
    scan edge (search edge 3) 4 [73] 73 root18Tail35 winner := by
  rw [root18Tail34, scan, if_neg (by decide), root18Candidates34_checked]
  change scan edge (search edge 3) 4 [73] 73 root18Tail35
    (if 3 ≤ root18Candidates34.length then search edge 3 [73, 5953] root18Candidates34 6026 winner else winner) = _
  rw [if_neg (by decide)]
def root18Candidates35 : List Nat := [13693]
set_option maxRecDepth 8192 in
theorem root18Candidates35_checked : candidates edge 4 73 6091 root18Tail36 winner.bound = root18Candidates35 := by
  unfold candidates
  have row : edge 6091 = (SortedNatTree.contains FastRows.row6091) := by
    change Data.row6091.contains = (SortedNatTree.contains FastRows.row6091)
    exact FastRows.row6091_checked.symm
  rw [row]
  decide
theorem root18Step35 : scan edge (search edge 3) 4 [73] 73 root18Tail35 winner =
    scan edge (search edge 3) 4 [73] 73 root18Tail36 winner := by
  rw [root18Tail35, scan, if_neg (by decide), root18Candidates35_checked]
  change scan edge (search edge 3) 4 [73] 73 root18Tail36
    (if 3 ≤ root18Candidates35.length then search edge 3 [73, 6091] root18Candidates35 6164 winner else winner) = _
  rw [if_neg (by decide)]
def root18Candidates36 : List Nat := [11287]
set_option maxRecDepth 8192 in
theorem root18Candidates36_checked : candidates edge 4 73 6121 root18Tail37 winner.bound = root18Candidates36 := by
  unfold candidates
  have row : edge 6121 = (SortedNatTree.contains FastRows.row6121) := by
    change Data.row6121.contains = (SortedNatTree.contains FastRows.row6121)
    exact FastRows.row6121_checked.symm
  rw [row]
  decide
theorem root18Step36 : scan edge (search edge 3) 4 [73] 73 root18Tail36 winner =
    scan edge (search edge 3) 4 [73] 73 root18Tail37 winner := by
  rw [root18Tail36, scan, if_neg (by decide), root18Candidates36_checked]
  change scan edge (search edge 3) 4 [73] 73 root18Tail37
    (if 3 ≤ root18Candidates36.length then search edge 3 [73, 6121] root18Candidates36 6194 winner else winner) = _
  rw [if_neg (by decide)]
def root18Candidates37 : List Nat := []
set_option maxRecDepth 8192 in
theorem root18Candidates37_checked : candidates edge 4 73 6277 root18Tail38 winner.bound = root18Candidates37 := by
  unfold candidates
  have row : edge 6277 = (SortedNatTree.contains FastRows.row6277) := by
    change Data.row6277.contains = (SortedNatTree.contains FastRows.row6277)
    exact FastRows.row6277_checked.symm
  rw [row]
  decide
theorem root18Step37 : scan edge (search edge 3) 4 [73] 73 root18Tail37 winner =
    scan edge (search edge 3) 4 [73] 73 root18Tail38 winner := by
  rw [root18Tail37, scan, if_neg (by decide), root18Candidates37_checked]
  change scan edge (search edge 3) 4 [73] 73 root18Tail38
    (if 3 ≤ root18Candidates37.length then search edge 3 [73, 6277] root18Candidates37 6350 winner else winner) = _
  rw [if_neg (by decide)]
def root18Candidates38 : List Nat := [6793]
set_option maxRecDepth 8192 in
theorem root18Candidates38_checked : candidates edge 4 73 6367 root18Tail39 winner.bound = root18Candidates38 := by
  unfold candidates
  have row : edge 6367 = (SortedNatTree.contains FastRows.row6367) := by
    change Data.row6367.contains = (SortedNatTree.contains FastRows.row6367)
    exact FastRows.row6367_checked.symm
  rw [row]
  decide
theorem root18Step38 : scan edge (search edge 3) 4 [73] 73 root18Tail38 winner =
    scan edge (search edge 3) 4 [73] 73 root18Tail39 winner := by
  rw [root18Tail38, scan, if_neg (by decide), root18Candidates38_checked]
  change scan edge (search edge 3) 4 [73] 73 root18Tail39
    (if 3 ≤ root18Candidates38.length then search edge 3 [73, 6367] root18Candidates38 6440 winner else winner) = _
  rw [if_neg (by decide)]
end ProjectEulerSolutions.PrimePairLarge.Search
