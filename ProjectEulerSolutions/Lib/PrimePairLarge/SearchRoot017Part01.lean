import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot017Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root17Candidates32 : List Nat := [5849, 7589, 11087]
set_option maxRecDepth 8192 in
theorem root17Candidates32_checked : candidates edge 4 71 5153 root17Tail33 winner.bound = root17Candidates32 := by
  unfold candidates
  have row : edge 5153 = (SortedNatTree.contains FastRows.row5153) := by
    change Data.row5153.contains = (SortedNatTree.contains FastRows.row5153)
    exact FastRows.row5153_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root17Child32 : search edge 3 [71, 5153] root17Candidates32 5224 winner = winner := by decide
theorem root17Step32 : scan edge (search edge 3) 4 [71] 71 root17Tail32 winner =
    scan edge (search edge 3) 4 [71] 71 root17Tail33 winner := by
  rw [root17Tail32, scan, if_neg (by decide), root17Candidates32_checked]
  change scan edge (search edge 3) 4 [71] 71 root17Tail33
    (if 3 ≤ root17Candidates32.length then search edge 3 [71, 5153] root17Candidates32 5224 winner else winner) = _
  rw [if_pos (by decide), root17Child32]
def root17Candidates33 : List Nat := [6563, 7151]
set_option maxRecDepth 8192 in
theorem root17Candidates33_checked : candidates edge 4 71 5237 root17Tail34 winner.bound = root17Candidates33 := by
  unfold candidates
  have row : edge 5237 = (SortedNatTree.contains FastRows.row5237) := by
    change Data.row5237.contains = (SortedNatTree.contains FastRows.row5237)
    exact FastRows.row5237_checked.symm
  rw [row]
  decide
theorem root17Step33 : scan edge (search edge 3) 4 [71] 71 root17Tail33 winner =
    scan edge (search edge 3) 4 [71] 71 root17Tail34 winner := by
  rw [root17Tail33, scan, if_neg (by decide), root17Candidates33_checked]
  change scan edge (search edge 3) 4 [71] 71 root17Tail34
    (if 3 ≤ root17Candidates33.length then search edge 3 [71, 5237] root17Candidates33 5308 winner else winner) = _
  rw [if_neg (by decide)]
def root17Candidates34 : List Nat := [7001, 9239, 9689, 17021]
set_option maxRecDepth 8192 in
theorem root17Candidates34_checked : candidates edge 4 71 5417 root17Tail35 winner.bound = root17Candidates34 := by
  unfold candidates
  have row : edge 5417 = (SortedNatTree.contains FastRows.row5417) := by
    change Data.row5417.contains = (SortedNatTree.contains FastRows.row5417)
    exact FastRows.row5417_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root17Child34 : search edge 3 [71, 5417] root17Candidates34 5488 winner = winner := by decide
theorem root17Step34 : scan edge (search edge 3) 4 [71] 71 root17Tail34 winner =
    scan edge (search edge 3) 4 [71] 71 root17Tail35 winner := by
  rw [root17Tail34, scan, if_neg (by decide), root17Candidates34_checked]
  change scan edge (search edge 3) 4 [71] 71 root17Tail35
    (if 3 ≤ root17Candidates34.length then search edge 3 [71, 5417] root17Candidates34 5488 winner else winner) = _
  rw [if_pos (by decide), root17Child34]
def root17Candidates35 : List Nat := [5849]
set_option maxRecDepth 8192 in
theorem root17Candidates35_checked : candidates edge 4 71 5441 root17Tail36 winner.bound = root17Candidates35 := by
  unfold candidates
  have row : edge 5441 = (SortedNatTree.contains FastRows.row5441) := by
    change Data.row5441.contains = (SortedNatTree.contains FastRows.row5441)
    exact FastRows.row5441_checked.symm
  rw [row]
  decide
theorem root17Step35 : scan edge (search edge 3) 4 [71] 71 root17Tail35 winner =
    scan edge (search edge 3) 4 [71] 71 root17Tail36 winner := by
  rw [root17Tail35, scan, if_neg (by decide), root17Candidates35_checked]
  change scan edge (search edge 3) 4 [71] 71 root17Tail36
    (if 3 ≤ root17Candidates35.length then search edge 3 [71, 5441] root17Candidates35 5512 winner else winner) = _
  rw [if_neg (by decide)]
def root17Candidates36 : List Nat := [6563]
set_option maxRecDepth 8192 in
theorem root17Candidates36_checked : candidates edge 4 71 5639 root17Tail37 winner.bound = root17Candidates36 := by
  unfold candidates
  have row : edge 5639 = (SortedNatTree.contains FastRows.row5639) := by
    change Data.row5639.contains = (SortedNatTree.contains FastRows.row5639)
    exact FastRows.row5639_checked.symm
  rw [row]
  decide
theorem root17Step36 : scan edge (search edge 3) 4 [71] 71 root17Tail36 winner =
    scan edge (search edge 3) 4 [71] 71 root17Tail37 winner := by
  rw [root17Tail36, scan, if_neg (by decide), root17Candidates36_checked]
  change scan edge (search edge 3) 4 [71] 71 root17Tail37
    (if 3 ≤ root17Candidates36.length then search edge 3 [71, 5639] root17Candidates36 5710 winner else winner) = _
  rw [if_neg (by decide)]
def root17Candidates37 : List Nat := [13553, 20123]
set_option maxRecDepth 8192 in
theorem root17Candidates37_checked : candidates edge 4 71 5651 root17Tail38 winner.bound = root17Candidates37 := by
  unfold candidates
  have row : edge 5651 = (SortedNatTree.contains FastRows.row5651) := by
    change Data.row5651.contains = (SortedNatTree.contains FastRows.row5651)
    exact FastRows.row5651_checked.symm
  rw [row]
  decide
theorem root17Step37 : scan edge (search edge 3) 4 [71] 71 root17Tail37 winner =
    scan edge (search edge 3) 4 [71] 71 root17Tail38 winner := by
  rw [root17Tail37, scan, if_neg (by decide), root17Candidates37_checked]
  change scan edge (search edge 3) 4 [71] 71 root17Tail38
    (if 3 ≤ root17Candidates37.length then search edge 3 [71, 5651] root17Candidates37 5722 winner else winner) = _
  rw [if_neg (by decide)]
def root17Candidates38 : List Nat := [8387]
set_option maxRecDepth 8192 in
theorem root17Candidates38_checked : candidates edge 4 71 5657 root17Tail39 winner.bound = root17Candidates38 := by
  unfold candidates
  have row : edge 5657 = (SortedNatTree.contains FastRows.row5657) := by
    change Data.row5657.contains = (SortedNatTree.contains FastRows.row5657)
    exact FastRows.row5657_checked.symm
  rw [row]
  decide
theorem root17Step38 : scan edge (search edge 3) 4 [71] 71 root17Tail38 winner =
    scan edge (search edge 3) 4 [71] 71 root17Tail39 winner := by
  rw [root17Tail38, scan, if_neg (by decide), root17Candidates38_checked]
  change scan edge (search edge 3) 4 [71] 71 root17Tail39
    (if 3 ≤ root17Candidates38.length then search edge 3 [71, 5657] root17Candidates38 5728 winner else winner) = _
  rw [if_neg (by decide)]
def root17Candidates39 : List Nat := []
set_option maxRecDepth 8192 in
theorem root17Candidates39_checked : candidates edge 4 71 5849 root17Tail40 winner.bound = root17Candidates39 := by
  unfold candidates
  have row : edge 5849 = (SortedNatTree.contains FastRows.row5849) := by
    change Data.row5849.contains = (SortedNatTree.contains FastRows.row5849)
    exact FastRows.row5849_checked.symm
  rw [row]
  decide
theorem root17Step39 : scan edge (search edge 3) 4 [71] 71 root17Tail39 winner =
    scan edge (search edge 3) 4 [71] 71 root17Tail40 winner := by
  rw [root17Tail39, scan, if_neg (by decide), root17Candidates39_checked]
  change scan edge (search edge 3) 4 [71] 71 root17Tail40
    (if 3 ≤ root17Candidates39.length then search edge 3 [71, 5849] root17Candidates39 5920 winner else winner) = _
  rw [if_neg (by decide)]
def root17Candidates40 : List Nat := [17021]
set_option maxRecDepth 8192 in
theorem root17Candidates40_checked : candidates edge 4 71 5879 root17Tail41 winner.bound = root17Candidates40 := by
  unfold candidates
  have row : edge 5879 = (SortedNatTree.contains FastRows.row5879) := by
    change Data.row5879.contains = (SortedNatTree.contains FastRows.row5879)
    exact FastRows.row5879_checked.symm
  rw [row]
  decide
theorem root17Step40 : scan edge (search edge 3) 4 [71] 71 root17Tail40 winner =
    scan edge (search edge 3) 4 [71] 71 root17Tail41 winner := by
  rw [root17Tail40, scan, if_neg (by decide), root17Candidates40_checked]
  change scan edge (search edge 3) 4 [71] 71 root17Tail41
    (if 3 ≤ root17Candidates40.length then search edge 3 [71, 5879] root17Candidates40 5950 winner else winner) = _
  rw [if_neg (by decide)]
def root17Candidates41 : List Nat := [8423]
set_option maxRecDepth 8192 in
theorem root17Candidates41_checked : candidates edge 4 71 6389 root17Tail42 winner.bound = root17Candidates41 := by
  unfold candidates
  have row : edge 6389 = (SortedNatTree.contains FastRows.row6389) := by
    change Data.row6389.contains = (SortedNatTree.contains FastRows.row6389)
    exact FastRows.row6389_checked.symm
  rw [row]
  decide
theorem root17Step41 : scan edge (search edge 3) 4 [71] 71 root17Tail41 winner =
    scan edge (search edge 3) 4 [71] 71 root17Tail42 winner := by
  rw [root17Tail41, scan, if_neg (by decide), root17Candidates41_checked]
  change scan edge (search edge 3) 4 [71] 71 root17Tail42
    (if 3 ≤ root17Candidates41.length then search edge 3 [71, 6389] root17Candidates41 6460 winner else winner) = _
  rw [if_neg (by decide)]
end ProjectEulerSolutions.PrimePairLarge.Search
