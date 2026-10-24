import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot022Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root22Candidates32 : List Nat := [6271, 8053]
set_option maxRecDepth 8192 in
theorem root22Candidates32_checked : candidates edge 4 97 5011 root22Tail33 winner.bound = root22Candidates32 := by
  unfold candidates
  have row : edge 5011 = (SortedNatTree.contains FastRows.row5011) := by
    change Data.row5011.contains = (SortedNatTree.contains FastRows.row5011)
    exact FastRows.row5011_checked.symm
  rw [row]
  decide
theorem root22Step32 : scan edge (search edge 3) 4 [97] 97 root22Tail32 winner =
    scan edge (search edge 3) 4 [97] 97 root22Tail33 winner := by
  rw [root22Tail32, scan, if_neg (by decide), root22Candidates32_checked]
  change scan edge (search edge 3) 4 [97] 97 root22Tail33
    (if 3 ≤ root22Candidates32.length then search edge 3 [97, 5011] root22Candidates32 5108 winner else winner) = _
  rw [if_neg (by decide)]
def root22Candidates33 : List Nat := [8287]
set_option maxRecDepth 8192 in
theorem root22Candidates33_checked : candidates edge 4 97 5281 root22Tail34 winner.bound = root22Candidates33 := by
  unfold candidates
  have row : edge 5281 = (SortedNatTree.contains FastRows.row5281) := by
    change Data.row5281.contains = (SortedNatTree.contains FastRows.row5281)
    exact FastRows.row5281_checked.symm
  rw [row]
  decide
theorem root22Step33 : scan edge (search edge 3) 4 [97] 97 root22Tail33 winner =
    scan edge (search edge 3) 4 [97] 97 root22Tail34 winner := by
  rw [root22Tail33, scan, if_neg (by decide), root22Candidates33_checked]
  change scan edge (search edge 3) 4 [97] 97 root22Tail34
    (if 3 ≤ root22Candidates33.length then search edge 3 [97, 5281] root22Candidates33 5378 winner else winner) = _
  rw [if_neg (by decide)]
def root22Candidates34 : List Nat := [6991, 9949, 12697]
set_option maxRecDepth 8192 in
theorem root22Candidates34_checked : candidates edge 4 97 5581 root22Tail35 winner.bound = root22Candidates34 := by
  unfold candidates
  have row : edge 5581 = (SortedNatTree.contains FastRows.row5581) := by
    change Data.row5581.contains = (SortedNatTree.contains FastRows.row5581)
    exact FastRows.row5581_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root22Child34 : search edge 3 [97, 5581] root22Candidates34 5678 winner = winner := by decide
theorem root22Step34 : scan edge (search edge 3) 4 [97] 97 root22Tail34 winner =
    scan edge (search edge 3) 4 [97] 97 root22Tail35 winner := by
  rw [root22Tail34, scan, if_neg (by decide), root22Candidates34_checked]
  change scan edge (search edge 3) 4 [97] 97 root22Tail35
    (if 3 ≤ root22Candidates34.length then search edge 3 [97, 5581] root22Candidates34 5678 winner else winner) = _
  rw [if_pos (by decide), root22Child34]
def root22Candidates35 : List Nat := [8011, 10711, 11257]
set_option maxRecDepth 8192 in
theorem root22Candidates35_checked : candidates edge 4 97 6271 root22Tail36 winner.bound = root22Candidates35 := by
  unfold candidates
  have row : edge 6271 = (SortedNatTree.contains FastRows.row6271) := by
    change Data.row6271.contains = (SortedNatTree.contains FastRows.row6271)
    exact FastRows.row6271_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root22Child35 : search edge 3 [97, 6271] root22Candidates35 6368 winner = winner := by decide
theorem root22Step35 : scan edge (search edge 3) 4 [97] 97 root22Tail35 winner =
    scan edge (search edge 3) 4 [97] 97 root22Tail36 winner := by
  rw [root22Tail35, scan, if_neg (by decide), root22Candidates35_checked]
  change scan edge (search edge 3) 4 [97] 97 root22Tail36
    (if 3 ≤ root22Candidates35.length then search edge 3 [97, 6271] root22Candidates35 6368 winner else winner) = _
  rw [if_pos (by decide), root22Child35]
def root22Candidates36 : List Nat := [9949, 10513]
set_option maxRecDepth 8192 in
theorem root22Candidates36_checked : candidates edge 4 97 6301 root22Tail37 winner.bound = root22Candidates36 := by
  unfold candidates
  have row : edge 6301 = (SortedNatTree.contains FastRows.row6301) := by
    change Data.row6301.contains = (SortedNatTree.contains FastRows.row6301)
    exact FastRows.row6301_checked.symm
  rw [row]
  decide
theorem root22Step36 : scan edge (search edge 3) 4 [97] 97 root22Tail36 winner =
    scan edge (search edge 3) 4 [97] 97 root22Tail37 winner := by
  rw [root22Tail36, scan, if_neg (by decide), root22Candidates36_checked]
  change scan edge (search edge 3) 4 [97] 97 root22Tail37
    (if 3 ≤ root22Candidates36.length then search edge 3 [97, 6301] root22Candidates36 6398 winner else winner) = _
  rw [if_neg (by decide)]
end ProjectEulerSolutions.PrimePairLarge.Search
