import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot010Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root10Candidates32 : List Nat := [7457]
set_option maxRecDepth 8192 in
theorem root10Candidates32_checked : candidates edge 4 41 4691 root10Tail33 winner.bound = root10Candidates32 := by
  unfold candidates
  have row : edge 4691 = (SortedNatTree.contains FastRows.row4691) := by
    change Data.row4691.contains = (SortedNatTree.contains FastRows.row4691)
    exact FastRows.row4691_checked.symm
  rw [row]
  decide
theorem root10Step32 : scan edge (search edge 3) 4 [41] 41 root10Tail32 winner =
    scan edge (search edge 3) 4 [41] 41 root10Tail33 winner := by
  rw [root10Tail32, scan, if_neg (by decide), root10Candidates32_checked]
  change scan edge (search edge 3) 4 [41] 41 root10Tail33
    (if 3 ≤ root10Candidates32.length then search edge 3 [41, 4691] root10Candidates32 4732 winner else winner) = _
  rw [if_neg (by decide)]
def root10Candidates33 : List Nat := [7331, 13259, 15101]
set_option maxRecDepth 8192 in
theorem root10Candidates33_checked : candidates edge 4 41 5147 root10Tail34 winner.bound = root10Candidates33 := by
  unfold candidates
  have row : edge 5147 = (SortedNatTree.contains FastRows.row5147) := by
    change Data.row5147.contains = (SortedNatTree.contains FastRows.row5147)
    exact FastRows.row5147_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root10Child33 : search edge 3 [41, 5147] root10Candidates33 5188 winner = winner := by decide
theorem root10Step33 : scan edge (search edge 3) 4 [41] 41 root10Tail33 winner =
    scan edge (search edge 3) 4 [41] 41 root10Tail34 winner := by
  rw [root10Tail33, scan, if_neg (by decide), root10Candidates33_checked]
  change scan edge (search edge 3) 4 [41] 41 root10Tail34
    (if 3 ≤ root10Candidates33.length then search edge 3 [41, 5147] root10Candidates33 5188 winner else winner) = _
  rw [if_pos (by decide), root10Child33]
def root10Candidates34 : List Nat := [6089, 14009, 15263]
set_option maxRecDepth 8192 in
theorem root10Candidates34_checked : candidates edge 4 41 5477 root10Tail35 winner.bound = root10Candidates34 := by
  unfold candidates
  have row : edge 5477 = (SortedNatTree.contains FastRows.row5477) := by
    change Data.row5477.contains = (SortedNatTree.contains FastRows.row5477)
    exact FastRows.row5477_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root10Child34 : search edge 3 [41, 5477] root10Candidates34 5518 winner = winner := by decide
theorem root10Step34 : scan edge (search edge 3) 4 [41] 41 root10Tail34 winner =
    scan edge (search edge 3) 4 [41] 41 root10Tail35 winner := by
  rw [root10Tail34, scan, if_neg (by decide), root10Candidates34_checked]
  change scan edge (search edge 3) 4 [41] 41 root10Tail35
    (if 3 ≤ root10Candidates34.length then search edge 3 [41, 5477] root10Candidates34 5518 winner else winner) = _
  rw [if_pos (by decide), root10Child34]
def root10Candidates35 : List Nat := []
set_option maxRecDepth 8192 in
theorem root10Candidates35_checked : candidates edge 4 41 5717 root10Tail36 winner.bound = root10Candidates35 := by
  unfold candidates
  have row : edge 5717 = (SortedNatTree.contains FastRows.row5717) := by
    change Data.row5717.contains = (SortedNatTree.contains FastRows.row5717)
    exact FastRows.row5717_checked.symm
  rw [row]
  decide
theorem root10Step35 : scan edge (search edge 3) 4 [41] 41 root10Tail35 winner =
    scan edge (search edge 3) 4 [41] 41 root10Tail36 winner := by
  rw [root10Tail35, scan, if_neg (by decide), root10Candidates35_checked]
  change scan edge (search edge 3) 4 [41] 41 root10Tail36
    (if 3 ≤ root10Candidates35.length then search edge 3 [41, 5717] root10Candidates35 5758 winner else winner) = _
  rw [if_neg (by decide)]
def root10Candidates36 : List Nat := [9467, 17291]
set_option maxRecDepth 8192 in
theorem root10Candidates36_checked : candidates edge 4 41 6089 root10Tail37 winner.bound = root10Candidates36 := by
  unfold candidates
  have row : edge 6089 = (SortedNatTree.contains FastRows.row6089) := by
    change Data.row6089.contains = (SortedNatTree.contains FastRows.row6089)
    exact FastRows.row6089_checked.symm
  rw [row]
  decide
theorem root10Step36 : scan edge (search edge 3) 4 [41] 41 root10Tail36 winner =
    scan edge (search edge 3) 4 [41] 41 root10Tail37 winner := by
  rw [root10Tail36, scan, if_neg (by decide), root10Candidates36_checked]
  change scan edge (search edge 3) 4 [41] 41 root10Tail37
    (if 3 ≤ root10Candidates36.length then search edge 3 [41, 6089] root10Candidates36 6130 winner else winner) = _
  rw [if_neg (by decide)]
def root10Candidates37 : List Nat := [11489, 12497]
set_option maxRecDepth 8192 in
theorem root10Candidates37_checked : candidates edge 4 41 6263 root10Tail38 winner.bound = root10Candidates37 := by
  unfold candidates
  have row : edge 6263 = (SortedNatTree.contains FastRows.row6263) := by
    change Data.row6263.contains = (SortedNatTree.contains FastRows.row6263)
    exact FastRows.row6263_checked.symm
  rw [row]
  decide
theorem root10Step37 : scan edge (search edge 3) 4 [41] 41 root10Tail37 winner =
    scan edge (search edge 3) 4 [41] 41 root10Tail38 winner := by
  rw [root10Tail37, scan, if_neg (by decide), root10Candidates37_checked]
  change scan edge (search edge 3) 4 [41] 41 root10Tail38
    (if 3 ≤ root10Candidates37.length then search edge 3 [41, 6263] root10Candidates37 6304 winner else winner) = _
  rw [if_neg (by decide)]
def root10Candidates38 : List Nat := [7643]
set_option maxRecDepth 8192 in
theorem root10Candidates38_checked : candidates edge 4 41 6473 root10Tail39 winner.bound = root10Candidates38 := by
  unfold candidates
  have row : edge 6473 = (SortedNatTree.contains FastRows.row6473) := by
    change Data.row6473.contains = (SortedNatTree.contains FastRows.row6473)
    exact FastRows.row6473_checked.symm
  rw [row]
  decide
theorem root10Step38 : scan edge (search edge 3) 4 [41] 41 root10Tail38 winner =
    scan edge (search edge 3) 4 [41] 41 root10Tail39 winner := by
  rw [root10Tail38, scan, if_neg (by decide), root10Candidates38_checked]
  change scan edge (search edge 3) 4 [41] 41 root10Tail39
    (if 3 ≤ root10Candidates38.length then search edge 3 [41, 6473] root10Candidates38 6514 winner else winner) = _
  rw [if_neg (by decide)]
def root10Candidates39 : List Nat := [7583, 9161, 9473, 13259]
set_option maxRecDepth 8192 in
theorem root10Candidates39_checked : candidates edge 4 41 6491 root10Tail40 winner.bound = root10Candidates39 := by
  unfold candidates
  have row : edge 6491 = (SortedNatTree.contains FastRows.row6491) := by
    change Data.row6491.contains = (SortedNatTree.contains FastRows.row6491)
    exact FastRows.row6491_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root10Child39 : search edge 3 [41, 6491] root10Candidates39 6532 winner = winner := by decide
theorem root10Step39 : scan edge (search edge 3) 4 [41] 41 root10Tail39 winner =
    scan edge (search edge 3) 4 [41] 41 root10Tail40 winner := by
  rw [root10Tail39, scan, if_neg (by decide), root10Candidates39_checked]
  change scan edge (search edge 3) 4 [41] 41 root10Tail40
    (if 3 ≤ root10Candidates39.length then search edge 3 [41, 6491] root10Candidates39 6532 winner else winner) = _
  rw [if_pos (by decide), root10Child39]
end ProjectEulerSolutions.PrimePairLarge.Search
