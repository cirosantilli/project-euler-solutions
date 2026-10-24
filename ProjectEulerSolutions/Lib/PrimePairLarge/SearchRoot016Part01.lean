import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot016Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root16Candidates32 : List Nat := [8641, 16453]
set_option maxRecDepth 8192 in
theorem root16Candidates32_checked : candidates edge 4 67 4987 root16Tail33 winner.bound = root16Candidates32 := by
  unfold candidates
  have row : edge 4987 = (SortedNatTree.contains FastRows.row4987) := by
    change Data.row4987.contains = (SortedNatTree.contains FastRows.row4987)
    exact FastRows.row4987_checked.symm
  rw [row]
  decide
theorem root16Step32 : scan edge (search edge 3) 4 [67] 67 root16Tail32 winner =
    scan edge (search edge 3) 4 [67] 67 root16Tail33 winner := by
  rw [root16Tail32, scan, if_neg (by decide), root16Candidates32_checked]
  change scan edge (search edge 3) 4 [67] 67 root16Tail33
    (if 3 ≤ root16Candidates32.length then search edge 3 [67, 4987] root16Candidates32 5054 winner else winner) = _
  rw [if_neg (by decide)]
def root16Candidates33 : List Nat := [11491, 15859]
set_option maxRecDepth 8192 in
theorem root16Candidates33_checked : candidates edge 4 67 5179 root16Tail34 winner.bound = root16Candidates33 := by
  unfold candidates
  have row : edge 5179 = (SortedNatTree.contains FastRows.row5179) := by
    change Data.row5179.contains = (SortedNatTree.contains FastRows.row5179)
    exact FastRows.row5179_checked.symm
  rw [row]
  decide
theorem root16Step33 : scan edge (search edge 3) 4 [67] 67 root16Tail33 winner =
    scan edge (search edge 3) 4 [67] 67 root16Tail34 winner := by
  rw [root16Tail33, scan, if_neg (by decide), root16Candidates33_checked]
  change scan edge (search edge 3) 4 [67] 67 root16Tail34
    (if 3 ≤ root16Candidates33.length then search edge 3 [67, 5179] root16Candidates33 5246 winner else winner) = _
  rw [if_neg (by decide)]
def root16Candidates34 : List Nat := [7687]
set_option maxRecDepth 8192 in
theorem root16Candidates34_checked : candidates edge 4 67 5419 root16Tail35 winner.bound = root16Candidates34 := by
  unfold candidates
  have row : edge 5419 = (SortedNatTree.contains FastRows.row5419) := by
    change Data.row5419.contains = (SortedNatTree.contains FastRows.row5419)
    exact FastRows.row5419_checked.symm
  rw [row]
  decide
theorem root16Step34 : scan edge (search edge 3) 4 [67] 67 root16Tail34 winner =
    scan edge (search edge 3) 4 [67] 67 root16Tail35 winner := by
  rw [root16Tail34, scan, if_neg (by decide), root16Candidates34_checked]
  change scan edge (search edge 3) 4 [67] 67 root16Tail35
    (if 3 ≤ root16Candidates34.length then search edge 3 [67, 5419] root16Candidates34 5486 winner else winner) = _
  rw [if_neg (by decide)]
def root16Candidates35 : List Nat := [5923, 7687, 9433, 11119]
set_option maxRecDepth 8192 in
theorem root16Candidates35_checked : candidates edge 4 67 5569 root16Tail36 winner.bound = root16Candidates35 := by
  unfold candidates
  have row : edge 5569 = (SortedNatTree.contains FastRows.row5569) := by
    change Data.row5569.contains = (SortedNatTree.contains FastRows.row5569)
    exact FastRows.row5569_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root16Child35 : search edge 3 [67, 5569] root16Candidates35 5636 winner = winner := by decide
theorem root16Step35 : scan edge (search edge 3) 4 [67] 67 root16Tail35 winner =
    scan edge (search edge 3) 4 [67] 67 root16Tail36 winner := by
  rw [root16Tail35, scan, if_neg (by decide), root16Candidates35_checked]
  change scan edge (search edge 3) 4 [67] 67 root16Tail36
    (if 3 ≤ root16Candidates35.length then search edge 3 [67, 5569] root16Candidates35 5636 winner else winner) = _
  rw [if_pos (by decide), root16Child35]
def root16Candidates36 : List Nat := [7561, 14107]
set_option maxRecDepth 8192 in
theorem root16Candidates36_checked : candidates edge 4 67 5581 root16Tail37 winner.bound = root16Candidates36 := by
  unfold candidates
  have row : edge 5581 = (SortedNatTree.contains FastRows.row5581) := by
    change Data.row5581.contains = (SortedNatTree.contains FastRows.row5581)
    exact FastRows.row5581_checked.symm
  rw [row]
  decide
theorem root16Step36 : scan edge (search edge 3) 4 [67] 67 root16Tail36 winner =
    scan edge (search edge 3) 4 [67] 67 root16Tail37 winner := by
  rw [root16Tail36, scan, if_neg (by decide), root16Candidates36_checked]
  change scan edge (search edge 3) 4 [67] 67 root16Tail37
    (if 3 ≤ root16Candidates36.length then search edge 3 [67, 5581] root16Candidates36 5648 winner else winner) = _
  rw [if_neg (by decide)]
def root16Candidates37 : List Nat := [9883, 10159, 10909, 14107]
set_option maxRecDepth 8192 in
theorem root16Candidates37_checked : candidates edge 4 67 5743 root16Tail38 winner.bound = root16Candidates37 := by
  unfold candidates
  have row : edge 5743 = (SortedNatTree.contains FastRows.row5743) := by
    change Data.row5743.contains = (SortedNatTree.contains FastRows.row5743)
    exact FastRows.row5743_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root16Child37 : search edge 3 [67, 5743] root16Candidates37 5810 winner = winner := by decide
theorem root16Step37 : scan edge (search edge 3) 4 [67] 67 root16Tail37 winner =
    scan edge (search edge 3) 4 [67] 67 root16Tail38 winner := by
  rw [root16Tail37, scan, if_neg (by decide), root16Candidates37_checked]
  change scan edge (search edge 3) 4 [67] 67 root16Tail38
    (if 3 ≤ root16Candidates37.length then search edge 3 [67, 5743] root16Candidates37 5810 winner else winner) = _
  rw [if_pos (by decide), root16Child37]
def root16Candidates38 : List Nat := [16363]
set_option maxRecDepth 8192 in
theorem root16Candidates38_checked : candidates edge 4 67 5827 root16Tail39 winner.bound = root16Candidates38 := by
  unfold candidates
  have row : edge 5827 = (SortedNatTree.contains FastRows.row5827) := by
    change Data.row5827.contains = (SortedNatTree.contains FastRows.row5827)
    exact FastRows.row5827_checked.symm
  rw [row]
  decide
theorem root16Step38 : scan edge (search edge 3) 4 [67] 67 root16Tail38 winner =
    scan edge (search edge 3) 4 [67] 67 root16Tail39 winner := by
  rw [root16Tail38, scan, if_neg (by decide), root16Candidates38_checked]
  change scan edge (search edge 3) 4 [67] 67 root16Tail39
    (if 3 ≤ root16Candidates38.length then search edge 3 [67, 5827] root16Candidates38 5894 winner else winner) = _
  rw [if_neg (by decide)]
def root16Candidates39 : List Nat := [16453]
set_option maxRecDepth 8192 in
theorem root16Candidates39_checked : candidates edge 4 67 5881 root16Tail40 winner.bound = root16Candidates39 := by
  unfold candidates
  have row : edge 5881 = (SortedNatTree.contains FastRows.row5881) := by
    change Data.row5881.contains = (SortedNatTree.contains FastRows.row5881)
    exact FastRows.row5881_checked.symm
  rw [row]
  decide
theorem root16Step39 : scan edge (search edge 3) 4 [67] 67 root16Tail39 winner =
    scan edge (search edge 3) 4 [67] 67 root16Tail40 winner := by
  rw [root16Tail39, scan, if_neg (by decide), root16Candidates39_checked]
  change scan edge (search edge 3) 4 [67] 67 root16Tail40
    (if 3 ≤ root16Candidates39.length then search edge 3 [67, 5881] root16Candidates39 5948 winner else winner) = _
  rw [if_neg (by decide)]
def root16Candidates40 : List Nat := [8581]
set_option maxRecDepth 8192 in
theorem root16Candidates40_checked : candidates edge 4 67 5923 root16Tail41 winner.bound = root16Candidates40 := by
  unfold candidates
  have row : edge 5923 = (SortedNatTree.contains FastRows.row5923) := by
    change Data.row5923.contains = (SortedNatTree.contains FastRows.row5923)
    exact FastRows.row5923_checked.symm
  rw [row]
  decide
theorem root16Step40 : scan edge (search edge 3) 4 [67] 67 root16Tail40 winner =
    scan edge (search edge 3) 4 [67] 67 root16Tail41 winner := by
  rw [root16Tail40, scan, if_neg (by decide), root16Candidates40_checked]
  change scan edge (search edge 3) 4 [67] 67 root16Tail41
    (if 3 ≤ root16Candidates40.length then search edge 3 [67, 5923] root16Candidates40 5990 winner else winner) = _
  rw [if_neg (by decide)]
def root16Candidates41 : List Nat := [9907, 10159]
set_option maxRecDepth 8192 in
theorem root16Candidates41_checked : candidates edge 4 67 6337 root16Tail42 winner.bound = root16Candidates41 := by
  unfold candidates
  have row : edge 6337 = (SortedNatTree.contains FastRows.row6337) := by
    change Data.row6337.contains = (SortedNatTree.contains FastRows.row6337)
    exact FastRows.row6337_checked.symm
  rw [row]
  decide
theorem root16Step41 : scan edge (search edge 3) 4 [67] 67 root16Tail41 winner =
    scan edge (search edge 3) 4 [67] 67 root16Tail42 winner := by
  rw [root16Tail41, scan, if_neg (by decide), root16Candidates41_checked]
  change scan edge (search edge 3) 4 [67] 67 root16Tail42
    (if 3 ≤ root16Candidates41.length then search edge 3 [67, 6337] root16Candidates41 6404 winner else winner) = _
  rw [if_neg (by decide)]
end ProjectEulerSolutions.PrimePairLarge.Search
