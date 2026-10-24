import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot014Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root14Candidates32 : List Nat := [13763]
set_option maxRecDepth 8192 in
theorem root14Candidates32_checked : candidates edge 4 59 4751 root14Tail33 winner.bound = root14Candidates32 := by
  unfold candidates
  have row : edge 4751 = (SortedNatTree.contains FastRows.row4751) := by
    change Data.row4751.contains = (SortedNatTree.contains FastRows.row4751)
    exact FastRows.row4751_checked.symm
  rw [row]
  decide
theorem root14Step32 : scan edge (search edge 3) 4 [59] 59 root14Tail32 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail33 winner := by
  rw [root14Tail32, scan, if_neg (by decide), root14Candidates32_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail33
    (if 3 ≤ root14Candidates32.length then search edge 3 [59, 4751] root14Candidates32 4810 winner else winner) = _
  rw [if_neg (by decide)]
def root14Candidates33 : List Nat := [8123]
set_option maxRecDepth 8192 in
theorem root14Candidates33_checked : candidates edge 4 59 4889 root14Tail34 winner.bound = root14Candidates33 := by
  unfold candidates
  have row : edge 4889 = (SortedNatTree.contains FastRows.row4889) := by
    change Data.row4889.contains = (SortedNatTree.contains FastRows.row4889)
    exact FastRows.row4889_checked.symm
  rw [row]
  decide
theorem root14Step33 : scan edge (search edge 3) 4 [59] 59 root14Tail33 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail34 winner := by
  rw [root14Tail33, scan, if_neg (by decide), root14Candidates33_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail34
    (if 3 ≤ root14Candidates33.length then search edge 3 [59, 4889] root14Candidates33 4948 winner else winner) = _
  rw [if_neg (by decide)]
def root14Candidates34 : List Nat := [5843, 7349, 10973, 17903, 19031]
set_option maxRecDepth 8192 in
theorem root14Candidates34_checked : candidates edge 4 59 4931 root14Tail35 winner.bound = root14Candidates34 := by
  unfold candidates
  have row : edge 4931 = (SortedNatTree.contains FastRows.row4931) := by
    change Data.row4931.contains = (SortedNatTree.contains FastRows.row4931)
    exact FastRows.row4931_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root14Child34 : search edge 3 [59, 4931] root14Candidates34 4990 winner = winner := by decide
theorem root14Step34 : scan edge (search edge 3) 4 [59] 59 root14Tail34 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail35 winner := by
  rw [root14Tail34, scan, if_neg (by decide), root14Candidates34_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail35
    (if 3 ≤ root14Candidates34.length then search edge 3 [59, 4931] root14Candidates34 4990 winner else winner) = _
  rw [if_pos (by decide), root14Child34]
def root14Candidates35 : List Nat := [8093, 8867]
set_option maxRecDepth 8192 in
theorem root14Candidates35_checked : candidates edge 4 59 5039 root14Tail36 winner.bound = root14Candidates35 := by
  unfold candidates
  have row : edge 5039 = (SortedNatTree.contains FastRows.row5039) := by
    change Data.row5039.contains = (SortedNatTree.contains FastRows.row5039)
    exact FastRows.row5039_checked.symm
  rw [row]
  decide
theorem root14Step35 : scan edge (search edge 3) 4 [59] 59 root14Tail35 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail36 winner := by
  rw [root14Tail35, scan, if_neg (by decide), root14Candidates35_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail36
    (if 3 ≤ root14Candidates35.length then search edge 3 [59, 5039] root14Candidates35 5098 winner else winner) = _
  rw [if_neg (by decide)]
def root14Candidates36 : List Nat := [7253, 15809]
set_option maxRecDepth 8192 in
theorem root14Candidates36_checked : candidates edge 4 59 5081 root14Tail37 winner.bound = root14Candidates36 := by
  unfold candidates
  have row : edge 5081 = (SortedNatTree.contains FastRows.row5081) := by
    change Data.row5081.contains = (SortedNatTree.contains FastRows.row5081)
    exact FastRows.row5081_checked.symm
  rw [row]
  decide
theorem root14Step36 : scan edge (search edge 3) 4 [59] 59 root14Tail36 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail37 winner := by
  rw [root14Tail36, scan, if_neg (by decide), root14Candidates36_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail37
    (if 3 ≤ root14Candidates36.length then search edge 3 [59, 5081] root14Candidates36 5140 winner else winner) = _
  rw [if_neg (by decide)]
def root14Candidates37 : List Nat := [6779, 6983, 9629, 10973]
set_option maxRecDepth 8192 in
theorem root14Candidates37_checked : candidates edge 4 59 5261 root14Tail38 winner.bound = root14Candidates37 := by
  unfold candidates
  have row : edge 5261 = (SortedNatTree.contains FastRows.row5261) := by
    change Data.row5261.contains = (SortedNatTree.contains FastRows.row5261)
    exact FastRows.row5261_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root14Child37 : search edge 3 [59, 5261] root14Candidates37 5320 winner = winner := by decide
theorem root14Step37 : scan edge (search edge 3) 4 [59] 59 root14Tail37 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail38 winner := by
  rw [root14Tail37, scan, if_neg (by decide), root14Candidates37_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail38
    (if 3 ≤ root14Candidates37.length then search edge 3 [59, 5261] root14Candidates37 5320 winner else winner) = _
  rw [if_pos (by decide), root14Child37]
def root14Candidates38 : List Nat := [8123, 9341, 9719]
set_option maxRecDepth 8192 in
theorem root14Candidates38_checked : candidates edge 4 59 5303 root14Tail39 winner.bound = root14Candidates38 := by
  unfold candidates
  have row : edge 5303 = (SortedNatTree.contains FastRows.row5303) := by
    change Data.row5303.contains = (SortedNatTree.contains FastRows.row5303)
    exact FastRows.row5303_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root14Child38 : search edge 3 [59, 5303] root14Candidates38 5362 winner = winner := by decide
theorem root14Step38 : scan edge (search edge 3) 4 [59] 59 root14Tail38 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail39 winner := by
  rw [root14Tail38, scan, if_neg (by decide), root14Candidates38_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail39
    (if 3 ≤ root14Candidates38.length then search edge 3 [59, 5303] root14Candidates38 5362 winner else winner) = _
  rw [if_pos (by decide), root14Child38]
def root14Candidates39 : List Nat := [5519, 5927, 6899, 8537]
set_option maxRecDepth 8192 in
theorem root14Candidates39_checked : candidates edge 4 59 5351 root14Tail40 winner.bound = root14Candidates39 := by
  unfold candidates
  have row : edge 5351 = (SortedNatTree.contains FastRows.row5351) := by
    change Data.row5351.contains = (SortedNatTree.contains FastRows.row5351)
    exact FastRows.row5351_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root14Child39 : search edge 3 [59, 5351] root14Candidates39 5410 winner = winner := by decide
theorem root14Step39 : scan edge (search edge 3) 4 [59] 59 root14Tail39 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail40 winner := by
  rw [root14Tail39, scan, if_neg (by decide), root14Candidates39_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail40
    (if 3 ≤ root14Candidates39.length then search edge 3 [59, 5351] root14Candidates39 5410 winner else winner) = _
  rw [if_pos (by decide), root14Child39]
def root14Candidates40 : List Nat := [5981, 13931]
set_option maxRecDepth 8192 in
theorem root14Candidates40_checked : candidates edge 4 59 5381 root14Tail41 winner.bound = root14Candidates40 := by
  unfold candidates
  have row : edge 5381 = (SortedNatTree.contains FastRows.row5381) := by
    change Data.row5381.contains = (SortedNatTree.contains FastRows.row5381)
    exact FastRows.row5381_checked.symm
  rw [row]
  decide
theorem root14Step40 : scan edge (search edge 3) 4 [59] 59 root14Tail40 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail41 winner := by
  rw [root14Tail40, scan, if_neg (by decide), root14Candidates40_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail41
    (if 3 ≤ root14Candidates40.length then search edge 3 [59, 5381] root14Candidates40 5440 winner else winner) = _
  rw [if_neg (by decide)]
def root14Candidates41 : List Nat := [15809]
set_option maxRecDepth 8192 in
theorem root14Candidates41_checked : candidates edge 4 59 5519 root14Tail42 winner.bound = root14Candidates41 := by
  unfold candidates
  have row : edge 5519 = (SortedNatTree.contains FastRows.row5519) := by
    change Data.row5519.contains = (SortedNatTree.contains FastRows.row5519)
    exact FastRows.row5519_checked.symm
  rw [row]
  decide
theorem root14Step41 : scan edge (search edge 3) 4 [59] 59 root14Tail41 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail42 winner := by
  rw [root14Tail41, scan, if_neg (by decide), root14Candidates41_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail42
    (if 3 ≤ root14Candidates41.length then search edge 3 [59, 5519] root14Candidates41 5578 winner else winner) = _
  rw [if_neg (by decide)]
def root14Candidates42 : List Nat := []
set_option maxRecDepth 8192 in
theorem root14Candidates42_checked : candidates edge 4 59 5717 root14Tail43 winner.bound = root14Candidates42 := by
  unfold candidates
  have row : edge 5717 = (SortedNatTree.contains FastRows.row5717) := by
    change Data.row5717.contains = (SortedNatTree.contains FastRows.row5717)
    exact FastRows.row5717_checked.symm
  rw [row]
  decide
theorem root14Step42 : scan edge (search edge 3) 4 [59] 59 root14Tail42 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail43 winner := by
  rw [root14Tail42, scan, if_neg (by decide), root14Candidates42_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail43
    (if 3 ≤ root14Candidates42.length then search edge 3 [59, 5717] root14Candidates42 5776 winner else winner) = _
  rw [if_neg (by decide)]
def root14Candidates43 : List Nat := [8093, 13931, 19289]
set_option maxRecDepth 8192 in
theorem root14Candidates43_checked : candidates edge 4 59 5741 root14Tail44 winner.bound = root14Candidates43 := by
  unfold candidates
  have row : edge 5741 = (SortedNatTree.contains FastRows.row5741) := by
    change Data.row5741.contains = (SortedNatTree.contains FastRows.row5741)
    exact FastRows.row5741_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root14Child43 : search edge 3 [59, 5741] root14Candidates43 5800 winner = winner := by decide
theorem root14Step43 : scan edge (search edge 3) 4 [59] 59 root14Tail43 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail44 winner := by
  rw [root14Tail43, scan, if_neg (by decide), root14Candidates43_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail44
    (if 3 ≤ root14Candidates43.length then search edge 3 [59, 5741] root14Candidates43 5800 winner else winner) = _
  rw [if_pos (by decide), root14Child43]
def root14Candidates44 : List Nat := []
set_option maxRecDepth 8192 in
theorem root14Candidates44_checked : candidates edge 4 59 5807 root14Tail45 winner.bound = root14Candidates44 := by
  unfold candidates
  have row : edge 5807 = (SortedNatTree.contains FastRows.row5807) := by
    change Data.row5807.contains = (SortedNatTree.contains FastRows.row5807)
    exact FastRows.row5807_checked.symm
  rw [row]
  decide
theorem root14Step44 : scan edge (search edge 3) 4 [59] 59 root14Tail44 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail45 winner := by
  rw [root14Tail44, scan, if_neg (by decide), root14Candidates44_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail45
    (if 3 ≤ root14Candidates44.length then search edge 3 [59, 5807] root14Candidates44 5866 winner else winner) = _
  rw [if_neg (by decide)]
def root14Candidates45 : List Nat := [6977, 19157]
set_option maxRecDepth 8192 in
theorem root14Candidates45_checked : candidates edge 4 59 5843 root14Tail46 winner.bound = root14Candidates45 := by
  unfold candidates
  have row : edge 5843 = (SortedNatTree.contains FastRows.row5843) := by
    change Data.row5843.contains = (SortedNatTree.contains FastRows.row5843)
    exact FastRows.row5843_checked.symm
  rw [row]
  decide
theorem root14Step45 : scan edge (search edge 3) 4 [59] 59 root14Tail45 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail46 winner := by
  rw [root14Tail45, scan, if_neg (by decide), root14Candidates45_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail46
    (if 3 ≤ root14Candidates45.length then search edge 3 [59, 5843] root14Candidates45 5902 winner else winner) = _
  rw [if_neg (by decide)]
def root14Candidates46 : List Nat := [13907, 19157]
set_option maxRecDepth 8192 in
theorem root14Candidates46_checked : candidates edge 4 59 5927 root14Tail47 winner.bound = root14Candidates46 := by
  unfold candidates
  have row : edge 5927 = (SortedNatTree.contains FastRows.row5927) := by
    change Data.row5927.contains = (SortedNatTree.contains FastRows.row5927)
    exact FastRows.row5927_checked.symm
  rw [row]
  decide
theorem root14Step46 : scan edge (search edge 3) 4 [59] 59 root14Tail46 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail47 winner := by
  rw [root14Tail46, scan, if_neg (by decide), root14Candidates46_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail47
    (if 3 ≤ root14Candidates46.length then search edge 3 [59, 5927] root14Candidates46 5986 winner else winner) = _
  rw [if_neg (by decide)]
def root14Candidates47 : List Nat := [9341, 17903]
set_option maxRecDepth 8192 in
theorem root14Candidates47_checked : candidates edge 4 59 5981 root14Tail48 winner.bound = root14Candidates47 := by
  unfold candidates
  have row : edge 5981 = (SortedNatTree.contains FastRows.row5981) := by
    change Data.row5981.contains = (SortedNatTree.contains FastRows.row5981)
    exact FastRows.row5981_checked.symm
  rw [row]
  decide
theorem root14Step47 : scan edge (search edge 3) 4 [59] 59 root14Tail47 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail48 winner := by
  rw [root14Tail47, scan, if_neg (by decide), root14Candidates47_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail48
    (if 3 ≤ root14Candidates47.length then search edge 3 [59, 5981] root14Candidates47 6040 winner else winner) = _
  rw [if_neg (by decide)]
def root14Candidates48 : List Nat := [6569]
set_option maxRecDepth 8192 in
theorem root14Candidates48_checked : candidates edge 4 59 6047 root14Tail49 winner.bound = root14Candidates48 := by
  unfold candidates
  have row : edge 6047 = (SortedNatTree.contains FastRows.row6047) := by
    change Data.row6047.contains = (SortedNatTree.contains FastRows.row6047)
    exact FastRows.row6047_checked.symm
  rw [row]
  decide
theorem root14Step48 : scan edge (search edge 3) 4 [59] 59 root14Tail48 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail49 winner := by
  rw [root14Tail48, scan, if_neg (by decide), root14Candidates48_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail49
    (if 3 ≤ root14Candidates48.length then search edge 3 [59, 6047] root14Candidates48 6106 winner else winner) = _
  rw [if_neg (by decide)]
end ProjectEulerSolutions.PrimePairLarge.Search
