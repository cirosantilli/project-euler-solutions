import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot006Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root6Candidates32 : List Nat := [6053, 11321, 16553, 17519]
set_option maxRecDepth 8192 in
theorem root6Candidates32_checked : candidates edge 4 23 4007 root6Tail33 winner.bound = root6Candidates32 := by
  unfold candidates
  have row : edge 4007 = (SortedNatTree.contains FastRows.row4007) := by
    change Data.row4007.contains = (SortedNatTree.contains FastRows.row4007)
    exact FastRows.row4007_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root6Child32 : search edge 3 [23, 4007] root6Candidates32 4030 winner = winner := by decide
theorem root6Step32 : scan edge (search edge 3) 4 [23] 23 root6Tail32 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail33 winner := by
  rw [root6Tail32, scan, if_neg (by decide), root6Candidates32_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail33
    (if 3 ≤ root6Candidates32.length then search edge 3 [23, 4007] root6Candidates32 4030 winner else winner) = _
  rw [if_pos (by decide), root6Child32]
def root6Candidates33 : List Nat := [7607, 7673, 11159, 15233]
set_option maxRecDepth 8192 in
theorem root6Candidates33_checked : candidates edge 4 23 4139 root6Tail34 winner.bound = root6Candidates33 := by
  unfold candidates
  have row : edge 4139 = (SortedNatTree.contains FastRows.row4139) := by
    change Data.row4139.contains = (SortedNatTree.contains FastRows.row4139)
    exact FastRows.row4139_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root6Child33 : search edge 3 [23, 4139] root6Candidates33 4162 winner = winner := by decide
theorem root6Step33 : scan edge (search edge 3) 4 [23] 23 root6Tail33 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail34 winner := by
  rw [root6Tail33, scan, if_neg (by decide), root6Candidates33_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail34
    (if 3 ≤ root6Candidates33.length then search edge 3 [23, 4139] root6Candidates33 4162 winner else winner) = _
  rw [if_pos (by decide), root6Child33]
def root6Candidates34 : List Nat := [5483]
set_option maxRecDepth 8192 in
theorem root6Candidates34_checked : candidates edge 4 23 4211 root6Tail35 winner.bound = root6Candidates34 := by
  unfold candidates
  have row : edge 4211 = (SortedNatTree.contains FastRows.row4211) := by
    change Data.row4211.contains = (SortedNatTree.contains FastRows.row4211)
    exact FastRows.row4211_checked.symm
  rw [row]
  decide
theorem root6Step34 : scan edge (search edge 3) 4 [23] 23 root6Tail34 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail35 winner := by
  rw [root6Tail34, scan, if_neg (by decide), root6Candidates34_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail35
    (if 3 ≤ root6Candidates34.length then search edge 3 [23, 4211] root6Candidates34 4234 winner else winner) = _
  rw [if_neg (by decide)]
def root6Candidates35 : List Nat := [5441, 5813, 18077]
set_option maxRecDepth 8192 in
theorem root6Candidates35_checked : candidates edge 4 23 4463 root6Tail36 winner.bound = root6Candidates35 := by
  unfold candidates
  have row : edge 4463 = (SortedNatTree.contains FastRows.row4463) := by
    change Data.row4463.contains = (SortedNatTree.contains FastRows.row4463)
    exact FastRows.row4463_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root6Child35 : search edge 3 [23, 4463] root6Candidates35 4486 winner = winner := by decide
theorem root6Step35 : scan edge (search edge 3) 4 [23] 23 root6Tail35 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail36 winner := by
  rw [root6Tail35, scan, if_neg (by decide), root6Candidates35_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail36
    (if 3 ≤ root6Candidates35.length then search edge 3 [23, 4463] root6Candidates35 4486 winner else winner) = _
  rw [if_pos (by decide), root6Child35]
def root6Candidates36 : List Nat := [7547, 11321]
set_option maxRecDepth 8192 in
theorem root6Candidates36_checked : candidates edge 4 23 4547 root6Tail37 winner.bound = root6Candidates36 := by
  unfold candidates
  have row : edge 4547 = (SortedNatTree.contains FastRows.row4547) := by
    change Data.row4547.contains = (SortedNatTree.contains FastRows.row4547)
    exact FastRows.row4547_checked.symm
  rw [row]
  decide
theorem root6Step36 : scan edge (search edge 3) 4 [23] 23 root6Tail36 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail37 winner := by
  rw [root6Tail36, scan, if_neg (by decide), root6Candidates36_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail37
    (if 3 ≤ root6Candidates36.length then search edge 3 [23, 4547] root6Candidates36 4570 winner else winner) = _
  rw [if_neg (by decide)]
def root6Candidates37 : List Nat := [5813]
set_option maxRecDepth 8192 in
theorem root6Candidates37_checked : candidates edge 4 23 4721 root6Tail38 winner.bound = root6Candidates37 := by
  unfold candidates
  have row : edge 4721 = (SortedNatTree.contains FastRows.row4721) := by
    change Data.row4721.contains = (SortedNatTree.contains FastRows.row4721)
    exact FastRows.row4721_checked.symm
  rw [row]
  decide
theorem root6Step37 : scan edge (search edge 3) 4 [23] 23 root6Tail37 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail38 winner := by
  rw [root6Tail37, scan, if_neg (by decide), root6Candidates37_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail38
    (if 3 ≤ root6Candidates37.length then search edge 3 [23, 4721] root6Candidates37 4744 winner else winner) = _
  rw [if_neg (by decide)]
def root6Candidates38 : List Nat := [5009, 17747]
set_option maxRecDepth 8192 in
theorem root6Candidates38_checked : candidates edge 4 23 4931 root6Tail39 winner.bound = root6Candidates38 := by
  unfold candidates
  have row : edge 4931 = (SortedNatTree.contains FastRows.row4931) := by
    change Data.row4931.contains = (SortedNatTree.contains FastRows.row4931)
    exact FastRows.row4931_checked.symm
  rw [row]
  decide
theorem root6Step38 : scan edge (search edge 3) 4 [23] 23 root6Tail38 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail39 winner := by
  rw [root6Tail38, scan, if_neg (by decide), root6Candidates38_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail39
    (if 3 ≤ root6Candidates38.length then search edge 3 [23, 4931] root6Candidates38 4954 winner else winner) = _
  rw [if_neg (by decide)]
def root6Candidates39 : List Nat := [5849, 8093]
set_option maxRecDepth 8192 in
theorem root6Candidates39_checked : candidates edge 4 23 5009 root6Tail40 winner.bound = root6Candidates39 := by
  unfold candidates
  have row : edge 5009 = (SortedNatTree.contains FastRows.row5009) := by
    change Data.row5009.contains = (SortedNatTree.contains FastRows.row5009)
    exact FastRows.row5009_checked.symm
  rw [row]
  decide
theorem root6Step39 : scan edge (search edge 3) 4 [23] 23 root6Tail39 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail40 winner := by
  rw [root6Tail39, scan, if_neg (by decide), root6Candidates39_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail40
    (if 3 ≤ root6Candidates39.length then search edge 3 [23, 5009] root6Candidates39 5032 winner else winner) = _
  rw [if_neg (by decide)]
def root6Candidates40 : List Nat := [9929]
set_option maxRecDepth 8192 in
theorem root6Candidates40_checked : candidates edge 4 23 5051 root6Tail41 winner.bound = root6Candidates40 := by
  unfold candidates
  have row : edge 5051 = (SortedNatTree.contains FastRows.row5051) := by
    change Data.row5051.contains = (SortedNatTree.contains FastRows.row5051)
    exact FastRows.row5051_checked.symm
  rw [row]
  decide
theorem root6Step40 : scan edge (search edge 3) 4 [23] 23 root6Tail40 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail41 winner := by
  rw [root6Tail40, scan, if_neg (by decide), root6Candidates40_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail41
    (if 3 ≤ root6Candidates40.length then search edge 3 [23, 5051] root6Candidates40 5074 winner else winner) = _
  rw [if_neg (by decide)]
def root6Candidates41 : List Nat := [5849, 7547, 13799]
set_option maxRecDepth 8192 in
theorem root6Candidates41_checked : candidates edge 4 23 5441 root6Tail42 winner.bound = root6Candidates41 := by
  unfold candidates
  have row : edge 5441 = (SortedNatTree.contains FastRows.row5441) := by
    change Data.row5441.contains = (SortedNatTree.contains FastRows.row5441)
    exact FastRows.row5441_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root6Child41 : search edge 3 [23, 5441] root6Candidates41 5464 winner = winner := by decide
theorem root6Step41 : scan edge (search edge 3) 4 [23] 23 root6Tail41 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail42 winner := by
  rw [root6Tail41, scan, if_neg (by decide), root6Candidates41_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail42
    (if 3 ≤ root6Candidates41.length then search edge 3 [23, 5441] root6Candidates41 5464 winner else winner) = _
  rw [if_pos (by decide), root6Child41]
def root6Candidates42 : List Nat := [7487, 8423, 11321]
set_option maxRecDepth 8192 in
theorem root6Candidates42_checked : candidates edge 4 23 5483 root6Tail43 winner.bound = root6Candidates42 := by
  unfold candidates
  have row : edge 5483 = (SortedNatTree.contains FastRows.row5483) := by
    change Data.row5483.contains = (SortedNatTree.contains FastRows.row5483)
    exact FastRows.row5483_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root6Child42 : search edge 3 [23, 5483] root6Candidates42 5506 winner = winner := by decide
theorem root6Step42 : scan edge (search edge 3) 4 [23] 23 root6Tail42 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail43 winner := by
  rw [root6Tail42, scan, if_neg (by decide), root6Candidates42_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail43
    (if 3 ≤ root6Candidates42.length then search edge 3 [23, 5483] root6Candidates42 5506 winner else winner) = _
  rw [if_pos (by decide), root6Child42]
def root6Candidates43 : List Nat := [15233, 16529]
set_option maxRecDepth 8192 in
theorem root6Candidates43_checked : candidates edge 4 23 5591 root6Tail44 winner.bound = root6Candidates43 := by
  unfold candidates
  have row : edge 5591 = (SortedNatTree.contains FastRows.row5591) := by
    change Data.row5591.contains = (SortedNatTree.contains FastRows.row5591)
    exact FastRows.row5591_checked.symm
  rw [row]
  decide
theorem root6Step43 : scan edge (search edge 3) 4 [23] 23 root6Tail43 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail44 winner := by
  rw [root6Tail43, scan, if_neg (by decide), root6Candidates43_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail44
    (if 3 ≤ root6Candidates43.length then search edge 3 [23, 5591] root6Candidates43 5614 winner else winner) = _
  rw [if_neg (by decide)]
def root6Candidates44 : List Nat := [7151, 14957, 17747]
set_option maxRecDepth 8192 in
theorem root6Candidates44_checked : candidates edge 4 23 5813 root6Tail45 winner.bound = root6Candidates44 := by
  unfold candidates
  have row : edge 5813 = (SortedNatTree.contains FastRows.row5813) := by
    change Data.row5813.contains = (SortedNatTree.contains FastRows.row5813)
    exact FastRows.row5813_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root6Child44 : search edge 3 [23, 5813] root6Candidates44 5836 winner = winner := by decide
theorem root6Step44 : scan edge (search edge 3) 4 [23] 23 root6Tail44 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail45 winner := by
  rw [root6Tail44, scan, if_neg (by decide), root6Candidates44_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail45
    (if 3 ≤ root6Candidates44.length then search edge 3 [23, 5813] root6Candidates44 5836 winner else winner) = _
  rw [if_pos (by decide), root6Child44]
def root6Candidates45 : List Nat := [8681, 9929]
set_option maxRecDepth 8192 in
theorem root6Candidates45_checked : candidates edge 4 23 5849 root6Tail46 winner.bound = root6Candidates45 := by
  unfold candidates
  have row : edge 5849 = (SortedNatTree.contains FastRows.row5849) := by
    change Data.row5849.contains = (SortedNatTree.contains FastRows.row5849)
    exact FastRows.row5849_checked.symm
  rw [row]
  decide
theorem root6Step45 : scan edge (search edge 3) 4 [23] 23 root6Tail45 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail46 winner := by
  rw [root6Tail45, scan, if_neg (by decide), root6Candidates45_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail46
    (if 3 ≤ root6Candidates45.length then search edge 3 [23, 5849] root6Candidates45 5872 winner else winner) = _
  rw [if_neg (by decide)]
def root6Candidates46 : List Nat := [8237, 9137]
set_option maxRecDepth 8192 in
theorem root6Candidates46_checked : candidates edge 4 23 5927 root6Tail47 winner.bound = root6Candidates46 := by
  unfold candidates
  have row : edge 5927 = (SortedNatTree.contains FastRows.row5927) := by
    change Data.row5927.contains = (SortedNatTree.contains FastRows.row5927)
    exact FastRows.row5927_checked.symm
  rw [row]
  decide
theorem root6Step46 : scan edge (search edge 3) 4 [23] 23 root6Tail46 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail47 winner := by
  rw [root6Tail46, scan, if_neg (by decide), root6Candidates46_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail47
    (if 3 ≤ root6Candidates46.length then search edge 3 [23, 5927] root6Candidates46 5950 winner else winner) = _
  rw [if_neg (by decide)]
def root6Candidates47 : List Nat := [8171, 9137, 11321]
set_option maxRecDepth 8192 in
theorem root6Candidates47_checked : candidates edge 4 23 6053 root6Tail48 winner.bound = root6Candidates47 := by
  unfold candidates
  have row : edge 6053 = (SortedNatTree.contains FastRows.row6053) := by
    change Data.row6053.contains = (SortedNatTree.contains FastRows.row6053)
    exact FastRows.row6053_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root6Child47 : search edge 3 [23, 6053] root6Candidates47 6076 winner = winner := by decide
theorem root6Step47 : scan edge (search edge 3) 4 [23] 23 root6Tail47 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail48 winner := by
  rw [root6Tail47, scan, if_neg (by decide), root6Candidates47_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail48
    (if 3 ≤ root6Candidates47.length then search edge 3 [23, 6053] root6Candidates47 6076 winner else winner) = _
  rw [if_pos (by decide), root6Child47]
def root6Candidates48 : List Nat := [8237]
set_option maxRecDepth 8192 in
theorem root6Candidates48_checked : candidates edge 4 23 6323 root6Tail49 winner.bound = root6Candidates48 := by
  unfold candidates
  have row : edge 6323 = (SortedNatTree.contains FastRows.row6323) := by
    change Data.row6323.contains = (SortedNatTree.contains FastRows.row6323)
    exact FastRows.row6323_checked.symm
  rw [row]
  decide
theorem root6Step48 : scan edge (search edge 3) 4 [23] 23 root6Tail48 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail49 winner := by
  rw [root6Tail48, scan, if_neg (by decide), root6Candidates48_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail49
    (if 3 ≤ root6Candidates48.length then search edge 3 [23, 6323] root6Candidates48 6346 winner else winner) = _
  rw [if_neg (by decide)]
def root6Candidates49 : List Nat := [7331]
set_option maxRecDepth 8192 in
theorem root6Candidates49_checked : candidates edge 4 23 6329 root6Tail50 winner.bound = root6Candidates49 := by
  unfold candidates
  have row : edge 6329 = (SortedNatTree.contains FastRows.row6329) := by
    change Data.row6329.contains = (SortedNatTree.contains FastRows.row6329)
    exact FastRows.row6329_checked.symm
  rw [row]
  decide
theorem root6Step49 : scan edge (search edge 3) 4 [23] 23 root6Tail49 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail50 winner := by
  rw [root6Tail49, scan, if_neg (by decide), root6Candidates49_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail50
    (if 3 ≤ root6Candidates49.length then search edge 3 [23, 6329] root6Candidates49 6352 winner else winner) = _
  rw [if_neg (by decide)]
def root6Candidates50 : List Nat := [8171, 8747]
set_option maxRecDepth 8192 in
theorem root6Candidates50_checked : candidates edge 4 23 6449 root6Tail51 winner.bound = root6Candidates50 := by
  unfold candidates
  have row : edge 6449 = (SortedNatTree.contains FastRows.row6449) := by
    change Data.row6449.contains = (SortedNatTree.contains FastRows.row6449)
    exact FastRows.row6449_checked.symm
  rw [row]
  decide
theorem root6Step50 : scan edge (search edge 3) 4 [23] 23 root6Tail50 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail51 winner := by
  rw [root6Tail50, scan, if_neg (by decide), root6Candidates50_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail51
    (if 3 ≤ root6Candidates50.length then search edge 3 [23, 6449] root6Candidates50 6472 winner else winner) = _
  rw [if_neg (by decide)]
end ProjectEulerSolutions.PrimePairLarge.Search
