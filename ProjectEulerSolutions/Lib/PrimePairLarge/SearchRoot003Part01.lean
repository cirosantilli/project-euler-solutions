import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot003Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root3Candidates32 : List Nat := [5209, 7477, 19207, 19273, 19543, 20173]
set_option maxRecDepth 8192 in
theorem root3Candidates32_checked : candidates edge 4 13 3187 root3Tail33 seed.bound = root3Candidates32 := by
  unfold candidates
  have row : edge 3187 = (SortedNatTree.contains FastRows.row3187) := by
    change Data.row3187.contains = (SortedNatTree.contains FastRows.row3187)
    exact FastRows.row3187_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child32 : search edge 3 [13, 3187] root3Candidates32 3200 seed = seed := by decide
theorem root3Step32 : scan edge (search edge 3) 4 [13] 13 root3Tail32 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail33 seed := by
  rw [root3Tail32, scan, if_neg (by decide), root3Candidates32_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail33
    (if 3 ≤ root3Candidates32.length then search edge 3 [13, 3187] root3Candidates32 3200 seed else seed) = _
  rw [if_pos (by decide), root3Child32]
def root3Candidates33 : List Nat := [7993, 8599, 14437]
set_option maxRecDepth 8192 in
theorem root3Candidates33_checked : candidates edge 4 13 3499 root3Tail34 seed.bound = root3Candidates33 := by
  unfold candidates
  have row : edge 3499 = (SortedNatTree.contains FastRows.row3499) := by
    change Data.row3499.contains = (SortedNatTree.contains FastRows.row3499)
    exact FastRows.row3499_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child33 : search edge 3 [13, 3499] root3Candidates33 3512 seed = seed := by decide
theorem root3Step33 : scan edge (search edge 3) 4 [13] 13 root3Tail33 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail34 seed := by
  rw [root3Tail33, scan, if_neg (by decide), root3Candidates33_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail34
    (if 3 ≤ root3Candidates33.length then search edge 3 [13, 3499] root3Candidates33 3512 seed else seed) = _
  rw [if_pos (by decide), root3Child33]
def root3Candidates34 : List Nat := [3583, 8053, 9397, 9721]
set_option maxRecDepth 8192 in
theorem root3Candidates34_checked : candidates edge 4 13 3559 root3Tail35 seed.bound = root3Candidates34 := by
  unfold candidates
  have row : edge 3559 = (SortedNatTree.contains FastRows.row3559) := by
    change Data.row3559.contains = (SortedNatTree.contains FastRows.row3559)
    exact FastRows.row3559_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child34 : search edge 3 [13, 3559] root3Candidates34 3572 seed = seed := by decide
theorem root3Step34 : scan edge (search edge 3) 4 [13] 13 root3Tail34 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail35 seed := by
  rw [root3Tail34, scan, if_neg (by decide), root3Candidates34_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail35
    (if 3 ≤ root3Candidates34.length then search edge 3 [13, 3559] root3Candidates34 3572 seed else seed) = _
  rw [if_pos (by decide), root3Child34]
def root3Candidates35 : List Nat := [9787, 12517, 22147]
set_option maxRecDepth 8192 in
theorem root3Candidates35_checked : candidates edge 4 13 3583 root3Tail36 seed.bound = root3Candidates35 := by
  unfold candidates
  have row : edge 3583 = (SortedNatTree.contains FastRows.row3583) := by
    change Data.row3583.contains = (SortedNatTree.contains FastRows.row3583)
    exact FastRows.row3583_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child35 : search edge 3 [13, 3583] root3Candidates35 3596 seed = seed := by decide
theorem root3Step35 : scan edge (search edge 3) 4 [13] 13 root3Tail35 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail36 seed := by
  rw [root3Tail35, scan, if_neg (by decide), root3Candidates35_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail36
    (if 3 ≤ root3Candidates35.length then search edge 3 [13, 3583] root3Candidates35 3596 seed else seed) = _
  rw [if_pos (by decide), root3Child35]
def root3Candidates36 : List Nat := [5581, 6691, 8629, 17761, 18859]
set_option maxRecDepth 8192 in
theorem root3Candidates36_checked : candidates edge 4 13 3673 root3Tail37 seed.bound = root3Candidates36 := by
  unfold candidates
  have row : edge 3673 = (SortedNatTree.contains FastRows.row3673) := by
    change Data.row3673.contains = (SortedNatTree.contains FastRows.row3673)
    exact FastRows.row3673_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child36 : search edge 3 [13, 3673] root3Candidates36 3686 seed = seed := by decide
theorem root3Step36 : scan edge (search edge 3) 4 [13] 13 root3Tail36 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail37 seed := by
  rw [root3Tail36, scan, if_neg (by decide), root3Candidates36_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail37
    (if 3 ≤ root3Candidates36.length then search edge 3 [13, 3673] root3Candidates36 3686 seed else seed) = _
  rw [if_pos (by decide), root3Child36]
def root3Candidates37 : List Nat := [7927, 8053, 8893, 11971, 12517, 18661]
set_option maxRecDepth 8192 in
theorem root3Candidates37_checked : candidates edge 4 13 3967 root3Tail38 seed.bound = root3Candidates37 := by
  unfold candidates
  have row : edge 3967 = (SortedNatTree.contains FastRows.row3967) := by
    change Data.row3967.contains = (SortedNatTree.contains FastRows.row3967)
    exact FastRows.row3967_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child37 : search edge 3 [13, 3967] root3Candidates37 3980 seed = seed := by decide
theorem root3Step37 : scan edge (search edge 3) 4 [13] 13 root3Tail37 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail38 seed := by
  rw [root3Tail37, scan, if_neg (by decide), root3Candidates37_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail38
    (if 3 ≤ root3Candidates37.length then search edge 3 [13, 3967] root3Candidates37 3980 seed else seed) = _
  rw [if_pos (by decide), root3Child37]
def root3Candidates38 : List Nat := [21601]
set_option maxRecDepth 8192 in
theorem root3Candidates38_checked : candidates edge 4 13 4219 root3Tail39 seed.bound = root3Candidates38 := by
  unfold candidates
  have row : edge 4219 = (SortedNatTree.contains FastRows.row4219) := by
    change Data.row4219.contains = (SortedNatTree.contains FastRows.row4219)
    exact FastRows.row4219_checked.symm
  rw [row]
  decide
theorem root3Step38 : scan edge (search edge 3) 4 [13] 13 root3Tail38 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail39 seed := by
  rw [root3Tail38, scan, if_neg (by decide), root3Candidates38_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail39
    (if 3 ≤ root3Candidates38.length then search edge 3 [13, 4219] root3Candidates38 4232 seed else seed) = _
  rw [if_neg (by decide)]
def root3Candidates39 : List Nat := [4591, 8581]
set_option maxRecDepth 8192 in
theorem root3Candidates39_checked : candidates edge 4 13 4243 root3Tail40 seed.bound = root3Candidates39 := by
  unfold candidates
  have row : edge 4243 = (SortedNatTree.contains FastRows.row4243) := by
    change Data.row4243.contains = (SortedNatTree.contains FastRows.row4243)
    exact FastRows.row4243_checked.symm
  rw [row]
  decide
theorem root3Step39 : scan edge (search edge 3) 4 [13] 13 root3Tail39 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail40 seed := by
  rw [root3Tail39, scan, if_neg (by decide), root3Candidates39_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail40
    (if 3 ≤ root3Candidates39.length then search edge 3 [13, 4243] root3Candidates39 4256 seed else seed) = _
  rw [if_neg (by decide)]
def root3Candidates40 : List Nat := [4591, 7639, 9697, 9883]
set_option maxRecDepth 8192 in
theorem root3Candidates40_checked : candidates edge 4 13 4327 root3Tail41 seed.bound = root3Candidates40 := by
  unfold candidates
  have row : edge 4327 = (SortedNatTree.contains FastRows.row4327) := by
    change Data.row4327.contains = (SortedNatTree.contains FastRows.row4327)
    exact FastRows.row4327_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child40 : search edge 3 [13, 4327] root3Candidates40 4340 seed = seed := by decide
theorem root3Step40 : scan edge (search edge 3) 4 [13] 13 root3Tail40 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail41 seed := by
  rw [root3Tail40, scan, if_neg (by decide), root3Candidates40_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail41
    (if 3 ≤ root3Candidates40.length then search edge 3 [13, 4327] root3Candidates40 4340 seed else seed) = _
  rw [if_pos (by decide), root3Child40]
def root3Candidates41 : List Nat := [8461, 9091, 9199, 14779, 16963]
set_option maxRecDepth 8192 in
theorem root3Candidates41_checked : candidates edge 4 13 4363 root3Tail42 seed.bound = root3Candidates41 := by
  unfold candidates
  have row : edge 4363 = (SortedNatTree.contains FastRows.row4363) := by
    change Data.row4363.contains = (SortedNatTree.contains FastRows.row4363)
    exact FastRows.row4363_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child41 : search edge 3 [13, 4363] root3Candidates41 4376 seed = seed := by decide
theorem root3Step41 : scan edge (search edge 3) 4 [13] 13 root3Tail41 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail42 seed := by
  rw [root3Tail41, scan, if_neg (by decide), root3Candidates41_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail42
    (if 3 ≤ root3Candidates41.length then search edge 3 [13, 4363] root3Candidates41 4376 seed else seed) = _
  rw [if_pos (by decide), root3Child41]
def root3Candidates42 : List Nat := [4591, 5581, 6163, 6397, 6841, 7723, 9697, 10039, 16963]
set_option maxRecDepth 8192 in
theorem root3Candidates42_checked : candidates edge 4 13 4513 root3Tail43 seed.bound = root3Candidates42 := by
  unfold candidates
  have row : edge 4513 = (SortedNatTree.contains FastRows.row4513) := by
    change Data.row4513.contains = (SortedNatTree.contains FastRows.row4513)
    exact FastRows.row4513_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child42 : search edge 3 [13, 4513] root3Candidates42 4526 seed = seed := by decide
theorem root3Step42 : scan edge (search edge 3) 4 [13] 13 root3Tail42 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail43 seed := by
  rw [root3Tail42, scan, if_neg (by decide), root3Candidates42_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail43
    (if 3 ≤ root3Candidates42.length then search edge 3 [13, 4513] root3Candidates42 4526 seed else seed) = _
  rw [if_pos (by decide), root3Child42]
def root3Candidates43 : List Nat := [5197, 9883, 17359, 17443, 17683]
set_option maxRecDepth 8192 in
theorem root3Candidates43_checked : candidates edge 4 13 4591 root3Tail44 seed.bound = root3Candidates43 := by
  unfold candidates
  have row : edge 4591 = (SortedNatTree.contains FastRows.row4591) := by
    change Data.row4591.contains = (SortedNatTree.contains FastRows.row4591)
    exact FastRows.row4591_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child43 : search edge 3 [13, 4591] root3Candidates43 4604 seed = seed := by decide
theorem root3Step43 : scan edge (search edge 3) 4 [13] 13 root3Tail43 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail44 seed := by
  rw [root3Tail43, scan, if_neg (by decide), root3Candidates43_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail44
    (if 3 ≤ root3Candidates43.length then search edge 3 [13, 4591] root3Candidates43 4604 seed else seed) = _
  rw [if_pos (by decide), root3Child43]
def root3Candidates44 : List Nat := [17683, 18301]
set_option maxRecDepth 8192 in
theorem root3Candidates44_checked : candidates edge 4 13 4789 root3Tail45 seed.bound = root3Candidates44 := by
  unfold candidates
  have row : edge 4789 = (SortedNatTree.contains FastRows.row4789) := by
    change Data.row4789.contains = (SortedNatTree.contains FastRows.row4789)
    exact FastRows.row4789_checked.symm
  rw [row]
  decide
theorem root3Step44 : scan edge (search edge 3) 4 [13] 13 root3Tail44 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail45 seed := by
  rw [root3Tail44, scan, if_neg (by decide), root3Candidates44_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail45
    (if 3 ≤ root3Candidates44.length then search edge 3 [13, 4789] root3Candidates44 4802 seed else seed) = _
  rw [if_neg (by decide)]
def root3Candidates45 : List Nat := [5647, 7723, 10369, 17989, 18859, 19543]
set_option maxRecDepth 8192 in
theorem root3Candidates45_checked : candidates edge 4 13 4909 root3Tail46 seed.bound = root3Candidates45 := by
  unfold candidates
  have row : edge 4909 = (SortedNatTree.contains FastRows.row4909) := by
    change Data.row4909.contains = (SortedNatTree.contains FastRows.row4909)
    exact FastRows.row4909_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child45 : search edge 3 [13, 4909] root3Candidates45 4922 seed = seed := by decide
theorem root3Step45 : scan edge (search edge 3) 4 [13] 13 root3Tail45 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail46 seed := by
  rw [root3Tail45, scan, if_neg (by decide), root3Candidates45_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail46
    (if 3 ≤ root3Candidates45.length then search edge 3 [13, 4909] root3Candidates45 4922 seed else seed) = _
  rw [if_pos (by decide), root3Child45]
def root3Candidates46 : List Nat := [6343, 9697, 11311]
set_option maxRecDepth 8192 in
theorem root3Candidates46_checked : candidates edge 4 13 4951 root3Tail47 seed.bound = root3Candidates46 := by
  unfold candidates
  have row : edge 4951 = (SortedNatTree.contains FastRows.row4951) := by
    change Data.row4951.contains = (SortedNatTree.contains FastRows.row4951)
    exact FastRows.row4951_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child46 : search edge 3 [13, 4951] root3Candidates46 4964 seed = seed := by decide
theorem root3Step46 : scan edge (search edge 3) 4 [13] 13 root3Tail46 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail47 seed := by
  rw [root3Tail46, scan, if_neg (by decide), root3Candidates46_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail47
    (if 3 ≤ root3Candidates46.length then search edge 3 [13, 4951] root3Candidates46 4964 seed else seed) = _
  rw [if_pos (by decide), root3Child46]
def root3Candidates47 : List Nat := [6043, 20749]
set_option maxRecDepth 8192 in
theorem root3Candidates47_checked : candidates edge 4 13 5077 root3Tail48 seed.bound = root3Candidates47 := by
  unfold candidates
  have row : edge 5077 = (SortedNatTree.contains FastRows.row5077) := by
    change Data.row5077.contains = (SortedNatTree.contains FastRows.row5077)
    exact FastRows.row5077_checked.symm
  rw [row]
  decide
theorem root3Step47 : scan edge (search edge 3) 4 [13] 13 root3Tail47 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail48 seed := by
  rw [root3Tail47, scan, if_neg (by decide), root3Candidates47_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail48
    (if 3 ≤ root3Candidates47.length then search edge 3 [13, 5077] root3Candidates47 5090 seed else seed) = _
  rw [if_neg (by decide)]
def root3Candidates48 : List Nat := [5701, 6733, 8389, 8629, 18553]
set_option maxRecDepth 8192 in
theorem root3Candidates48_checked : candidates edge 4 13 5197 root3Tail49 seed.bound = root3Candidates48 := by
  unfold candidates
  have row : edge 5197 = (SortedNatTree.contains FastRows.row5197) := by
    change Data.row5197.contains = (SortedNatTree.contains FastRows.row5197)
    exact FastRows.row5197_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child48 : search edge 3 [13, 5197] root3Candidates48 5210 seed = winner := by decide
theorem root3Step48 : scan edge (search edge 3) 4 [13] 13 root3Tail48 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail49 winner := by
  rw [root3Tail48, scan, if_neg (by decide), root3Candidates48_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail49
    (if 3 ≤ root3Candidates48.length then search edge 3 [13, 5197] root3Candidates48 5210 seed else seed) = _
  rw [if_pos (by decide), root3Child48]
def root3Candidates49 : List Nat := [6397, 11971]
set_option maxRecDepth 8192 in
theorem root3Candidates49_checked : candidates edge 4 13 5209 root3Tail50 winner.bound = root3Candidates49 := by
  unfold candidates
  have row : edge 5209 = (SortedNatTree.contains FastRows.row5209) := by
    change Data.row5209.contains = (SortedNatTree.contains FastRows.row5209)
    exact FastRows.row5209_checked.symm
  rw [row]
  decide
theorem root3Step49 : scan edge (search edge 3) 4 [13] 13 root3Tail49 winner =
    scan edge (search edge 3) 4 [13] 13 root3Tail50 winner := by
  rw [root3Tail49, scan, if_neg (by decide), root3Candidates49_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail50
    (if 3 ≤ root3Candidates49.length then search edge 3 [13, 5209] root3Candidates49 5222 winner else winner) = _
  rw [if_neg (by decide)]
def root3Candidates50 : List Nat := [7507, 8599]
set_option maxRecDepth 8192 in
theorem root3Candidates50_checked : candidates edge 4 13 5431 root3Tail51 winner.bound = root3Candidates50 := by
  unfold candidates
  have row : edge 5431 = (SortedNatTree.contains FastRows.row5431) := by
    change Data.row5431.contains = (SortedNatTree.contains FastRows.row5431)
    exact FastRows.row5431_checked.symm
  rw [row]
  decide
theorem root3Step50 : scan edge (search edge 3) 4 [13] 13 root3Tail50 winner =
    scan edge (search edge 3) 4 [13] 13 root3Tail51 winner := by
  rw [root3Tail50, scan, if_neg (by decide), root3Candidates50_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail51
    (if 3 ≤ root3Candidates50.length then search edge 3 [13, 5431] root3Candidates50 5444 winner else winner) = _
  rw [if_neg (by decide)]
def root3Candidates51 : List Nat := [6991]
set_option maxRecDepth 8192 in
theorem root3Candidates51_checked : candidates edge 4 13 5581 root3Tail52 winner.bound = root3Candidates51 := by
  unfold candidates
  have row : edge 5581 = (SortedNatTree.contains FastRows.row5581) := by
    change Data.row5581.contains = (SortedNatTree.contains FastRows.row5581)
    exact FastRows.row5581_checked.symm
  rw [row]
  decide
theorem root3Step51 : scan edge (search edge 3) 4 [13] 13 root3Tail51 winner =
    scan edge (search edge 3) 4 [13] 13 root3Tail52 winner := by
  rw [root3Tail51, scan, if_neg (by decide), root3Candidates51_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail52
    (if 3 ≤ root3Candidates51.length then search edge 3 [13, 5581] root3Candidates51 5594 winner else winner) = _
  rw [if_neg (by decide)]
def root3Candidates52 : List Nat := [6421, 7639, 10369, 17761]
set_option maxRecDepth 8192 in
theorem root3Candidates52_checked : candidates edge 4 13 5623 root3Tail53 winner.bound = root3Candidates52 := by
  unfold candidates
  have row : edge 5623 = (SortedNatTree.contains FastRows.row5623) := by
    change Data.row5623.contains = (SortedNatTree.contains FastRows.row5623)
    exact FastRows.row5623_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child52 : search edge 3 [13, 5623] root3Candidates52 5636 winner = winner := by decide
theorem root3Step52 : scan edge (search edge 3) 4 [13] 13 root3Tail52 winner =
    scan edge (search edge 3) 4 [13] 13 root3Tail53 winner := by
  rw [root3Tail52, scan, if_neg (by decide), root3Candidates52_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail53
    (if 3 ≤ root3Candidates52.length then search edge 3 [13, 5623] root3Candidates52 5636 winner else winner) = _
  rw [if_pos (by decide), root3Child52]
def root3Candidates53 : List Nat := [6733, 11311, 14779]
set_option maxRecDepth 8192 in
theorem root3Candidates53_checked : candidates edge 4 13 5647 root3Tail54 winner.bound = root3Candidates53 := by
  unfold candidates
  have row : edge 5647 = (SortedNatTree.contains FastRows.row5647) := by
    change Data.row5647.contains = (SortedNatTree.contains FastRows.row5647)
    exact FastRows.row5647_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child53 : search edge 3 [13, 5647] root3Candidates53 5660 winner = winner := by decide
theorem root3Step53 : scan edge (search edge 3) 4 [13] 13 root3Tail53 winner =
    scan edge (search edge 3) 4 [13] 13 root3Tail54 winner := by
  rw [root3Tail53, scan, if_neg (by decide), root3Candidates53_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail54
    (if 3 ≤ root3Candidates53.length then search edge 3 [13, 5647] root3Candidates53 5660 winner else winner) = _
  rw [if_pos (by decide), root3Child53]
def root3Candidates54 : List Nat := [6733, 8389, 9439]
set_option maxRecDepth 8192 in
theorem root3Candidates54_checked : candidates edge 4 13 5701 root3Tail55 winner.bound = root3Candidates54 := by
  unfold candidates
  have row : edge 5701 = (SortedNatTree.contains FastRows.row5701) := by
    change Data.row5701.contains = (SortedNatTree.contains FastRows.row5701)
    exact FastRows.row5701_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child54 : search edge 3 [13, 5701] root3Candidates54 5714 winner = winner := by decide
theorem root3Step54 : scan edge (search edge 3) 4 [13] 13 root3Tail54 winner =
    scan edge (search edge 3) 4 [13] 13 root3Tail55 winner := by
  rw [root3Tail54, scan, if_neg (by decide), root3Candidates54_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail55
    (if 3 ≤ root3Candidates54.length then search edge 3 [13, 5701] root3Candidates54 5714 winner else winner) = _
  rw [if_pos (by decide), root3Child54]
def root3Candidates55 : List Nat := [7507, 13567]
set_option maxRecDepth 8192 in
theorem root3Candidates55_checked : candidates edge 4 13 5851 root3Tail56 winner.bound = root3Candidates55 := by
  unfold candidates
  have row : edge 5851 = (SortedNatTree.contains FastRows.row5851) := by
    change Data.row5851.contains = (SortedNatTree.contains FastRows.row5851)
    exact FastRows.row5851_checked.symm
  rw [row]
  decide
theorem root3Step55 : scan edge (search edge 3) 4 [13] 13 root3Tail55 winner =
    scan edge (search edge 3) 4 [13] 13 root3Tail56 winner := by
  rw [root3Tail55, scan, if_neg (by decide), root3Candidates55_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail56
    (if 3 ≤ root3Candidates55.length then search edge 3 [13, 5851] root3Candidates55 5864 winner else winner) = _
  rw [if_neg (by decide)]
def root3Candidates56 : List Nat := [8461, 12517, 16963]
set_option maxRecDepth 8192 in
theorem root3Candidates56_checked : candidates edge 4 13 6043 root3Tail57 winner.bound = root3Candidates56 := by
  unfold candidates
  have row : edge 6043 = (SortedNatTree.contains FastRows.row6043) := by
    change Data.row6043.contains = (SortedNatTree.contains FastRows.row6043)
    exact FastRows.row6043_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child56 : search edge 3 [13, 6043] root3Candidates56 6056 winner = winner := by decide
theorem root3Step56 : scan edge (search edge 3) 4 [13] 13 root3Tail56 winner =
    scan edge (search edge 3) 4 [13] 13 root3Tail57 winner := by
  rw [root3Tail56, scan, if_neg (by decide), root3Candidates56_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail57
    (if 3 ≤ root3Candidates56.length then search edge 3 [13, 6043] root3Candidates56 6056 winner else winner) = _
  rw [if_pos (by decide), root3Child56]
def root3Candidates57 : List Nat := [11287, 14437, 19609]
set_option maxRecDepth 8192 in
theorem root3Candidates57_checked : candidates edge 4 13 6163 root3Tail58 winner.bound = root3Candidates57 := by
  unfold candidates
  have row : edge 6163 = (SortedNatTree.contains FastRows.row6163) := by
    change Data.row6163.contains = (SortedNatTree.contains FastRows.row6163)
    exact FastRows.row6163_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child57 : search edge 3 [13, 6163] root3Candidates57 6176 winner = winner := by decide
theorem root3Step57 : scan edge (search edge 3) 4 [13] 13 root3Tail57 winner =
    scan edge (search edge 3) 4 [13] 13 root3Tail58 winner := by
  rw [root3Tail57, scan, if_neg (by decide), root3Candidates57_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail58
    (if 3 ≤ root3Candidates57.length then search edge 3 [13, 6163] root3Candidates57 6176 winner else winner) = _
  rw [if_pos (by decide), root3Child57]
def root3Candidates58 : List Nat := [9661, 9697]
set_option maxRecDepth 8192 in
theorem root3Candidates58_checked : candidates edge 4 13 6343 root3Tail59 winner.bound = root3Candidates58 := by
  unfold candidates
  have row : edge 6343 = (SortedNatTree.contains FastRows.row6343) := by
    change Data.row6343.contains = (SortedNatTree.contains FastRows.row6343)
    exact FastRows.row6343_checked.symm
  rw [row]
  decide
theorem root3Step58 : scan edge (search edge 3) 4 [13] 13 root3Tail58 winner =
    scan edge (search edge 3) 4 [13] 13 root3Tail59 winner := by
  rw [root3Tail58, scan, if_neg (by decide), root3Candidates58_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail59
    (if 3 ≤ root3Candidates58.length then search edge 3 [13, 6343] root3Candidates58 6356 winner else winner) = _
  rw [if_neg (by decide)]
def root3Candidates59 : List Nat := [8179, 9661, 10993, 18553]
set_option maxRecDepth 8192 in
theorem root3Candidates59_checked : candidates edge 4 13 6397 root3Tail60 winner.bound = root3Candidates59 := by
  unfold candidates
  have row : edge 6397 = (SortedNatTree.contains FastRows.row6397) := by
    change Data.row6397.contains = (SortedNatTree.contains FastRows.row6397)
    exact FastRows.row6397_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child59 : search edge 3 [13, 6397] root3Candidates59 6410 winner = winner := by decide
theorem root3Step59 : scan edge (search edge 3) 4 [13] 13 root3Tail59 winner =
    scan edge (search edge 3) 4 [13] 13 root3Tail60 winner := by
  rw [root3Tail59, scan, if_neg (by decide), root3Candidates59_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail60
    (if 3 ≤ root3Candidates59.length then search edge 3 [13, 6397] root3Candidates59 6410 winner else winner) = _
  rw [if_pos (by decide), root3Child59]
def root3Candidates60 : List Nat := []
set_option maxRecDepth 8192 in
theorem root3Candidates60_checked : candidates edge 4 13 6421 root3Tail61 winner.bound = root3Candidates60 := by
  unfold candidates
  have row : edge 6421 = (SortedNatTree.contains FastRows.row6421) := by
    change Data.row6421.contains = (SortedNatTree.contains FastRows.row6421)
    exact FastRows.row6421_checked.symm
  rw [row]
  decide
theorem root3Step60 : scan edge (search edge 3) 4 [13] 13 root3Tail60 winner =
    scan edge (search edge 3) 4 [13] 13 root3Tail61 winner := by
  rw [root3Tail60, scan, if_neg (by decide), root3Candidates60_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail61
    (if 3 ≤ root3Candidates60.length then search edge 3 [13, 6421] root3Candidates60 6434 winner else winner) = _
  rw [if_neg (by decide)]
end ProjectEulerSolutions.PrimePairLarge.Search
