import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot009Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root9Candidates32 : List Nat := []
set_option maxRecDepth 8192 in
theorem root9Candidates32_checked : candidates edge 4 37 3181 root9Tail33 winner.bound = root9Candidates32 := by
  unfold candidates
  have row : edge 3181 = (SortedNatTree.contains FastRows.row3181) := by
    change Data.row3181.contains = (SortedNatTree.contains FastRows.row3181)
    exact FastRows.row3181_checked.symm
  rw [row]
  decide
theorem root9Step32 : scan edge (search edge 3) 4 [37] 37 root9Tail32 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail33 winner := by
  rw [root9Tail32, scan, if_neg (by decide), root9Candidates32_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail33
    (if 3 ≤ root9Candidates32.length then search edge 3 [37, 3181] root9Candidates32 3218 winner else winner) = _
  rw [if_neg (by decide)]
def root9Candidates33 : List Nat := [4093, 4729, 5227, 7477, 8101, 19273]
set_option maxRecDepth 8192 in
theorem root9Candidates33_checked : candidates edge 4 37 3187 root9Tail34 winner.bound = root9Candidates33 := by
  unfold candidates
  have row : edge 3187 = (SortedNatTree.contains FastRows.row3187) := by
    change Data.row3187.contains = (SortedNatTree.contains FastRows.row3187)
    exact FastRows.row3187_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root9Child33 : search edge 3 [37, 3187] root9Candidates33 3224 winner = winner := by decide
theorem root9Step33 : scan edge (search edge 3) 4 [37] 37 root9Tail33 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail34 winner := by
  rw [root9Tail33, scan, if_neg (by decide), root9Candidates33_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail34
    (if 3 ≤ root9Candidates33.length then search edge 3 [37, 3187] root9Candidates33 3224 winner else winner) = _
  rw [if_pos (by decide), root9Child33]
def root9Candidates34 : List Nat := [8011]
set_option maxRecDepth 8192 in
theorem root9Candidates34_checked : candidates edge 4 37 3463 root9Tail35 winner.bound = root9Candidates34 := by
  unfold candidates
  have row : edge 3463 = (SortedNatTree.contains FastRows.row3463) := by
    change Data.row3463.contains = (SortedNatTree.contains FastRows.row3463)
    exact FastRows.row3463_checked.symm
  rw [row]
  decide
theorem root9Step34 : scan edge (search edge 3) 4 [37] 37 root9Tail34 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail35 winner := by
  rw [root9Tail34, scan, if_neg (by decide), root9Candidates34_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail35
    (if 3 ≤ root9Candidates34.length then search edge 3 [37, 3463] root9Candidates34 3500 winner else winner) = _
  rw [if_neg (by decide)]
def root9Candidates35 : List Nat := [5623, 17041]
set_option maxRecDepth 8192 in
theorem root9Candidates35_checked : candidates edge 4 37 3613 root9Tail36 winner.bound = root9Candidates35 := by
  unfold candidates
  have row : edge 3613 = (SortedNatTree.contains FastRows.row3613) := by
    change Data.row3613.contains = (SortedNatTree.contains FastRows.row3613)
    exact FastRows.row3613_checked.symm
  rw [row]
  decide
theorem root9Step35 : scan edge (search edge 3) 4 [37] 37 root9Tail35 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail36 winner := by
  rw [root9Tail35, scan, if_neg (by decide), root9Candidates35_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail36
    (if 3 ≤ root9Candidates35.length then search edge 3 [37, 3613] root9Candidates35 3650 winner else winner) = _
  rw [if_neg (by decide)]
def root9Candidates36 : List Nat := [6823, 9721]
set_option maxRecDepth 8192 in
theorem root9Candidates36_checked : candidates edge 4 37 4093 root9Tail37 winner.bound = root9Candidates36 := by
  unfold candidates
  have row : edge 4093 = (SortedNatTree.contains FastRows.row4093) := by
    change Data.row4093.contains = (SortedNatTree.contains FastRows.row4093)
    exact FastRows.row4093_checked.symm
  rw [row]
  decide
theorem root9Step36 : scan edge (search edge 3) 4 [37] 37 root9Tail36 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail37 winner := by
  rw [root9Tail36, scan, if_neg (by decide), root9Candidates36_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail37
    (if 3 ≤ root9Candidates36.length then search edge 3 [37, 4093] root9Candidates36 4130 winner else winner) = _
  rw [if_neg (by decide)]
def root9Candidates37 : List Nat := [4789, 5779, 16759, 17209, 18397]
set_option maxRecDepth 8192 in
theorem root9Candidates37_checked : candidates edge 4 37 4159 root9Tail38 winner.bound = root9Candidates37 := by
  unfold candidates
  have row : edge 4159 = (SortedNatTree.contains FastRows.row4159) := by
    change Data.row4159.contains = (SortedNatTree.contains FastRows.row4159)
    exact FastRows.row4159_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root9Child37 : search edge 3 [37, 4159] root9Candidates37 4196 winner = winner := by decide
theorem root9Step37 : scan edge (search edge 3) 4 [37] 37 root9Tail37 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail38 winner := by
  rw [root9Tail37, scan, if_neg (by decide), root9Candidates37_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail38
    (if 3 ≤ root9Candidates37.length then search edge 3 [37, 4159] root9Candidates37 4196 winner else winner) = _
  rw [if_pos (by decide), root9Child37]
def root9Candidates38 : List Nat := [7297, 7537, 8101]
set_option maxRecDepth 8192 in
theorem root9Candidates38_checked : candidates edge 4 37 4177 root9Tail39 winner.bound = root9Candidates38 := by
  unfold candidates
  have row : edge 4177 = (SortedNatTree.contains FastRows.row4177) := by
    change Data.row4177.contains = (SortedNatTree.contains FastRows.row4177)
    exact FastRows.row4177_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root9Child38 : search edge 3 [37, 4177] root9Candidates38 4214 winner = winner := by decide
theorem root9Step38 : scan edge (search edge 3) 4 [37] 37 root9Tail38 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail39 winner := by
  rw [root9Tail38, scan, if_neg (by decide), root9Candidates38_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail39
    (if 3 ≤ root9Candidates38.length then search edge 3 [37, 4177] root9Candidates38 4214 winner else winner) = _
  rw [if_pos (by decide), root9Child38]
def root9Candidates39 : List Nat := [8761]
set_option maxRecDepth 8192 in
theorem root9Candidates39_checked : candidates edge 4 37 4603 root9Tail40 winner.bound = root9Candidates39 := by
  unfold candidates
  have row : edge 4603 = (SortedNatTree.contains FastRows.row4603) := by
    change Data.row4603.contains = (SortedNatTree.contains FastRows.row4603)
    exact FastRows.row4603_checked.symm
  rw [row]
  decide
theorem root9Step39 : scan edge (search edge 3) 4 [37] 37 root9Tail39 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail40 winner := by
  rw [root9Tail39, scan, if_neg (by decide), root9Candidates39_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail40
    (if 3 ≤ root9Candidates39.length then search edge 3 [37, 4603] root9Candidates39 4640 winner else winner) = _
  rw [if_neg (by decide)]
def root9Candidates40 : List Nat := [4909, 7963, 9859, 16879, 20947]
set_option maxRecDepth 8192 in
theorem root9Candidates40_checked : candidates edge 4 37 4729 root9Tail41 winner.bound = root9Candidates40 := by
  unfold candidates
  have row : edge 4729 = (SortedNatTree.contains FastRows.row4729) := by
    change Data.row4729.contains = (SortedNatTree.contains FastRows.row4729)
    exact FastRows.row4729_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root9Child40 : search edge 3 [37, 4729] root9Candidates40 4766 winner = winner := by decide
theorem root9Step40 : scan edge (search edge 3) 4 [37] 37 root9Tail40 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail41 winner := by
  rw [root9Tail40, scan, if_neg (by decide), root9Candidates40_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail41
    (if 3 ≤ root9Candidates40.length then search edge 3 [37, 4729] root9Candidates40 4766 winner else winner) = _
  rw [if_pos (by decide), root9Child40]
def root9Candidates41 : List Nat := [15991, 16369, 17041, 17209]
set_option maxRecDepth 8192 in
theorem root9Candidates41_checked : candidates edge 4 37 4789 root9Tail42 winner.bound = root9Candidates41 := by
  unfold candidates
  have row : edge 4789 = (SortedNatTree.contains FastRows.row4789) := by
    change Data.row4789.contains = (SortedNatTree.contains FastRows.row4789)
    exact FastRows.row4789_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root9Child41 : search edge 3 [37, 4789] root9Candidates41 4826 winner = winner := by decide
theorem root9Step41 : scan edge (search edge 3) 4 [37] 37 root9Tail41 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail42 winner := by
  rw [root9Tail41, scan, if_neg (by decide), root9Candidates41_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail42
    (if 3 ≤ root9Candidates41.length then search edge 3 [37, 4789] root9Candidates41 4826 winner else winner) = _
  rw [if_pos (by decide), root9Child41]
def root9Candidates42 : List Nat := [5857, 8713, 9157]
set_option maxRecDepth 8192 in
theorem root9Candidates42_checked : candidates edge 4 37 4909 root9Tail43 winner.bound = root9Candidates42 := by
  unfold candidates
  have row : edge 4909 = (SortedNatTree.contains FastRows.row4909) := by
    change Data.row4909.contains = (SortedNatTree.contains FastRows.row4909)
    exact FastRows.row4909_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root9Child42 : search edge 3 [37, 4909] root9Candidates42 4946 winner = winner := by decide
theorem root9Step42 : scan edge (search edge 3) 4 [37] 37 root9Tail42 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail43 winner := by
  rw [root9Tail42, scan, if_neg (by decide), root9Candidates42_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail43
    (if 3 ≤ root9Candidates42.length then search edge 3 [37, 4909] root9Candidates42 4946 winner else winner) = _
  rw [if_pos (by decide), root9Child42]
def root9Candidates43 : List Nat := [9931, 15031]
set_option maxRecDepth 8192 in
theorem root9Candidates43_checked : candidates edge 4 37 5101 root9Tail44 winner.bound = root9Candidates43 := by
  unfold candidates
  have row : edge 5101 = (SortedNatTree.contains FastRows.row5101) := by
    change Data.row5101.contains = (SortedNatTree.contains FastRows.row5101)
    exact FastRows.row5101_checked.symm
  rw [row]
  decide
theorem root9Step43 : scan edge (search edge 3) 4 [37] 37 root9Tail43 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail44 winner := by
  rw [root9Tail43, scan, if_neg (by decide), root9Candidates43_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail44
    (if 3 ≤ root9Candidates43.length then search edge 3 [37, 5101] root9Candidates43 5138 winner else winner) = _
  rw [if_neg (by decide)]
def root9Candidates44 : List Nat := [5449, 6823, 9007]
set_option maxRecDepth 8192 in
theorem root9Candidates44_checked : candidates edge 4 37 5113 root9Tail45 winner.bound = root9Candidates44 := by
  unfold candidates
  have row : edge 5113 = (SortedNatTree.contains FastRows.row5113) := by
    change Data.row5113.contains = (SortedNatTree.contains FastRows.row5113)
    exact FastRows.row5113_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root9Child44 : search edge 3 [37, 5113] root9Candidates44 5150 winner = winner := by decide
theorem root9Step44 : scan edge (search edge 3) 4 [37] 37 root9Tail44 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail45 winner := by
  rw [root9Tail44, scan, if_neg (by decide), root9Candidates44_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail45
    (if 3 ≤ root9Candidates44.length then search edge 3 [37, 5113] root9Candidates44 5150 winner else winner) = _
  rw [if_pos (by decide), root9Child44]
def root9Candidates45 : List Nat := []
set_option maxRecDepth 8192 in
theorem root9Candidates45_checked : candidates edge 4 37 5227 root9Tail46 winner.bound = root9Candidates45 := by
  unfold candidates
  have row : edge 5227 = (SortedNatTree.contains FastRows.row5227) := by
    change Data.row5227.contains = (SortedNatTree.contains FastRows.row5227)
    exact FastRows.row5227_checked.symm
  rw [row]
  decide
theorem root9Step45 : scan edge (search edge 3) 4 [37] 37 root9Tail45 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail46 winner := by
  rw [root9Tail45, scan, if_neg (by decide), root9Candidates45_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail46
    (if 3 ≤ root9Candidates45.length then search edge 3 [37, 5227] root9Candidates45 5264 winner else winner) = _
  rw [if_neg (by decide)]
def root9Candidates46 : List Nat := [7873, 11863]
set_option maxRecDepth 8192 in
theorem root9Candidates46_checked : candidates edge 4 37 5281 root9Tail47 winner.bound = root9Candidates46 := by
  unfold candidates
  have row : edge 5281 = (SortedNatTree.contains FastRows.row5281) := by
    change Data.row5281.contains = (SortedNatTree.contains FastRows.row5281)
    exact FastRows.row5281_checked.symm
  rw [row]
  decide
theorem root9Step46 : scan edge (search edge 3) 4 [37] 37 root9Tail46 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail47 winner := by
  rw [root9Tail46, scan, if_neg (by decide), root9Candidates46_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail47
    (if 3 ≤ root9Candidates46.length then search edge 3 [37, 5281] root9Candidates46 5318 winner else winner) = _
  rw [if_neg (by decide)]
def root9Candidates47 : List Nat := [8089]
set_option maxRecDepth 8192 in
theorem root9Candidates47_checked : candidates edge 4 37 5449 root9Tail48 winner.bound = root9Candidates47 := by
  unfold candidates
  have row : edge 5449 = (SortedNatTree.contains FastRows.row5449) := by
    change Data.row5449.contains = (SortedNatTree.contains FastRows.row5449)
    exact FastRows.row5449_checked.symm
  rw [row]
  decide
theorem root9Step47 : scan edge (search edge 3) 4 [37] 37 root9Tail47 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail48 winner := by
  rw [root9Tail47, scan, if_neg (by decide), root9Candidates47_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail48
    (if 3 ≤ root9Candidates47.length then search edge 3 [37, 5449] root9Candidates47 5486 winner else winner) = _
  rw [if_neg (by decide)]
def root9Candidates48 : List Nat := []
set_option maxRecDepth 8192 in
theorem root9Candidates48_checked : candidates edge 4 37 5623 root9Tail49 winner.bound = root9Candidates48 := by
  unfold candidates
  have row : edge 5623 = (SortedNatTree.contains FastRows.row5623) := by
    change Data.row5623.contains = (SortedNatTree.contains FastRows.row5623)
    exact FastRows.row5623_checked.symm
  rw [row]
  decide
theorem root9Step48 : scan edge (search edge 3) 4 [37] 37 root9Tail48 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail49 winner := by
  rw [root9Tail48, scan, if_neg (by decide), root9Candidates48_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail49
    (if 3 ≤ root9Candidates48.length then search edge 3 [37, 5623] root9Candidates48 5660 winner else winner) = _
  rw [if_neg (by decide)]
def root9Candidates49 : List Nat := [8761, 8941]
set_option maxRecDepth 8192 in
theorem root9Candidates49_checked : candidates edge 4 37 5779 root9Tail50 winner.bound = root9Candidates49 := by
  unfold candidates
  have row : edge 5779 = (SortedNatTree.contains FastRows.row5779) := by
    change Data.row5779.contains = (SortedNatTree.contains FastRows.row5779)
    exact FastRows.row5779_checked.symm
  rw [row]
  decide
theorem root9Step49 : scan edge (search edge 3) 4 [37] 37 root9Tail49 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail50 winner := by
  rw [root9Tail49, scan, if_neg (by decide), root9Candidates49_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail50
    (if 3 ≤ root9Candidates49.length then search edge 3 [37, 5779] root9Candidates49 5816 winner else winner) = _
  rw [if_neg (by decide)]
def root9Candidates50 : List Nat := [9007, 19273]
set_option maxRecDepth 8192 in
theorem root9Candidates50_checked : candidates edge 4 37 5857 root9Tail51 winner.bound = root9Candidates50 := by
  unfold candidates
  have row : edge 5857 = (SortedNatTree.contains FastRows.row5857) := by
    change Data.row5857.contains = (SortedNatTree.contains FastRows.row5857)
    exact FastRows.row5857_checked.symm
  rw [row]
  decide
theorem root9Step50 : scan edge (search edge 3) 4 [37] 37 root9Tail50 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail51 winner := by
  rw [root9Tail50, scan, if_neg (by decide), root9Candidates50_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail51
    (if 3 ≤ root9Candidates50.length then search edge 3 [37, 5857] root9Candidates50 5894 winner else winner) = _
  rw [if_neg (by decide)]
def root9Candidates51 : List Nat := [7963, 12757]
set_option maxRecDepth 8192 in
theorem root9Candidates51_checked : candidates edge 4 37 5923 root9Tail52 winner.bound = root9Candidates51 := by
  unfold candidates
  have row : edge 5923 = (SortedNatTree.contains FastRows.row5923) := by
    change Data.row5923.contains = (SortedNatTree.contains FastRows.row5923)
    exact FastRows.row5923_checked.symm
  rw [row]
  decide
theorem root9Step51 : scan edge (search edge 3) 4 [37] 37 root9Tail51 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail52 winner := by
  rw [root9Tail51, scan, if_neg (by decide), root9Candidates51_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail52
    (if 3 ≤ root9Candidates51.length then search edge 3 [37, 5923] root9Candidates51 5960 winner else winner) = _
  rw [if_neg (by decide)]
def root9Candidates52 : List Nat := []
set_option maxRecDepth 8192 in
theorem root9Candidates52_checked : candidates edge 4 37 6133 root9Tail53 winner.bound = root9Candidates52 := by
  unfold candidates
  have row : edge 6133 = (SortedNatTree.contains FastRows.row6133) := by
    change Data.row6133.contains = (SortedNatTree.contains FastRows.row6133)
    exact FastRows.row6133_checked.symm
  rw [row]
  decide
theorem root9Step52 : scan edge (search edge 3) 4 [37] 37 root9Tail52 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail53 winner := by
  rw [root9Tail52, scan, if_neg (by decide), root9Candidates52_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail53
    (if 3 ≤ root9Candidates52.length then search edge 3 [37, 6133] root9Candidates52 6170 winner else winner) = _
  rw [if_neg (by decide)]
def root9Candidates53 : List Nat := [11497, 16333]
set_option maxRecDepth 8192 in
theorem root9Candidates53_checked : candidates edge 4 37 6373 root9Tail54 winner.bound = root9Candidates53 := by
  unfold candidates
  have row : edge 6373 = (SortedNatTree.contains FastRows.row6373) := by
    change Data.row6373.contains = (SortedNatTree.contains FastRows.row6373)
    exact FastRows.row6373_checked.symm
  rw [row]
  decide
theorem root9Step53 : scan edge (search edge 3) 4 [37] 37 root9Tail53 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail54 winner := by
  rw [root9Tail53, scan, if_neg (by decide), root9Candidates53_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail54
    (if 3 ≤ root9Candidates53.length then search edge 3 [37, 6373] root9Candidates53 6410 winner else winner) = _
  rw [if_neg (by decide)]
def root9Candidates54 : List Nat := [11833]
set_option maxRecDepth 8192 in
theorem root9Candidates54_checked : candidates edge 4 37 6469 root9Tail55 winner.bound = root9Candidates54 := by
  unfold candidates
  have row : edge 6469 = (SortedNatTree.contains FastRows.row6469) := by
    change Data.row6469.contains = (SortedNatTree.contains FastRows.row6469)
    exact FastRows.row6469_checked.symm
  rw [row]
  decide
theorem root9Step54 : scan edge (search edge 3) 4 [37] 37 root9Tail54 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail55 winner := by
  rw [root9Tail54, scan, if_neg (by decide), root9Candidates54_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail55
    (if 3 ≤ root9Candidates54.length then search edge 3 [37, 6469] root9Candidates54 6506 winner else winner) = _
  rw [if_neg (by decide)]
end ProjectEulerSolutions.PrimePairLarge.Search
