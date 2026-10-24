import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot001Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root1Candidates32 : List Nat := [3361, 8779, 14557, 14563, 22369]
set_option maxRecDepth 8192 in
theorem root1Candidates32_checked : candidates edge 4 7 1999 root1Tail33 seed.bound = root1Candidates32 := by
  unfold candidates
  have row : edge 1999 = (SortedNatTree.contains FastRows.row1999) := by
    change Data.row1999.contains = (SortedNatTree.contains FastRows.row1999)
    exact FastRows.row1999_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child32 : search edge 3 [7, 1999] root1Candidates32 2006 seed = seed := by decide
theorem root1Step32 : scan edge (search edge 3) 4 [7] 7 root1Tail32 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail33 seed := by
  rw [root1Tail32, scan, if_neg (by decide), root1Candidates32_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail33
    (if 3 ≤ root1Candidates32.length then search edge 3 [7, 1999] root1Candidates32 2006 seed else seed) = _
  rw [if_pos (by decide), root1Child32]
def root1Candidates33 : List Nat := [2341, 2953, 3181, 4219, 15859, 21997]
set_option maxRecDepth 8192 in
theorem root1Candidates33_checked : candidates edge 4 7 2089 root1Tail34 seed.bound = root1Candidates33 := by
  unfold candidates
  have row : edge 2089 = (SortedNatTree.contains FastRows.row2089) := by
    change Data.row2089.contains = (SortedNatTree.contains FastRows.row2089)
    exact FastRows.row2089_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child33 : search edge 3 [7, 2089] root1Candidates33 2096 seed = seed := by decide
theorem root1Step33 : scan edge (search edge 3) 4 [7] 7 root1Tail33 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail34 seed := by
  rw [root1Tail33, scan, if_neg (by decide), root1Candidates33_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail34
    (if 3 ≤ root1Candidates33.length then search edge 3 [7, 2089] root1Candidates33 2096 seed else seed) = _
  rw [if_pos (by decide), root1Child33]
def root1Candidates34 : List Nat := [4357, 6949, 11437, 12511, 12781, 13147, 18457, 22639]
set_option maxRecDepth 8192 in
theorem root1Candidates34_checked : candidates edge 4 7 2161 root1Tail35 seed.bound = root1Candidates34 := by
  unfold candidates
  have row : edge 2161 = (SortedNatTree.contains FastRows.row2161) := by
    change Data.row2161.contains = (SortedNatTree.contains FastRows.row2161)
    exact FastRows.row2161_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child34 : search edge 3 [7, 2161] root1Candidates34 2168 seed = seed := by decide
theorem root1Step34 : scan edge (search edge 3) 4 [7] 7 root1Tail34 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail35 seed := by
  rw [root1Tail34, scan, if_neg (by decide), root1Candidates34_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail35
    (if 3 ≤ root1Candidates34.length then search edge 3 [7, 2161] root1Candidates34 2168 seed else seed) = _
  rw [if_pos (by decide), root1Child34]
def root1Candidates35 : List Nat := [3613, 5821, 7489, 8779, 9337, 11131, 17191, 18379]
set_option maxRecDepth 8192 in
theorem root1Candidates35_checked : candidates edge 4 7 2269 root1Tail36 seed.bound = root1Candidates35 := by
  unfold candidates
  have row : edge 2269 = (SortedNatTree.contains FastRows.row2269) := by
    change Data.row2269.contains = (SortedNatTree.contains FastRows.row2269)
    exact FastRows.row2269_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child35 : search edge 3 [7, 2269] root1Candidates35 2276 seed = seed := by decide
theorem root1Step35 : scan edge (search edge 3) 4 [7] 7 root1Tail35 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail36 seed := by
  rw [root1Tail35, scan, if_neg (by decide), root1Candidates35_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail36
    (if 3 ≤ root1Candidates35.length then search edge 3 [7, 2269] root1Candidates35 2276 seed else seed) = _
  rw [if_pos (by decide), root1Child35]
def root1Candidates36 : List Nat := [2341, 8101, 9613, 10663, 11437, 21481, 22147]
set_option maxRecDepth 8192 in
theorem root1Candidates36_checked : candidates edge 4 7 2287 root1Tail37 seed.bound = root1Candidates36 := by
  unfold candidates
  have row : edge 2287 = (SortedNatTree.contains FastRows.row2287) := by
    change Data.row2287.contains = (SortedNatTree.contains FastRows.row2287)
    exact FastRows.row2287_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child36 : search edge 3 [7, 2287] root1Candidates36 2294 seed = seed := by decide
theorem root1Step36 : scan edge (search edge 3) 4 [7] 7 root1Tail36 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail37 seed := by
  rw [root1Tail36, scan, if_neg (by decide), root1Candidates36_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail37
    (if 3 ≤ root1Candidates36.length then search edge 3 [7, 2287] root1Candidates36 2294 seed else seed) = _
  rw [if_pos (by decide), root1Child36]
def root1Candidates37 : List Nat := [7489, 11701, 12409, 13177, 13309, 14563, 15727, 18433]
set_option maxRecDepth 8192 in
theorem root1Candidates37_checked : candidates edge 4 7 2341 root1Tail38 seed.bound = root1Candidates37 := by
  unfold candidates
  have row : edge 2341 = (SortedNatTree.contains FastRows.row2341) := by
    change Data.row2341.contains = (SortedNatTree.contains FastRows.row2341)
    exact FastRows.row2341_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child37 : search edge 3 [7, 2341] root1Candidates37 2348 seed = seed := by decide
theorem root1Step37 : scan edge (search edge 3) 4 [7] 7 root1Tail37 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail38 seed := by
  rw [root1Tail37, scan, if_neg (by decide), root1Candidates37_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail38
    (if 3 ≤ root1Candidates37.length then search edge 3 [7, 2341] root1Candidates37 2348 seed else seed) = _
  rw [if_pos (by decide), root1Child37]
def root1Candidates38 : List Nat := [5689, 8167, 9601, 13477, 13627, 17659, 19801]
set_option maxRecDepth 8192 in
theorem root1Candidates38_checked : candidates edge 4 7 2467 root1Tail39 seed.bound = root1Candidates38 := by
  unfold candidates
  have row : edge 2467 = (SortedNatTree.contains FastRows.row2467) := by
    change Data.row2467.contains = (SortedNatTree.contains FastRows.row2467)
    exact FastRows.row2467_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child38 : search edge 3 [7, 2467] root1Candidates38 2474 seed = seed := by decide
theorem root1Step38 : scan edge (search edge 3) 4 [7] 7 root1Tail38 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail39 seed := by
  rw [root1Tail38, scan, if_neg (by decide), root1Candidates38_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail39
    (if 3 ≤ root1Candidates38.length then search edge 3 [7, 2467] root1Candidates38 2474 seed else seed) = _
  rw [if_pos (by decide), root1Child38]
def root1Candidates39 : List Nat := [6163, 6421, 9601, 9613, 18379, 21529]
set_option maxRecDepth 8192 in
theorem root1Candidates39_checked : candidates edge 4 7 2503 root1Tail40 seed.bound = root1Candidates39 := by
  unfold candidates
  have row : edge 2503 = (SortedNatTree.contains FastRows.row2503) := by
    change Data.row2503.contains = (SortedNatTree.contains FastRows.row2503)
    exact FastRows.row2503_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child39 : search edge 3 [7, 2503] root1Candidates39 2510 seed = seed := by decide
theorem root1Step39 : scan edge (search edge 3) 4 [7] 7 root1Tail39 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail40 seed := by
  rw [root1Tail39, scan, if_neg (by decide), root1Candidates39_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail40
    (if 3 ≤ root1Candidates39.length then search edge 3 [7, 2503] root1Candidates39 2510 seed else seed) = _
  rw [if_pos (by decide), root1Child39]
def root1Candidates40 : List Nat := [3181, 4567, 5689, 19333]
set_option maxRecDepth 8192 in
theorem root1Candidates40_checked : candidates edge 4 7 2617 root1Tail41 seed.bound = root1Candidates40 := by
  unfold candidates
  have row : edge 2617 = (SortedNatTree.contains FastRows.row2617) := by
    change Data.row2617.contains = (SortedNatTree.contains FastRows.row2617)
    exact FastRows.row2617_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child40 : search edge 3 [7, 2617] root1Candidates40 2624 seed = seed := by decide
theorem root1Step40 : scan edge (search edge 3) 4 [7] 7 root1Tail40 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail41 seed := by
  rw [root1Tail40, scan, if_neg (by decide), root1Candidates40_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail41
    (if 3 ≤ root1Candidates40.length then search edge 3 [7, 2617] root1Candidates40 2624 seed else seed) = _
  rw [if_pos (by decide), root1Child40]
def root1Candidates41 : List Nat := [3361, 7723, 11287, 11617, 13177, 18457, 19597]
set_option maxRecDepth 8192 in
theorem root1Candidates41_checked : candidates edge 4 7 2671 root1Tail42 seed.bound = root1Candidates41 := by
  unfold candidates
  have row : edge 2671 = (SortedNatTree.contains FastRows.row2671) := by
    change Data.row2671.contains = (SortedNatTree.contains FastRows.row2671)
    exact FastRows.row2671_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child41 : search edge 3 [7, 2671] root1Candidates41 2678 seed = seed := by decide
theorem root1Step41 : scan edge (search edge 3) 4 [7] 7 root1Tail41 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail42 seed := by
  rw [root1Tail41, scan, if_neg (by decide), root1Candidates41_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail42
    (if 3 ≤ root1Candidates41.length then search edge 3 [7, 2671] root1Candidates41 2678 seed else seed) = _
  rw [if_pos (by decide), root1Child41]
def root1Candidates42 : List Nat := [4507, 12157, 14503]
set_option maxRecDepth 8192 in
theorem root1Candidates42_checked : candidates edge 4 7 2707 root1Tail43 seed.bound = root1Candidates42 := by
  unfold candidates
  have row : edge 2707 = (SortedNatTree.contains FastRows.row2707) := by
    change Data.row2707.contains = (SortedNatTree.contains FastRows.row2707)
    exact FastRows.row2707_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child42 : search edge 3 [7, 2707] root1Candidates42 2714 seed = seed := by decide
theorem root1Step42 : scan edge (search edge 3) 4 [7] 7 root1Tail42 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail43 seed := by
  rw [root1Tail42, scan, if_neg (by decide), root1Candidates42_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail43
    (if 3 ≤ root1Candidates42.length then search edge 3 [7, 2707] root1Candidates42 2714 seed else seed) = _
  rw [if_pos (by decide), root1Child42]
def root1Candidates43 : List Nat := [14563]
set_option maxRecDepth 8192 in
theorem root1Candidates43_checked : candidates edge 4 7 2719 root1Tail44 seed.bound = root1Candidates43 := by
  unfold candidates
  have row : edge 2719 = (SortedNatTree.contains FastRows.row2719) := by
    change Data.row2719.contains = (SortedNatTree.contains FastRows.row2719)
    exact FastRows.row2719_checked.symm
  rw [row]
  decide
theorem root1Step43 : scan edge (search edge 3) 4 [7] 7 root1Tail43 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail44 seed := by
  rw [root1Tail43, scan, if_neg (by decide), root1Candidates43_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail44
    (if 3 ≤ root1Candidates43.length then search edge 3 [7, 2719] root1Candidates43 2726 seed else seed) = _
  rw [if_neg (by decide)]
def root1Candidates44 : List Nat := [3181, 3433, 4357, 6679, 10459, 15679]
set_option maxRecDepth 8192 in
theorem root1Candidates44_checked : candidates edge 4 7 2953 root1Tail45 seed.bound = root1Candidates44 := by
  unfold candidates
  have row : edge 2953 = (SortedNatTree.contains FastRows.row2953) := by
    change Data.row2953.contains = (SortedNatTree.contains FastRows.row2953)
    exact FastRows.row2953_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child44 : search edge 3 [7, 2953] root1Candidates44 2960 seed = seed := by decide
theorem root1Step44 : scan edge (search edge 3) 4 [7] 7 root1Tail44 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail45 seed := by
  rw [root1Tail44, scan, if_neg (by decide), root1Candidates44_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail45
    (if 3 ≤ root1Candidates44.length then search edge 3 [7, 2953] root1Candidates44 2960 seed else seed) = _
  rw [if_pos (by decide), root1Child44]
def root1Candidates45 : List Nat := [7351, 7417, 8317, 8941, 9613, 16063]
set_option maxRecDepth 8192 in
theorem root1Candidates45_checked : candidates edge 4 7 3019 root1Tail46 seed.bound = root1Candidates45 := by
  unfold candidates
  have row : edge 3019 = (SortedNatTree.contains FastRows.row3019) := by
    change Data.row3019.contains = (SortedNatTree.contains FastRows.row3019)
    exact FastRows.row3019_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child45 : search edge 3 [7, 3019] root1Candidates45 3026 seed = seed := by decide
theorem root1Step45 : scan edge (search edge 3) 4 [7] 7 root1Tail45 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail46 seed := by
  rw [root1Tail45, scan, if_neg (by decide), root1Candidates45_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail46
    (if 3 ≤ root1Candidates45.length then search edge 3 [7, 3019] root1Candidates45 3026 seed else seed) = _
  rw [if_pos (by decide), root1Child45]
def root1Candidates46 : List Nat := [4219, 4507, 4903, 6481, 9043, 16063, 19267]
set_option maxRecDepth 8192 in
theorem root1Candidates46_checked : candidates edge 4 7 3181 root1Tail47 seed.bound = root1Candidates46 := by
  unfold candidates
  have row : edge 3181 = (SortedNatTree.contains FastRows.row3181) := by
    change Data.row3181.contains = (SortedNatTree.contains FastRows.row3181)
    exact FastRows.row3181_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child46 : search edge 3 [7, 3181] root1Candidates46 3188 seed = seed := by decide
theorem root1Step46 : scan edge (search edge 3) 4 [7] 7 root1Tail46 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail47 seed := by
  rw [root1Tail46, scan, if_neg (by decide), root1Candidates46_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail47
    (if 3 ≤ root1Candidates46.length then search edge 3 [7, 3181] root1Candidates46 3188 seed else seed) = _
  rw [if_pos (by decide), root1Child46]
def root1Candidates47 : List Nat := [4903, 19009, 20857]
set_option maxRecDepth 8192 in
theorem root1Candidates47_checked : candidates edge 4 7 3331 root1Tail48 seed.bound = root1Candidates47 := by
  unfold candidates
  have row : edge 3331 = (SortedNatTree.contains FastRows.row3331) := by
    change Data.row3331.contains = (SortedNatTree.contains FastRows.row3331)
    exact FastRows.row3331_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child47 : search edge 3 [7, 3331] root1Candidates47 3338 seed = seed := by decide
theorem root1Step47 : scan edge (search edge 3) 4 [7] 7 root1Tail47 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail48 seed := by
  rw [root1Tail47, scan, if_neg (by decide), root1Candidates47_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail48
    (if 3 ≤ root1Candidates47.length then search edge 3 [7, 3331] root1Candidates47 3338 seed else seed) = _
  rw [if_pos (by decide), root1Child47]
def root1Candidates48 : List Nat := [4507, 21661]
set_option maxRecDepth 8192 in
theorem root1Candidates48_checked : candidates edge 4 7 3361 root1Tail49 seed.bound = root1Candidates48 := by
  unfold candidates
  have row : edge 3361 = (SortedNatTree.contains FastRows.row3361) := by
    change Data.row3361.contains = (SortedNatTree.contains FastRows.row3361)
    exact FastRows.row3361_checked.symm
  rw [row]
  decide
theorem root1Step48 : scan edge (search edge 3) 4 [7] 7 root1Tail48 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail49 seed := by
  rw [root1Tail48, scan, if_neg (by decide), root1Candidates48_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail49
    (if 3 ≤ root1Candidates48.length then search edge 3 [7, 3361] root1Candidates48 3368 seed else seed) = _
  rw [if_neg (by decide)]
def root1Candidates49 : List Nat := [8167, 16063, 20983]
set_option maxRecDepth 8192 in
theorem root1Candidates49_checked : candidates edge 4 7 3433 root1Tail50 seed.bound = root1Candidates49 := by
  unfold candidates
  have row : edge 3433 = (SortedNatTree.contains FastRows.row3433) := by
    change Data.row3433.contains = (SortedNatTree.contains FastRows.row3433)
    exact FastRows.row3433_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child49 : search edge 3 [7, 3433] root1Candidates49 3440 seed = seed := by decide
theorem root1Step49 : scan edge (search edge 3) 4 [7] 7 root1Tail49 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail50 seed := by
  rw [root1Tail49, scan, if_neg (by decide), root1Candidates49_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail50
    (if 3 ≤ root1Candidates49.length then search edge 3 [7, 3433] root1Candidates49 3440 seed else seed) = _
  rw [if_pos (by decide), root1Child49]
def root1Candidates50 : List Nat := [4177, 5689, 8779, 10837, 14851, 19237, 21661, 22147]
set_option maxRecDepth 8192 in
theorem root1Candidates50_checked : candidates edge 4 7 3583 root1Tail51 seed.bound = root1Candidates50 := by
  unfold candidates
  have row : edge 3583 = (SortedNatTree.contains FastRows.row3583) := by
    change Data.row3583.contains = (SortedNatTree.contains FastRows.row3583)
    exact FastRows.row3583_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child50 : search edge 3 [7, 3583] root1Candidates50 3590 seed = seed := by decide
theorem root1Step50 : scan edge (search edge 3) 4 [7] 7 root1Tail50 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail51 seed := by
  rw [root1Tail50, scan, if_neg (by decide), root1Candidates50_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail51
    (if 3 ≤ root1Candidates50.length then search edge 3 [7, 3583] root1Candidates50 3590 seed else seed) = _
  rw [if_pos (by decide), root1Child50]
def root1Candidates51 : List Nat := [5821, 8803, 9613, 14029, 14851, 15289, 15313, 21739]
set_option maxRecDepth 8192 in
theorem root1Candidates51_checked : candidates edge 4 7 3613 root1Tail52 seed.bound = root1Candidates51 := by
  unfold candidates
  have row : edge 3613 = (SortedNatTree.contains FastRows.row3613) := by
    change Data.row3613.contains = (SortedNatTree.contains FastRows.row3613)
    exact FastRows.row3613_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child51 : search edge 3 [7, 3613] root1Candidates51 3620 seed = seed := by decide
theorem root1Step51 : scan edge (search edge 3) 4 [7] 7 root1Tail51 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail52 seed := by
  rw [root1Tail51, scan, if_neg (by decide), root1Candidates51_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail52
    (if 3 ≤ root1Candidates51.length then search edge 3 [7, 3613] root1Candidates51 3620 seed else seed) = _
  rw [if_pos (by decide), root1Child51]
def root1Candidates52 : List Nat := [8779, 14503, 15679]
set_option maxRecDepth 8192 in
theorem root1Candidates52_checked : candidates edge 4 7 3709 root1Tail53 seed.bound = root1Candidates52 := by
  unfold candidates
  have row : edge 3709 = (SortedNatTree.contains FastRows.row3709) := by
    change Data.row3709.contains = (SortedNatTree.contains FastRows.row3709)
    exact FastRows.row3709_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child52 : search edge 3 [7, 3709] root1Candidates52 3716 seed = seed := by decide
theorem root1Step52 : scan edge (search edge 3) 4 [7] 7 root1Tail52 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail53 seed := by
  rw [root1Tail52, scan, if_neg (by decide), root1Candidates52_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail53
    (if 3 ≤ root1Candidates52.length then search edge 3 [7, 3709] root1Candidates52 3716 seed else seed) = _
  rw [if_pos (by decide), root1Child52]
def root1Candidates53 : List Nat := [5659, 6949, 20899]
set_option maxRecDepth 8192 in
theorem root1Candidates53_checked : candidates edge 4 7 3727 root1Tail54 seed.bound = root1Candidates53 := by
  unfold candidates
  have row : edge 3727 = (SortedNatTree.contains FastRows.row3727) := by
    change Data.row3727.contains = (SortedNatTree.contains FastRows.row3727)
    exact FastRows.row3727_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child53 : search edge 3 [7, 3727] root1Candidates53 3734 seed = seed := by decide
theorem root1Step53 : scan edge (search edge 3) 4 [7] 7 root1Tail53 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail54 seed := by
  rw [root1Tail53, scan, if_neg (by decide), root1Candidates53_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail54
    (if 3 ≤ root1Candidates53.length then search edge 3 [7, 3727] root1Candidates53 3734 seed else seed) = _
  rw [if_pos (by decide), root1Child53]
def root1Candidates54 : List Nat := [4159, 4201, 7417, 11131, 11617, 14563, 15859, 19237, 20899]
set_option maxRecDepth 8192 in
theorem root1Candidates54_checked : candidates edge 4 7 3823 root1Tail55 seed.bound = root1Candidates54 := by
  unfold candidates
  have row : edge 3823 = (SortedNatTree.contains FastRows.row3823) := by
    change Data.row3823.contains = (SortedNatTree.contains FastRows.row3823)
    exact FastRows.row3823_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child54 : search edge 3 [7, 3823] root1Candidates54 3830 seed = seed := by decide
theorem root1Step54 : scan edge (search edge 3) 4 [7] 7 root1Tail54 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail55 seed := by
  rw [root1Tail54, scan, if_neg (by decide), root1Candidates54_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail55
    (if 3 ≤ root1Candidates54.length then search edge 3 [7, 3823] root1Candidates54 3830 seed else seed) = _
  rw [if_pos (by decide), root1Child54]
def root1Candidates55 : List Nat := [6949, 7417, 15727]
set_option maxRecDepth 8192 in
theorem root1Candidates55_checked : candidates edge 4 7 4027 root1Tail56 seed.bound = root1Candidates55 := by
  unfold candidates
  have row : edge 4027 = (SortedNatTree.contains FastRows.row4027) := by
    change Data.row4027.contains = (SortedNatTree.contains FastRows.row4027)
    exact FastRows.row4027_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child55 : search edge 3 [7, 4027] root1Candidates55 4034 seed = seed := by decide
theorem root1Step55 : scan edge (search edge 3) 4 [7] 7 root1Tail55 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail56 seed := by
  rw [root1Tail55, scan, if_neg (by decide), root1Candidates55_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail56
    (if 3 ≤ root1Candidates55.length then search edge 3 [7, 4027] root1Candidates55 4034 seed else seed) = _
  rw [if_pos (by decide), root1Child55]
def root1Candidates56 : List Nat := [18049, 19681]
set_option maxRecDepth 8192 in
theorem root1Candidates56_checked : candidates edge 4 7 4159 root1Tail57 seed.bound = root1Candidates56 := by
  unfold candidates
  have row : edge 4159 = (SortedNatTree.contains FastRows.row4159) := by
    change Data.row4159.contains = (SortedNatTree.contains FastRows.row4159)
    exact FastRows.row4159_checked.symm
  rw [row]
  decide
theorem root1Step56 : scan edge (search edge 3) 4 [7] 7 root1Tail56 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail57 seed := by
  rw [root1Tail56, scan, if_neg (by decide), root1Candidates56_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail57
    (if 3 ≤ root1Candidates56.length then search edge 3 [7, 4159] root1Candidates56 4166 seed else seed) = _
  rw [if_neg (by decide)]
def root1Candidates57 : List Nat := [4903, 8101, 8311]
set_option maxRecDepth 8192 in
theorem root1Candidates57_checked : candidates edge 4 7 4177 root1Tail58 seed.bound = root1Candidates57 := by
  unfold candidates
  have row : edge 4177 = (SortedNatTree.contains FastRows.row4177) := by
    change Data.row4177.contains = (SortedNatTree.contains FastRows.row4177)
    exact FastRows.row4177_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child57 : search edge 3 [7, 4177] root1Candidates57 4184 seed = seed := by decide
theorem root1Step57 : scan edge (search edge 3) 4 [7] 7 root1Tail57 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail58 seed := by
  rw [root1Tail57, scan, if_neg (by decide), root1Candidates57_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail58
    (if 3 ≤ root1Candidates57.length then search edge 3 [7, 4177] root1Candidates57 4184 seed else seed) = _
  rw [if_pos (by decide), root1Child57]
def root1Candidates58 : List Nat := [6421, 11437, 14797, 15877, 19237, 21529]
set_option maxRecDepth 8192 in
theorem root1Candidates58_checked : candidates edge 4 7 4201 root1Tail59 seed.bound = root1Candidates58 := by
  unfold candidates
  have row : edge 4201 = (SortedNatTree.contains FastRows.row4201) := by
    change Data.row4201.contains = (SortedNatTree.contains FastRows.row4201)
    exact FastRows.row4201_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child58 : search edge 3 [7, 4201] root1Candidates58 4208 seed = seed := by decide
theorem root1Step58 : scan edge (search edge 3) 4 [7] 7 root1Tail58 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail59 seed := by
  rw [root1Tail58, scan, if_neg (by decide), root1Candidates58_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail59
    (if 3 ≤ root1Candidates58.length then search edge 3 [7, 4201] root1Candidates58 4208 seed else seed) = _
  rw [if_pos (by decide), root1Child58]
def root1Candidates59 : List Nat := [5521, 14797, 15679]
set_option maxRecDepth 8192 in
theorem root1Candidates59_checked : candidates edge 4 7 4219 root1Tail60 seed.bound = root1Candidates59 := by
  unfold candidates
  have row : edge 4219 = (SortedNatTree.contains FastRows.row4219) := by
    change Data.row4219.contains = (SortedNatTree.contains FastRows.row4219)
    exact FastRows.row4219_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child59 : search edge 3 [7, 4219] root1Candidates59 4226 seed = seed := by decide
theorem root1Step59 : scan edge (search edge 3) 4 [7] 7 root1Tail59 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail60 seed := by
  rw [root1Tail59, scan, if_neg (by decide), root1Candidates59_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail60
    (if 3 ≤ root1Candidates59.length then search edge 3 [7, 4219] root1Candidates59 4226 seed else seed) = _
  rw [if_pos (by decide), root1Child59]
def root1Candidates60 : List Nat := [8893, 12973, 13183, 14551, 18043, 21397]
set_option maxRecDepth 8192 in
theorem root1Candidates60_checked : candidates edge 4 7 4357 root1Tail61 seed.bound = root1Candidates60 := by
  unfold candidates
  have row : edge 4357 = (SortedNatTree.contains FastRows.row4357) := by
    change Data.row4357.contains = (SortedNatTree.contains FastRows.row4357)
    exact FastRows.row4357_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child60 : search edge 3 [7, 4357] root1Candidates60 4364 seed = seed := by decide
theorem root1Step60 : scan edge (search edge 3) 4 [7] 7 root1Tail60 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail61 seed := by
  rw [root1Tail60, scan, if_neg (by decide), root1Candidates60_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail61
    (if 3 ≤ root1Candidates60.length then search edge 3 [7, 4357] root1Candidates60 4364 seed else seed) = _
  rw [if_pos (by decide), root1Child60]
def root1Candidates61 : List Nat := [6481, 6949, 12007]
set_option maxRecDepth 8192 in
theorem root1Candidates61_checked : candidates edge 4 7 4441 root1Tail62 seed.bound = root1Candidates61 := by
  unfold candidates
  have row : edge 4441 = (SortedNatTree.contains FastRows.row4441) := by
    change Data.row4441.contains = (SortedNatTree.contains FastRows.row4441)
    exact FastRows.row4441_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child61 : search edge 3 [7, 4441] root1Candidates61 4448 seed = seed := by decide
theorem root1Step61 : scan edge (search edge 3) 4 [7] 7 root1Tail61 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail62 seed := by
  rw [root1Tail61, scan, if_neg (by decide), root1Candidates61_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail62
    (if 3 ≤ root1Candidates61.length then search edge 3 [7, 4441] root1Candidates61 4448 seed else seed) = _
  rw [if_pos (by decide), root1Child61]
def root1Candidates62 : List Nat := [6091, 8893, 9901, 13807, 21283]
set_option maxRecDepth 8192 in
theorem root1Candidates62_checked : candidates edge 4 7 4507 root1Tail63 seed.bound = root1Candidates62 := by
  unfold candidates
  have row : edge 4507 = (SortedNatTree.contains FastRows.row4507) := by
    change Data.row4507.contains = (SortedNatTree.contains FastRows.row4507)
    exact FastRows.row4507_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child62 : search edge 3 [7, 4507] root1Candidates62 4514 seed = seed := by decide
theorem root1Step62 : scan edge (search edge 3) 4 [7] 7 root1Tail62 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail63 seed := by
  rw [root1Tail62, scan, if_neg (by decide), root1Candidates62_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail63
    (if 3 ≤ root1Candidates62.length then search edge 3 [7, 4507] root1Candidates62 4514 seed else seed) = _
  rw [if_pos (by decide), root1Child62]
def root1Candidates63 : List Nat := [7417, 10399, 15859, 18433]
set_option maxRecDepth 8192 in
theorem root1Candidates63_checked : candidates edge 4 7 4567 root1Tail64 seed.bound = root1Candidates63 := by
  unfold candidates
  have row : edge 4567 = (SortedNatTree.contains FastRows.row4567) := by
    change Data.row4567.contains = (SortedNatTree.contains FastRows.row4567)
    exact FastRows.row4567_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child63 : search edge 3 [7, 4567] root1Candidates63 4574 seed = seed := by decide
theorem root1Step63 : scan edge (search edge 3) 4 [7] 7 root1Tail63 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail64 seed := by
  rw [root1Tail63, scan, if_neg (by decide), root1Candidates63_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail64
    (if 3 ≤ root1Candidates63.length then search edge 3 [7, 4567] root1Candidates63 4574 seed else seed) = _
  rw [if_pos (by decide), root1Child63]
end ProjectEulerSolutions.PrimePairLarge.Search
