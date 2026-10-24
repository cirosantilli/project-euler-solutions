import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot002Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root2Candidates32 : List Nat := [11057, 13103, 21557]
set_option maxRecDepth 8192 in
theorem root2Candidates32_checked : candidates edge 4 11 2927 root2Tail33 seed.bound = root2Candidates32 := by
  unfold candidates
  have row : edge 2927 = (SortedNatTree.contains FastRows.row2927) := by
    change Data.row2927.contains = (SortedNatTree.contains FastRows.row2927)
    exact FastRows.row2927_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root2Child32 : search edge 3 [11, 2927] root2Candidates32 2938 seed = seed := by decide
theorem root2Step32 : scan edge (search edge 3) 4 [11] 11 root2Tail32 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail33 seed := by
  rw [root2Tail32, scan, if_neg (by decide), root2Candidates32_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail33
    (if 3 ≤ root2Candidates32.length then search edge 3 [11, 2927] root2Candidates32 2938 seed else seed) = _
  rw [if_pos (by decide), root2Child32]
def root2Candidates33 : List Nat := [8747]
set_option maxRecDepth 8192 in
theorem root2Candidates33_checked : candidates edge 4 11 3083 root2Tail34 seed.bound = root2Candidates33 := by
  unfold candidates
  have row : edge 3083 = (SortedNatTree.contains FastRows.row3083) := by
    change Data.row3083.contains = (SortedNatTree.contains FastRows.row3083)
    exact FastRows.row3083_checked.symm
  rw [row]
  decide
theorem root2Step33 : scan edge (search edge 3) 4 [11] 11 root2Tail33 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail34 seed := by
  rw [root2Tail33, scan, if_neg (by decide), root2Candidates33_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail34
    (if 3 ≤ root2Candidates33.length then search edge 3 [11, 3083] root2Candidates33 3094 seed else seed) = _
  rw [if_neg (by decide)]
def root2Candidates34 : List Nat := [9293, 21011, 22739]
set_option maxRecDepth 8192 in
theorem root2Candidates34_checked : candidates edge 4 11 3209 root2Tail35 seed.bound = root2Candidates34 := by
  unfold candidates
  have row : edge 3209 = (SortedNatTree.contains FastRows.row3209) := by
    change Data.row3209.contains = (SortedNatTree.contains FastRows.row3209)
    exact FastRows.row3209_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root2Child34 : search edge 3 [11, 3209] root2Candidates34 3220 seed = seed := by decide
theorem root2Step34 : scan edge (search edge 3) 4 [11] 11 root2Tail34 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail35 seed := by
  rw [root2Tail34, scan, if_neg (by decide), root2Candidates34_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail35
    (if 3 ≤ root2Candidates34.length then search edge 3 [11, 3209] root2Candidates34 3220 seed else seed) = _
  rw [if_pos (by decide), root2Child34]
def root2Candidates35 : List Nat := [5807]
set_option maxRecDepth 8192 in
theorem root2Candidates35_checked : candidates edge 4 11 3467 root2Tail36 seed.bound = root2Candidates35 := by
  unfold candidates
  have row : edge 3467 = (SortedNatTree.contains FastRows.row3467) := by
    change Data.row3467.contains = (SortedNatTree.contains FastRows.row3467)
    exact FastRows.row3467_checked.symm
  rw [row]
  decide
theorem root2Step35 : scan edge (search edge 3) 4 [11] 11 root2Tail35 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail36 seed := by
  rw [root2Tail35, scan, if_neg (by decide), root2Candidates35_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail36
    (if 3 ≤ root2Candidates35.length then search edge 3 [11, 3467] root2Candidates35 3478 seed else seed) = _
  rw [if_neg (by decide)]
def root2Candidates36 : List Nat := [4013, 5861, 21023]
set_option maxRecDepth 8192 in
theorem root2Candidates36_checked : candidates edge 4 11 3539 root2Tail37 seed.bound = root2Candidates36 := by
  unfold candidates
  have row : edge 3539 = (SortedNatTree.contains FastRows.row3539) := by
    change Data.row3539.contains = (SortedNatTree.contains FastRows.row3539)
    exact FastRows.row3539_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root2Child36 : search edge 3 [11, 3539] root2Candidates36 3550 seed = seed := by decide
theorem root2Step36 : scan edge (search edge 3) 4 [11] 11 root2Tail36 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail37 seed := by
  rw [root2Tail36, scan, if_neg (by decide), root2Candidates36_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail37
    (if 3 ≤ root2Candidates36.length then search edge 3 [11, 3539] root2Candidates36 3550 seed else seed) = _
  rw [if_pos (by decide), root2Child36]
def root2Candidates37 : List Nat := [5099, 6329, 7577, 12911, 17489]
set_option maxRecDepth 8192 in
theorem root2Candidates37_checked : candidates edge 4 11 3779 root2Tail38 seed.bound = root2Candidates37 := by
  unfold candidates
  have row : edge 3779 = (SortedNatTree.contains FastRows.row3779) := by
    change Data.row3779.contains = (SortedNatTree.contains FastRows.row3779)
    exact FastRows.row3779_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root2Child37 : search edge 3 [11, 3779] root2Candidates37 3790 seed = seed := by decide
theorem root2Step37 : scan edge (search edge 3) 4 [11] 11 root2Tail37 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail38 seed := by
  rw [root2Tail37, scan, if_neg (by decide), root2Candidates37_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail38
    (if 3 ≤ root2Candidates37.length then search edge 3 [11, 3779] root2Candidates37 3790 seed else seed) = _
  rw [if_pos (by decide), root2Child37]
def root2Candidates38 : List Nat := [4643, 5153, 7541, 8387]
set_option maxRecDepth 8192 in
theorem root2Candidates38_checked : candidates edge 4 11 4013 root2Tail39 seed.bound = root2Candidates38 := by
  unfold candidates
  have row : edge 4013 = (SortedNatTree.contains FastRows.row4013) := by
    change Data.row4013.contains = (SortedNatTree.contains FastRows.row4013)
    exact FastRows.row4013_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root2Child38 : search edge 3 [11, 4013] root2Candidates38 4024 seed = seed := by decide
theorem root2Step38 : scan edge (search edge 3) 4 [11] 11 root2Tail38 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail39 seed := by
  rw [root2Tail38, scan, if_neg (by decide), root2Candidates38_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail39
    (if 3 ≤ root2Candidates38.length then search edge 3 [11, 4013] root2Candidates38 4024 seed else seed) = _
  rw [if_pos (by decide), root2Child38]
def root2Candidates39 : List Nat := [10301]
set_option maxRecDepth 8192 in
theorem root2Candidates39_checked : candidates edge 4 11 4073 root2Tail40 seed.bound = root2Candidates39 := by
  unfold candidates
  have row : edge 4073 = (SortedNatTree.contains FastRows.row4073) := by
    change Data.row4073.contains = (SortedNatTree.contains FastRows.row4073)
    exact FastRows.row4073_checked.symm
  rw [row]
  decide
theorem root2Step39 : scan edge (search edge 3) 4 [11] 11 root2Tail39 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail40 seed := by
  rw [root2Tail39, scan, if_neg (by decide), root2Candidates39_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail40
    (if 3 ≤ root2Candidates39.length then search edge 3 [11, 4073] root2Candidates39 4084 seed else seed) = _
  rw [if_neg (by decide)]
def root2Candidates40 : List Nat := [15641]
set_option maxRecDepth 8192 in
theorem root2Candidates40_checked : candidates edge 4 11 4217 root2Tail41 seed.bound = root2Candidates40 := by
  unfold candidates
  have row : edge 4217 = (SortedNatTree.contains FastRows.row4217) := by
    change Data.row4217.contains = (SortedNatTree.contains FastRows.row4217)
    exact FastRows.row4217_checked.symm
  rw [row]
  decide
theorem root2Step40 : scan edge (search edge 3) 4 [11] 11 root2Tail40 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail41 seed := by
  rw [root2Tail40, scan, if_neg (by decide), root2Candidates40_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail41
    (if 3 ≤ root2Candidates40.length then search edge 3 [11, 4217] root2Candidates40 4228 seed else seed) = _
  rw [if_neg (by decide)]
def root2Candidates41 : List Nat := [8831, 11399]
set_option maxRecDepth 8192 in
theorem root2Candidates41_checked : candidates edge 4 11 4229 root2Tail42 seed.bound = root2Candidates41 := by
  unfold candidates
  have row : edge 4229 = (SortedNatTree.contains FastRows.row4229) := by
    change Data.row4229.contains = (SortedNatTree.contains FastRows.row4229)
    exact FastRows.row4229_checked.symm
  rw [row]
  decide
theorem root2Step41 : scan edge (search edge 3) 4 [11] 11 root2Tail41 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail42 seed := by
  rw [root2Tail41, scan, if_neg (by decide), root2Candidates41_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail42
    (if 3 ≤ root2Candidates41.length then search edge 3 [11, 4229] root2Candidates41 4240 seed else seed) = _
  rw [if_neg (by decide)]
def root2Candidates42 : List Nat := [5807, 11399]
set_option maxRecDepth 8192 in
theorem root2Candidates42_checked : candidates edge 4 11 4259 root2Tail43 seed.bound = root2Candidates42 := by
  unfold candidates
  have row : edge 4259 = (SortedNatTree.contains FastRows.row4259) := by
    change Data.row4259.contains = (SortedNatTree.contains FastRows.row4259)
    exact FastRows.row4259_checked.symm
  rw [row]
  decide
theorem root2Step42 : scan edge (search edge 3) 4 [11] 11 root2Tail42 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail43 seed := by
  rw [root2Tail42, scan, if_neg (by decide), root2Candidates42_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail43
    (if 3 ≤ root2Candidates42.length then search edge 3 [11, 4259] root2Candidates42 4270 seed else seed) = _
  rw [if_neg (by decide)]
def root2Candidates43 : List Nat := [15641, 17027]
set_option maxRecDepth 8192 in
theorem root2Candidates43_checked : candidates edge 4 11 4493 root2Tail44 seed.bound = root2Candidates43 := by
  unfold candidates
  have row : edge 4493 = (SortedNatTree.contains FastRows.row4493) := by
    change Data.row4493.contains = (SortedNatTree.contains FastRows.row4493)
    exact FastRows.row4493_checked.symm
  rw [row]
  decide
theorem root2Step43 : scan edge (search edge 3) 4 [11] 11 root2Tail43 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail44 seed := by
  rw [root2Tail43, scan, if_neg (by decide), root2Candidates43_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail44
    (if 3 ≤ root2Candidates43.length then search edge 3 [11, 4493] root2Candidates43 4504 seed else seed) = _
  rw [if_neg (by decide)]
def root2Candidates44 : List Nat := [6791, 8831, 15227, 16319]
set_option maxRecDepth 8192 in
theorem root2Candidates44_checked : candidates edge 4 11 4547 root2Tail45 seed.bound = root2Candidates44 := by
  unfold candidates
  have row : edge 4547 = (SortedNatTree.contains FastRows.row4547) := by
    change Data.row4547.contains = (SortedNatTree.contains FastRows.row4547)
    exact FastRows.row4547_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root2Child44 : search edge 3 [11, 4547] root2Candidates44 4558 seed = seed := by decide
theorem root2Step44 : scan edge (search edge 3) 4 [11] 11 root2Tail44 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail45 seed := by
  rw [root2Tail44, scan, if_neg (by decide), root2Candidates44_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail45
    (if 3 ≤ root2Candidates44.length then search edge 3 [11, 4547] root2Candidates44 4558 seed else seed) = _
  rw [if_pos (by decide), root2Child44]
def root2Candidates45 : List Nat := [5153, 5849, 7757]
set_option maxRecDepth 8192 in
theorem root2Candidates45_checked : candidates edge 4 11 4643 root2Tail46 seed.bound = root2Candidates45 := by
  unfold candidates
  have row : edge 4643 = (SortedNatTree.contains FastRows.row4643) := by
    change Data.row4643.contains = (SortedNatTree.contains FastRows.row4643)
    exact FastRows.row4643_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root2Child45 : search edge 3 [11, 4643] root2Candidates45 4654 seed = seed := by decide
theorem root2Step45 : scan edge (search edge 3) 4 [11] 11 root2Tail45 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail46 seed := by
  rw [root2Tail45, scan, if_neg (by decide), root2Candidates45_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail46
    (if 3 ≤ root2Candidates45.length then search edge 3 [11, 4643] root2Candidates45 4654 seed else seed) = _
  rw [if_pos (by decide), root2Child45]
def root2Candidates46 : List Nat := [5849, 7127, 7877, 8219, 8747, 11057, 12911]
set_option maxRecDepth 8192 in
theorem root2Candidates46_checked : candidates edge 4 11 4967 root2Tail47 seed.bound = root2Candidates46 := by
  unfold candidates
  have row : edge 4967 = (SortedNatTree.contains FastRows.row4967) := by
    change Data.row4967.contains = (SortedNatTree.contains FastRows.row4967)
    exact FastRows.row4967_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root2Child46 : search edge 3 [11, 4967] root2Candidates46 4978 seed = seed := by decide
theorem root2Step46 : scan edge (search edge 3) 4 [11] 11 root2Tail46 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail47 seed := by
  rw [root2Tail46, scan, if_neg (by decide), root2Candidates46_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail47
    (if 3 ≤ root2Candidates46.length then search edge 3 [11, 4967] root2Candidates46 4978 seed else seed) = _
  rw [if_pos (by decide), root2Child46]
def root2Candidates47 : List Nat := [8273]
set_option maxRecDepth 8192 in
theorem root2Candidates47_checked : candidates edge 4 11 5099 root2Tail48 seed.bound = root2Candidates47 := by
  unfold candidates
  have row : edge 5099 = (SortedNatTree.contains FastRows.row5099) := by
    change Data.row5099.contains = (SortedNatTree.contains FastRows.row5099)
    exact FastRows.row5099_checked.symm
  rw [row]
  decide
theorem root2Step47 : scan edge (search edge 3) 4 [11] 11 root2Tail47 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail48 seed := by
  rw [root2Tail47, scan, if_neg (by decide), root2Candidates47_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail48
    (if 3 ≤ root2Candidates47.length then search edge 3 [11, 5099] root2Candidates47 5110 seed else seed) = _
  rw [if_neg (by decide)]
def root2Candidates48 : List Nat := [5849, 6047, 7529, 8819, 11087]
set_option maxRecDepth 8192 in
theorem root2Candidates48_checked : candidates edge 4 11 5153 root2Tail49 seed.bound = root2Candidates48 := by
  unfold candidates
  have row : edge 5153 = (SortedNatTree.contains FastRows.row5153) := by
    change Data.row5153.contains = (SortedNatTree.contains FastRows.row5153)
    exact FastRows.row5153_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root2Child48 : search edge 3 [11, 5153] root2Candidates48 5164 seed = seed := by decide
theorem root2Step48 : scan edge (search edge 3) 4 [11] 11 root2Tail48 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail49 seed := by
  rw [root2Tail48, scan, if_neg (by decide), root2Candidates48_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail49
    (if 3 ≤ root2Candidates48.length then search edge 3 [11, 5153] root2Candidates48 5164 seed else seed) = _
  rw [if_pos (by decide), root2Child48]
def root2Candidates49 : List Nat := [15551]
set_option maxRecDepth 8192 in
theorem root2Candidates49_checked : candidates edge 4 11 5309 root2Tail50 seed.bound = root2Candidates49 := by
  unfold candidates
  have row : edge 5309 = (SortedNatTree.contains FastRows.row5309) := by
    change Data.row5309.contains = (SortedNatTree.contains FastRows.row5309)
    exact FastRows.row5309_checked.symm
  rw [row]
  decide
theorem root2Step49 : scan edge (search edge 3) 4 [11] 11 root2Tail49 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail50 seed := by
  rw [root2Tail49, scan, if_neg (by decide), root2Candidates49_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail50
    (if 3 ≤ root2Candidates49.length then search edge 3 [11, 5309] root2Candidates49 5320 seed else seed) = _
  rw [if_neg (by decide)]
def root2Candidates50 : List Nat := [6719, 7331, 9689, 11087, 15131]
set_option maxRecDepth 8192 in
theorem root2Candidates50_checked : candidates edge 4 11 5783 root2Tail51 seed.bound = root2Candidates50 := by
  unfold candidates
  have row : edge 5783 = (SortedNatTree.contains FastRows.row5783) := by
    change Data.row5783.contains = (SortedNatTree.contains FastRows.row5783)
    exact FastRows.row5783_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root2Child50 : search edge 3 [11, 5783] root2Candidates50 5794 seed = seed := by decide
theorem root2Step50 : scan edge (search edge 3) 4 [11] 11 root2Tail50 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail51 seed := by
  rw [root2Tail50, scan, if_neg (by decide), root2Candidates50_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail51
    (if 3 ≤ root2Candidates50.length then search edge 3 [11, 5783] root2Candidates50 5794 seed else seed) = _
  rw [if_pos (by decide), root2Child50]
def root2Candidates51 : List Nat := [6329, 7577, 9689]
set_option maxRecDepth 8192 in
theorem root2Candidates51_checked : candidates edge 4 11 5807 root2Tail52 seed.bound = root2Candidates51 := by
  unfold candidates
  have row : edge 5807 = (SortedNatTree.contains FastRows.row5807) := by
    change Data.row5807.contains = (SortedNatTree.contains FastRows.row5807)
    exact FastRows.row5807_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root2Child51 : search edge 3 [11, 5807] root2Candidates51 5818 seed = seed := by decide
theorem root2Step51 : scan edge (search edge 3) 4 [11] 11 root2Tail51 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail52 seed := by
  rw [root2Tail51, scan, if_neg (by decide), root2Candidates51_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail52
    (if 3 ≤ root2Candidates51.length then search edge 3 [11, 5807] root2Candidates51 5818 seed else seed) = _
  rw [if_pos (by decide), root2Child51]
def root2Candidates52 : List Nat := [8681, 11351]
set_option maxRecDepth 8192 in
theorem root2Candidates52_checked : candidates edge 4 11 5849 root2Tail53 seed.bound = root2Candidates52 := by
  unfold candidates
  have row : edge 5849 = (SortedNatTree.contains FastRows.row5849) := by
    change Data.row5849.contains = (SortedNatTree.contains FastRows.row5849)
    exact FastRows.row5849_checked.symm
  rw [row]
  decide
theorem root2Step52 : scan edge (search edge 3) 4 [11] 11 root2Tail52 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail53 seed := by
  rw [root2Tail52, scan, if_neg (by decide), root2Candidates52_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail53
    (if 3 ≤ root2Candidates52.length then search edge 3 [11, 5849] root2Candidates52 5860 seed else seed) = _
  rw [if_neg (by decide)]
def root2Candidates53 : List Nat := [5987, 8171]
set_option maxRecDepth 8192 in
theorem root2Candidates53_checked : candidates edge 4 11 5861 root2Tail54 seed.bound = root2Candidates53 := by
  unfold candidates
  have row : edge 5861 = (SortedNatTree.contains FastRows.row5861) := by
    change Data.row5861.contains = (SortedNatTree.contains FastRows.row5861)
    exact FastRows.row5861_checked.symm
  rw [row]
  decide
theorem root2Step53 : scan edge (search edge 3) 4 [11] 11 root2Tail53 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail54 seed := by
  rw [root2Tail53, scan, if_neg (by decide), root2Candidates53_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail54
    (if 3 ≤ root2Candidates53.length then search edge 3 [11, 5861] root2Candidates53 5872 seed else seed) = _
  rw [if_neg (by decide)]
def root2Candidates54 : List Nat := [8831]
set_option maxRecDepth 8192 in
theorem root2Candidates54_checked : candidates edge 4 11 5987 root2Tail55 seed.bound = root2Candidates54 := by
  unfold candidates
  have row : edge 5987 = (SortedNatTree.contains FastRows.row5987) := by
    change Data.row5987.contains = (SortedNatTree.contains FastRows.row5987)
    exact FastRows.row5987_checked.symm
  rw [row]
  decide
theorem root2Step54 : scan edge (search edge 3) 4 [11] 11 root2Tail54 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail55 seed := by
  rw [root2Tail54, scan, if_neg (by decide), root2Candidates54_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail55
    (if 3 ≤ root2Candidates54.length then search edge 3 [11, 5987] root2Candidates54 5998 seed else seed) = _
  rw [if_neg (by decide)]
def root2Candidates55 : List Nat := [6719]
set_option maxRecDepth 8192 in
theorem root2Candidates55_checked : candidates edge 4 11 6047 root2Tail56 seed.bound = root2Candidates55 := by
  unfold candidates
  have row : edge 6047 = (SortedNatTree.contains FastRows.row6047) := by
    change Data.row6047.contains = (SortedNatTree.contains FastRows.row6047)
    exact FastRows.row6047_checked.symm
  rw [row]
  decide
theorem root2Step55 : scan edge (search edge 3) 4 [11] 11 root2Tail55 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail56 seed := by
  rw [root2Tail55, scan, if_neg (by decide), root2Candidates55_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail56
    (if 3 ≤ root2Candidates55.length then search edge 3 [11, 6047] root2Candidates55 6058 seed else seed) = _
  rw [if_neg (by decide)]
def root2Candidates56 : List Nat := [7331, 7937, 16943]
set_option maxRecDepth 8192 in
theorem root2Candidates56_checked : candidates edge 4 11 6329 root2Tail57 seed.bound = root2Candidates56 := by
  unfold candidates
  have row : edge 6329 = (SortedNatTree.contains FastRows.row6329) := by
    change Data.row6329.contains = (SortedNatTree.contains FastRows.row6329)
    exact FastRows.row6329_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root2Child56 : search edge 3 [11, 6329] root2Candidates56 6340 seed = seed := by decide
theorem root2Step56 : scan edge (search edge 3) 4 [11] 11 root2Tail56 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail57 seed := by
  rw [root2Tail56, scan, if_neg (by decide), root2Candidates56_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail57
    (if 3 ≤ root2Candidates56.length then search edge 3 [11, 6329] root2Candidates56 6340 seed else seed) = _
  rw [if_pos (by decide), root2Child56]
end ProjectEulerSolutions.PrimePairLarge.Search
