import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot012Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root12Candidates32 : List Nat := [4211, 6803, 9041, 9377, 10313]
set_option maxRecDepth 8192 in
theorem root12Candidates32_checked : candidates edge 4 47 3923 root12Tail33 winner.bound = root12Candidates32 := by
  unfold candidates
  have row : edge 3923 = (SortedNatTree.contains FastRows.row3923) := by
    change Data.row3923.contains = (SortedNatTree.contains FastRows.row3923)
    exact FastRows.row3923_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root12Child32 : search edge 3 [47, 3923] root12Candidates32 3970 winner = winner := by decide
theorem root12Step32 : scan edge (search edge 3) 4 [47] 47 root12Tail32 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail33 winner := by
  rw [root12Tail32, scan, if_neg (by decide), root12Candidates32_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail33
    (if 3 ≤ root12Candidates32.length then search edge 3 [47, 3923] root12Candidates32 3970 winner else winner) = _
  rw [if_pos (by decide), root12Child32]
def root12Candidates33 : List Nat := [8573]
set_option maxRecDepth 8192 in
theorem root12Candidates33_checked : candidates edge 4 47 4073 root12Tail34 winner.bound = root12Candidates33 := by
  unfold candidates
  have row : edge 4073 = (SortedNatTree.contains FastRows.row4073) := by
    change Data.row4073.contains = (SortedNatTree.contains FastRows.row4073)
    exact FastRows.row4073_checked.symm
  rw [row]
  decide
theorem root12Step33 : scan edge (search edge 3) 4 [47] 47 root12Tail33 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail34 winner := by
  rw [root12Tail33, scan, if_neg (by decide), root12Candidates33_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail34
    (if 3 ≤ root12Candidates33.length then search edge 3 [47, 4073] root12Candidates33 4120 winner else winner) = _
  rw [if_neg (by decide)]
def root12Candidates34 : List Nat := [5483, 6911, 8963, 18671, 19289, 19763, 21737]
set_option maxRecDepth 8192 in
theorem root12Candidates34_checked : candidates edge 4 47 4211 root12Tail35 winner.bound = root12Candidates34 := by
  unfold candidates
  have row : edge 4211 = (SortedNatTree.contains FastRows.row4211) := by
    change Data.row4211.contains = (SortedNatTree.contains FastRows.row4211)
    exact FastRows.row4211_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root12Child34 : search edge 3 [47, 4211] root12Candidates34 4258 winner = winner := by decide
theorem root12Step34 : scan edge (search edge 3) 4 [47] 47 root12Tail34 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail35 winner := by
  rw [root12Tail34, scan, if_neg (by decide), root12Candidates34_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail35
    (if 3 ≤ root12Candidates34.length then search edge 3 [47, 4211] root12Candidates34 4258 winner else winner) = _
  rw [if_pos (by decide), root12Child34]
def root12Candidates35 : List Nat := [4751]
set_option maxRecDepth 8192 in
theorem root12Candidates35_checked : candidates edge 4 47 4241 root12Tail36 winner.bound = root12Candidates35 := by
  unfold candidates
  have row : edge 4241 = (SortedNatTree.contains FastRows.row4241) := by
    change Data.row4241.contains = (SortedNatTree.contains FastRows.row4241)
    exact FastRows.row4241_checked.symm
  rw [row]
  decide
theorem root12Step35 : scan edge (search edge 3) 4 [47] 47 root12Tail35 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail36 winner := by
  rw [root12Tail35, scan, if_neg (by decide), root12Candidates35_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail36
    (if 3 ≤ root12Candidates35.length then search edge 3 [47, 4241] root12Candidates35 4288 winner else winner) = _
  rw [if_neg (by decide)]
def root12Candidates36 : List Nat := []
set_option maxRecDepth 8192 in
theorem root12Candidates36_checked : candidates edge 4 47 4337 root12Tail37 winner.bound = root12Candidates36 := by
  unfold candidates
  have row : edge 4337 = (SortedNatTree.contains FastRows.row4337) := by
    change Data.row4337.contains = (SortedNatTree.contains FastRows.row4337)
    exact FastRows.row4337_checked.symm
  rw [row]
  decide
theorem root12Step36 : scan edge (search edge 3) 4 [47] 47 root12Tail36 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail37 winner := by
  rw [root12Tail36, scan, if_neg (by decide), root12Candidates36_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail37
    (if 3 ≤ root12Candidates36.length then search edge 3 [47, 4337] root12Candidates36 4384 winner else winner) = _
  rw [if_neg (by decide)]
def root12Candidates37 : List Nat := [5297, 5879, 6659, 8963, 17519]
set_option maxRecDepth 8192 in
theorem root12Candidates37_checked : candidates edge 4 47 4751 root12Tail38 winner.bound = root12Candidates37 := by
  unfold candidates
  have row : edge 4751 = (SortedNatTree.contains FastRows.row4751) := by
    change Data.row4751.contains = (SortedNatTree.contains FastRows.row4751)
    exact FastRows.row4751_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root12Child37 : search edge 3 [47, 4751] root12Candidates37 4798 winner = winner := by decide
theorem root12Step37 : scan edge (search edge 3) 4 [47] 47 root12Tail37 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail38 winner := by
  rw [root12Tail37, scan, if_neg (by decide), root12Candidates37_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail38
    (if 3 ≤ root12Candidates37.length then search edge 3 [47, 4751] root12Candidates37 4798 winner else winner) = _
  rw [if_pos (by decide), root12Child37]
def root12Candidates38 : List Nat := [8573, 9371, 19289, 19751]
set_option maxRecDepth 8192 in
theorem root12Candidates38_checked : candidates edge 4 47 4787 root12Tail39 winner.bound = root12Candidates38 := by
  unfold candidates
  have row : edge 4787 = (SortedNatTree.contains FastRows.row4787) := by
    change Data.row4787.contains = (SortedNatTree.contains FastRows.row4787)
    exact FastRows.row4787_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root12Child38 : search edge 3 [47, 4787] root12Candidates38 4834 winner = winner := by decide
theorem root12Step38 : scan edge (search edge 3) 4 [47] 47 root12Tail38 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail39 winner := by
  rw [root12Tail38, scan, if_neg (by decide), root12Candidates38_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail39
    (if 3 ≤ root12Candidates38.length then search edge 3 [47, 4787] root12Candidates38 4834 winner else winner) = _
  rw [if_pos (by decide), root12Child38]
def root12Candidates39 : List Nat := [9833, 10253, 19031]
set_option maxRecDepth 8192 in
theorem root12Candidates39_checked : candidates edge 4 47 4931 root12Tail40 winner.bound = root12Candidates39 := by
  unfold candidates
  have row : edge 4931 = (SortedNatTree.contains FastRows.row4931) := by
    change Data.row4931.contains = (SortedNatTree.contains FastRows.row4931)
    exact FastRows.row4931_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root12Child39 : search edge 3 [47, 4931] root12Candidates39 4978 winner = winner := by decide
theorem root12Step39 : scan edge (search edge 3) 4 [47] 47 root12Tail39 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail40 winner := by
  rw [root12Tail39, scan, if_neg (by decide), root12Candidates39_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail40
    (if 3 ≤ root12Candidates39.length then search edge 3 [47, 4931] root12Candidates39 4978 winner else winner) = _
  rw [if_pos (by decide), root12Child39]
def root12Candidates40 : List Nat := [8573, 9419]
set_option maxRecDepth 8192 in
theorem root12Candidates40_checked : candidates edge 4 47 4937 root12Tail41 winner.bound = root12Candidates40 := by
  unfold candidates
  have row : edge 4937 = (SortedNatTree.contains FastRows.row4937) := by
    change Data.row4937.contains = (SortedNatTree.contains FastRows.row4937)
    exact FastRows.row4937_checked.symm
  rw [row]
  decide
theorem root12Step40 : scan edge (search edge 3) 4 [47] 47 root12Tail40 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail41 winner := by
  rw [root12Tail40, scan, if_neg (by decide), root12Candidates40_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail41
    (if 3 ≤ root12Candidates40.length then search edge 3 [47, 4937] root12Candidates40 4984 winner else winner) = _
  rw [if_neg (by decide)]
def root12Candidates41 : List Nat := [5273, 5297]
set_option maxRecDepth 8192 in
theorem root12Candidates41_checked : candidates edge 4 47 5147 root12Tail42 winner.bound = root12Candidates41 := by
  unfold candidates
  have row : edge 5147 = (SortedNatTree.contains FastRows.row5147) := by
    change Data.row5147.contains = (SortedNatTree.contains FastRows.row5147)
    exact FastRows.row5147_checked.symm
  rw [row]
  decide
theorem root12Step41 : scan edge (search edge 3) 4 [47] 47 root12Tail41 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail42 winner := by
  rw [root12Tail41, scan, if_neg (by decide), root12Candidates41_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail42
    (if 3 ≤ root12Candidates41.length then search edge 3 [47, 5147] root12Candidates41 5194 winner else winner) = _
  rw [if_neg (by decide)]
def root12Candidates42 : List Nat := [5381]
set_option maxRecDepth 8192 in
theorem root12Candidates42_checked : candidates edge 4 47 5273 root12Tail43 winner.bound = root12Candidates42 := by
  unfold candidates
  have row : edge 5273 = (SortedNatTree.contains FastRows.row5273) := by
    change Data.row5273.contains = (SortedNatTree.contains FastRows.row5273)
    exact FastRows.row5273_checked.symm
  rw [row]
  decide
theorem root12Step42 : scan edge (search edge 3) 4 [47] 47 root12Tail42 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail43 winner := by
  rw [root12Tail42, scan, if_neg (by decide), root12Candidates42_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail43
    (if 3 ≤ root12Candidates42.length then search edge 3 [47, 5273] root12Candidates42 5320 winner else winner) = _
  rw [if_neg (by decide)]
def root12Candidates43 : List Nat := [6701]
set_option maxRecDepth 8192 in
theorem root12Candidates43_checked : candidates edge 4 47 5297 root12Tail44 winner.bound = root12Candidates43 := by
  unfold candidates
  have row : edge 5297 = (SortedNatTree.contains FastRows.row5297) := by
    change Data.row5297.contains = (SortedNatTree.contains FastRows.row5297)
    exact FastRows.row5297_checked.symm
  rw [row]
  decide
theorem root12Step43 : scan edge (search edge 3) 4 [47] 47 root12Tail43 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail44 winner := by
  rw [root12Tail43, scan, if_neg (by decide), root12Candidates43_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail44
    (if 3 ≤ root12Candidates43.length then search edge 3 [47, 5297] root12Candidates43 5344 winner else winner) = _
  rw [if_neg (by decide)]
def root12Candidates44 : List Nat := [6599]
set_option maxRecDepth 8192 in
theorem root12Candidates44_checked : candidates edge 4 47 5381 root12Tail45 winner.bound = root12Candidates44 := by
  unfold candidates
  have row : edge 5381 = (SortedNatTree.contains FastRows.row5381) := by
    change Data.row5381.contains = (SortedNatTree.contains FastRows.row5381)
    exact FastRows.row5381_checked.symm
  rw [row]
  decide
theorem root12Step44 : scan edge (search edge 3) 4 [47] 47 root12Tail44 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail45 winner := by
  rw [root12Tail44, scan, if_neg (by decide), root12Candidates44_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail45
    (if 3 ≤ root12Candidates44.length then search edge 3 [47, 5381] root12Candidates44 5428 winner else winner) = _
  rw [if_neg (by decide)]
def root12Candidates45 : List Nat := [6029, 10463]
set_option maxRecDepth 8192 in
theorem root12Candidates45_checked : candidates edge 4 47 5483 root12Tail46 winner.bound = root12Candidates45 := by
  unfold candidates
  have row : edge 5483 = (SortedNatTree.contains FastRows.row5483) := by
    change Data.row5483.contains = (SortedNatTree.contains FastRows.row5483)
    exact FastRows.row5483_checked.symm
  rw [row]
  decide
theorem root12Step45 : scan edge (search edge 3) 4 [47] 47 root12Tail45 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail46 winner := by
  rw [root12Tail45, scan, if_neg (by decide), root12Candidates45_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail46
    (if 3 ≤ root12Candidates45.length then search edge 3 [47, 5483] root12Candidates45 5530 winner else winner) = _
  rw [if_neg (by decide)]
def root12Candidates46 : List Nat := [6737, 8963, 9377]
set_option maxRecDepth 8192 in
theorem root12Candidates46_checked : candidates edge 4 47 5639 root12Tail47 winner.bound = root12Candidates46 := by
  unfold candidates
  have row : edge 5639 = (SortedNatTree.contains FastRows.row5639) := by
    change Data.row5639.contains = (SortedNatTree.contains FastRows.row5639)
    exact FastRows.row5639_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root12Child46 : search edge 3 [47, 5639] root12Candidates46 5686 winner = winner := by decide
theorem root12Step46 : scan edge (search edge 3) 4 [47] 47 root12Tail46 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail47 winner := by
  rw [root12Tail46, scan, if_neg (by decide), root12Candidates46_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail47
    (if 3 ≤ root12Candidates46.length then search edge 3 [47, 5639] root12Candidates46 5686 winner else winner) = _
  rw [if_pos (by decide), root12Child46]
def root12Candidates47 : List Nat := [5879, 7517, 8573, 16253]
set_option maxRecDepth 8192 in
theorem root12Candidates47_checked : candidates edge 4 47 5669 root12Tail48 winner.bound = root12Candidates47 := by
  unfold candidates
  have row : edge 5669 = (SortedNatTree.contains FastRows.row5669) := by
    change Data.row5669.contains = (SortedNatTree.contains FastRows.row5669)
    exact FastRows.row5669_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root12Child47 : search edge 3 [47, 5669] root12Candidates47 5716 winner = winner := by decide
theorem root12Step47 : scan edge (search edge 3) 4 [47] 47 root12Tail47 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail48 winner := by
  rw [root12Tail47, scan, if_neg (by decide), root12Candidates47_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail48
    (if 3 ≤ root12Candidates47.length then search edge 3 [47, 5669] root12Candidates47 5716 winner else winner) = _
  rw [if_pos (by decide), root12Child47]
def root12Candidates48 : List Nat := [16529]
set_option maxRecDepth 8192 in
theorem root12Candidates48_checked : candidates edge 4 47 5807 root12Tail49 winner.bound = root12Candidates48 := by
  unfold candidates
  have row : edge 5807 = (SortedNatTree.contains FastRows.row5807) := by
    change Data.row5807.contains = (SortedNatTree.contains FastRows.row5807)
    exact FastRows.row5807_checked.symm
  rw [row]
  decide
theorem root12Step48 : scan edge (search edge 3) 4 [47] 47 root12Tail48 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail49 winner := by
  rw [root12Tail48, scan, if_neg (by decide), root12Candidates48_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail49
    (if 3 ≤ root12Candidates48.length then search edge 3 [47, 5807] root12Candidates48 5854 winner else winner) = _
  rw [if_neg (by decide)]
def root12Candidates49 : List Nat := [8273]
set_option maxRecDepth 8192 in
theorem root12Candidates49_checked : candidates edge 4 47 5879 root12Tail50 winner.bound = root12Candidates49 := by
  unfold candidates
  have row : edge 5879 = (SortedNatTree.contains FastRows.row5879) := by
    change Data.row5879.contains = (SortedNatTree.contains FastRows.row5879)
    exact FastRows.row5879_checked.symm
  rw [row]
  decide
theorem root12Step49 : scan edge (search edge 3) 4 [47] 47 root12Tail49 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail50 winner := by
  rw [root12Tail49, scan, if_neg (by decide), root12Candidates49_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail50
    (if 3 ≤ root12Candidates49.length then search edge 3 [47, 5879] root12Candidates49 5926 winner else winner) = _
  rw [if_neg (by decide)]
def root12Candidates50 : List Nat := [13001, 16253, 18773]
set_option maxRecDepth 8192 in
theorem root12Candidates50_checked : candidates edge 4 47 5927 root12Tail51 winner.bound = root12Candidates50 := by
  unfold candidates
  have row : edge 5927 = (SortedNatTree.contains FastRows.row5927) := by
    change Data.row5927.contains = (SortedNatTree.contains FastRows.row5927)
    exact FastRows.row5927_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root12Child50 : search edge 3 [47, 5927] root12Candidates50 5974 winner = winner := by decide
theorem root12Step50 : scan edge (search edge 3) 4 [47] 47 root12Tail50 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail51 winner := by
  rw [root12Tail50, scan, if_neg (by decide), root12Candidates50_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail51
    (if 3 ≤ root12Candidates50.length then search edge 3 [47, 5927] root12Candidates50 5974 winner else winner) = _
  rw [if_pos (by decide), root12Child50]
def root12Candidates51 : List Nat := [9419, 18269]
set_option maxRecDepth 8192 in
theorem root12Candidates51_checked : candidates edge 4 47 6029 root12Tail52 winner.bound = root12Candidates51 := by
  unfold candidates
  have row : edge 6029 = (SortedNatTree.contains FastRows.row6029) := by
    change Data.row6029.contains = (SortedNatTree.contains FastRows.row6029)
    exact FastRows.row6029_checked.symm
  rw [row]
  decide
theorem root12Step51 : scan edge (search edge 3) 4 [47] 47 root12Tail51 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail52 winner := by
  rw [root12Tail51, scan, if_neg (by decide), root12Candidates51_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail52
    (if 3 ≤ root12Candidates51.length then search edge 3 [47, 6029] root12Candidates51 6076 winner else winner) = _
  rw [if_neg (by decide)]
def root12Candidates52 : List Nat := [8111]
set_option maxRecDepth 8192 in
theorem root12Candidates52_checked : candidates edge 4 47 6089 root12Tail53 winner.bound = root12Candidates52 := by
  unfold candidates
  have row : edge 6089 = (SortedNatTree.contains FastRows.row6089) := by
    change Data.row6089.contains = (SortedNatTree.contains FastRows.row6089)
    exact FastRows.row6089_checked.symm
  rw [row]
  decide
theorem root12Step52 : scan edge (search edge 3) 4 [47] 47 root12Tail52 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail53 winner := by
  rw [root12Tail52, scan, if_neg (by decide), root12Candidates52_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail53
    (if 3 ≤ root12Candidates52.length then search edge 3 [47, 6089] root12Candidates52 6136 winner else winner) = _
  rw [if_neg (by decide)]
end ProjectEulerSolutions.PrimePairLarge.Search
