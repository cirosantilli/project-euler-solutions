import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot000Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root0Candidates0 : List Nat := [109, 229, 541, 673, 823, 1237, 2503, 2707, 4159, 4729, 5521, 9901, 10459, 10627, 14851, 15313, 19183, 19237, 20899, 21397, 22921, 24517, 24847, 25189]
def root0Candidates0Block0 : List Nat := [11, 17, 31, 37, 59, 67, 73, 109, 137, 191, 229, 271, 331, 359, 373, 449, 467, 499, 541, 557, 607, 613, 617, 673, 701, 719, 733, 739, 823, 929, 947, 1013, 1019, 1033, 1051, 1181, 1193, 1237, 1481, 1531, 1607, 1627, 1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699]
set_option maxRecDepth 4096 in
theorem root0Candidates0Block0_shape : root0Tail1 = root0Candidates0Block0 ++ root0Tail129 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates0Block0_checked : root0Candidates0Block0.filter (SortedNatTree.contains FastRows.row7) = [109, 229, 541, 673, 823, 1237, 2503, 2707, 4159, 4729, 5521] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates0Block0_step : candidates edge 4 3 7 root0Tail1 seed.bound =
    [109, 229, 541, 673, 823, 1237, 2503, 2707, 4159, 4729, 5521] ++ candidates edge 4 3 7 root0Tail129 seed.bound := by
  rw [root0Candidates0Block0_shape, candidates_append_of_within edge 4 3 7 root0Candidates0Block0 root0Tail129 seed.bound (by decide)]
  have row : edge 7 = (SortedNatTree.contains FastRows.row7) := by
    change Data.row7.contains = (SortedNatTree.contains FastRows.row7)
    exact FastRows.row7_checked.symm
  rw [row, root0Candidates0Block0_checked]
def root0Candidates0Block1 : List Nat := [8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993]
set_option maxRecDepth 4096 in
theorem root0Candidates0Block1_shape : root0Tail129 = root0Candidates0Block1 ++ root0Tail257 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates0Block1_checked : root0Candidates0Block1.filter (SortedNatTree.contains FastRows.row7) = [9901, 10459, 10627, 14851, 15313, 19183, 19237] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates0Block1_step : candidates edge 4 3 7 root0Tail129 seed.bound =
    [9901, 10459, 10627, 14851, 15313, 19183, 19237] ++ candidates edge 4 3 7 root0Tail257 seed.bound := by
  rw [root0Candidates0Block1_shape, candidates_append_of_within edge 4 3 7 root0Candidates0Block1 root0Tail257 seed.bound (by decide)]
  have row : edge 7 = (SortedNatTree.contains FastRows.row7) := by
    change Data.row7.contains = (SortedNatTree.contains FastRows.row7)
    exact FastRows.row7_checked.symm
  rw [row, root0Candidates0Block1_checked]
def root0Candidates0Block2 : List Nat := [20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239, 24439, 24473, 24517, 24763, 24781, 24799, 24847, 24889, 24953, 25153, 25189, 25219, 25229, 25301, 25343, 25349, 25541, 25747, 25999]
set_option maxRecDepth 4096 in
theorem root0Candidates0Block2_shape : root0Tail257 = root0Candidates0Block2 ++ root0Tail336 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates0Block2_checked : root0Candidates0Block2.filter (SortedNatTree.contains FastRows.row7) = [20899, 21397, 22921, 24517, 24847, 25189] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates0Block2_step : candidates edge 4 3 7 root0Tail257 seed.bound =
    [20899, 21397, 22921, 24517, 24847, 25189] ++ candidates edge 4 3 7 root0Tail336 seed.bound := by
  rw [root0Candidates0Block2_shape, candidates_append_of_within edge 4 3 7 root0Candidates0Block2 root0Tail336 seed.bound (by decide)]
  have row : edge 7 = (SortedNatTree.contains FastRows.row7) := by
    change Data.row7.contains = (SortedNatTree.contains FastRows.row7)
    exact FastRows.row7_checked.symm
  rw [row, root0Candidates0Block2_checked]
theorem root0Candidates0_end : candidates edge 4 3 7 root0Tail336 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates0_checked : candidates edge 4 3 7 root0Tail1 seed.bound = root0Candidates0 := by
  rw [root0Candidates0Block0_step, root0Candidates0Block1_step, root0Candidates0Block2_step, root0Candidates0_end]
  rfl
def root0Child0Tail24 : List Nat := []
def root0Child0Tail23 : List Nat := 25189 :: root0Child0Tail24
def root0Child0Tail22 : List Nat := 24847 :: root0Child0Tail23
def root0Child0Tail21 : List Nat := 24517 :: root0Child0Tail22
def root0Child0Tail20 : List Nat := 22921 :: root0Child0Tail21
def root0Child0Tail19 : List Nat := 21397 :: root0Child0Tail20
def root0Child0Tail18 : List Nat := 20899 :: root0Child0Tail19
def root0Child0Tail17 : List Nat := 19237 :: root0Child0Tail18
def root0Child0Tail16 : List Nat := 19183 :: root0Child0Tail17
def root0Child0Tail15 : List Nat := 15313 :: root0Child0Tail16
def root0Child0Tail14 : List Nat := 14851 :: root0Child0Tail15
def root0Child0Tail13 : List Nat := 10627 :: root0Child0Tail14
def root0Child0Tail12 : List Nat := 10459 :: root0Child0Tail13
def root0Child0Tail11 : List Nat := 9901 :: root0Child0Tail12
def root0Child0Tail10 : List Nat := 5521 :: root0Child0Tail11
def root0Child0Tail9 : List Nat := 4729 :: root0Child0Tail10
def root0Child0Tail8 : List Nat := 4159 :: root0Child0Tail9
def root0Child0Tail7 : List Nat := 2707 :: root0Child0Tail8
def root0Child0Tail6 : List Nat := 2503 :: root0Child0Tail7
def root0Child0Tail5 : List Nat := 1237 :: root0Child0Tail6
def root0Child0Tail4 : List Nat := 823 :: root0Child0Tail5
def root0Child0Tail3 : List Nat := 673 :: root0Child0Tail4
def root0Child0Tail2 : List Nat := 541 :: root0Child0Tail3
def root0Child0Tail1 : List Nat := 229 :: root0Child0Tail2
def root0Child0Tail0 : List Nat := 109 :: root0Child0Tail1
def root0Child0Candidates0 : List Nat := [673]
set_option maxRecDepth 8192 in
theorem root0Child0Candidates0_checked : candidates edge 3 10 109 root0Child0Tail1 seed.bound = root0Child0Candidates0 := by
  unfold candidates
  have row : edge 109 = (SortedNatTree.contains FastRows.row109) := by
    change Data.row109.contains = (SortedNatTree.contains FastRows.row109)
    exact FastRows.row109_checked.symm
  rw [row]
  decide
def root0Child0Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root0Child0Candidates1_checked : candidates edge 3 10 229 root0Child0Tail2 seed.bound = root0Child0Candidates1 := by
  unfold candidates
  have row : edge 229 = (SortedNatTree.contains FastRows.row229) := by
    change Data.row229.contains = (SortedNatTree.contains FastRows.row229)
    exact FastRows.row229_checked.symm
  rw [row]
  decide
def root0Child0Candidates2 : List Nat := [4159]
set_option maxRecDepth 8192 in
theorem root0Child0Candidates2_checked : candidates edge 3 10 541 root0Child0Tail3 seed.bound = root0Child0Candidates2 := by
  unfold candidates
  have row : edge 541 = (SortedNatTree.contains FastRows.row541) := by
    change Data.row541.contains = (SortedNatTree.contains FastRows.row541)
    exact FastRows.row541_checked.symm
  rw [row]
  decide
def root0Child0Candidates3 : List Nat := [22921]
set_option maxRecDepth 8192 in
theorem root0Child0Candidates3_checked : candidates edge 3 10 673 root0Child0Tail4 seed.bound = root0Child0Candidates3 := by
  unfold candidates
  have row : edge 673 = (SortedNatTree.contains FastRows.row673) := by
    change Data.row673.contains = (SortedNatTree.contains FastRows.row673)
    exact FastRows.row673_checked.symm
  rw [row]
  decide
def root0Child0Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root0Child0Candidates4_checked : candidates edge 3 10 823 root0Child0Tail5 seed.bound = root0Child0Candidates4 := by
  unfold candidates
  have row : edge 823 = (SortedNatTree.contains FastRows.row823) := by
    change Data.row823.contains = (SortedNatTree.contains FastRows.row823)
    exact FastRows.row823_checked.symm
  rw [row]
  decide
def root0Child0Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root0Child0Candidates5_checked : candidates edge 3 10 1237 root0Child0Tail6 seed.bound = root0Child0Candidates5 := by
  unfold candidates
  have row : edge 1237 = (SortedNatTree.contains FastRows.row1237) := by
    change Data.row1237.contains = (SortedNatTree.contains FastRows.row1237)
    exact FastRows.row1237_checked.symm
  rw [row]
  decide
def root0Child0Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root0Child0Candidates6_checked : candidates edge 3 10 2503 root0Child0Tail7 seed.bound = root0Child0Candidates6 := by
  unfold candidates
  have row : edge 2503 = (SortedNatTree.contains FastRows.row2503) := by
    change Data.row2503.contains = (SortedNatTree.contains FastRows.row2503)
    exact FastRows.row2503_checked.symm
  rw [row]
  decide
def root0Child0Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root0Child0Candidates7_checked : candidates edge 3 10 2707 root0Child0Tail8 seed.bound = root0Child0Candidates7 := by
  unfold candidates
  have row : edge 2707 = (SortedNatTree.contains FastRows.row2707) := by
    change Data.row2707.contains = (SortedNatTree.contains FastRows.row2707)
    exact FastRows.row2707_checked.symm
  rw [row]
  decide
def root0Child0Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root0Child0Candidates8_checked : candidates edge 3 10 4159 root0Child0Tail9 seed.bound = root0Child0Candidates8 := by
  unfold candidates
  have row : edge 4159 = (SortedNatTree.contains FastRows.row4159) := by
    change Data.row4159.contains = (SortedNatTree.contains FastRows.row4159)
    exact FastRows.row4159_checked.symm
  rw [row]
  decide
def root0Child0Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root0Child0Candidates9_checked : candidates edge 3 10 4729 root0Child0Tail10 seed.bound = root0Child0Candidates9 := by
  unfold candidates
  have row : edge 4729 = (SortedNatTree.contains FastRows.row4729) := by
    change Data.row4729.contains = (SortedNatTree.contains FastRows.row4729)
    exact FastRows.row4729_checked.symm
  rw [row]
  decide
def root0Child0Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root0Child0Candidates10_checked : candidates edge 3 10 5521 root0Child0Tail11 seed.bound = root0Child0Candidates10 := by
  unfold candidates
  have row : edge 5521 = (SortedNatTree.contains FastRows.row5521) := by
    change Data.row5521.contains = (SortedNatTree.contains FastRows.row5521)
    exact FastRows.row5521_checked.symm
  rw [row]
  decide
theorem root0Child0Scan11 : scan edge (search edge 2) 3 [3, 7] 10 root0Child0Tail11 seed = seed := by rw [root0Child0Tail11, scan, if_pos (by decide)]
theorem root0Child0Scan10 : scan edge (search edge 2) 3 [3, 7] 10 root0Child0Tail10 seed = seed := by
  rw [root0Child0Tail10, scan, if_neg (by decide), root0Child0Candidates10_checked]
  change scan edge (search edge 2) 3 [3, 7] 10 root0Child0Tail11
    (if 2 ≤ root0Child0Candidates10.length then search edge 2 [3, 7, 5521] root0Child0Candidates10 5531 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root0Child0Scan11
theorem root0Child0Scan9 : scan edge (search edge 2) 3 [3, 7] 10 root0Child0Tail9 seed = seed := by
  rw [root0Child0Tail9, scan, if_neg (by decide), root0Child0Candidates9_checked]
  change scan edge (search edge 2) 3 [3, 7] 10 root0Child0Tail10
    (if 2 ≤ root0Child0Candidates9.length then search edge 2 [3, 7, 4729] root0Child0Candidates9 4739 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root0Child0Scan10
theorem root0Child0Scan8 : scan edge (search edge 2) 3 [3, 7] 10 root0Child0Tail8 seed = seed := by
  rw [root0Child0Tail8, scan, if_neg (by decide), root0Child0Candidates8_checked]
  change scan edge (search edge 2) 3 [3, 7] 10 root0Child0Tail9
    (if 2 ≤ root0Child0Candidates8.length then search edge 2 [3, 7, 4159] root0Child0Candidates8 4169 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root0Child0Scan9
theorem root0Child0Scan7 : scan edge (search edge 2) 3 [3, 7] 10 root0Child0Tail7 seed = seed := by
  rw [root0Child0Tail7, scan, if_neg (by decide), root0Child0Candidates7_checked]
  change scan edge (search edge 2) 3 [3, 7] 10 root0Child0Tail8
    (if 2 ≤ root0Child0Candidates7.length then search edge 2 [3, 7, 2707] root0Child0Candidates7 2717 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root0Child0Scan8
theorem root0Child0Scan6 : scan edge (search edge 2) 3 [3, 7] 10 root0Child0Tail6 seed = seed := by
  rw [root0Child0Tail6, scan, if_neg (by decide), root0Child0Candidates6_checked]
  change scan edge (search edge 2) 3 [3, 7] 10 root0Child0Tail7
    (if 2 ≤ root0Child0Candidates6.length then search edge 2 [3, 7, 2503] root0Child0Candidates6 2513 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root0Child0Scan7
theorem root0Child0Scan5 : scan edge (search edge 2) 3 [3, 7] 10 root0Child0Tail5 seed = seed := by
  rw [root0Child0Tail5, scan, if_neg (by decide), root0Child0Candidates5_checked]
  change scan edge (search edge 2) 3 [3, 7] 10 root0Child0Tail6
    (if 2 ≤ root0Child0Candidates5.length then search edge 2 [3, 7, 1237] root0Child0Candidates5 1247 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root0Child0Scan6
theorem root0Child0Scan4 : scan edge (search edge 2) 3 [3, 7] 10 root0Child0Tail4 seed = seed := by
  rw [root0Child0Tail4, scan, if_neg (by decide), root0Child0Candidates4_checked]
  change scan edge (search edge 2) 3 [3, 7] 10 root0Child0Tail5
    (if 2 ≤ root0Child0Candidates4.length then search edge 2 [3, 7, 823] root0Child0Candidates4 833 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root0Child0Scan5
theorem root0Child0Scan3 : scan edge (search edge 2) 3 [3, 7] 10 root0Child0Tail3 seed = seed := by
  rw [root0Child0Tail3, scan, if_neg (by decide), root0Child0Candidates3_checked]
  change scan edge (search edge 2) 3 [3, 7] 10 root0Child0Tail4
    (if 2 ≤ root0Child0Candidates3.length then search edge 2 [3, 7, 673] root0Child0Candidates3 683 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root0Child0Scan4
theorem root0Child0Scan2 : scan edge (search edge 2) 3 [3, 7] 10 root0Child0Tail2 seed = seed := by
  rw [root0Child0Tail2, scan, if_neg (by decide), root0Child0Candidates2_checked]
  change scan edge (search edge 2) 3 [3, 7] 10 root0Child0Tail3
    (if 2 ≤ root0Child0Candidates2.length then search edge 2 [3, 7, 541] root0Child0Candidates2 551 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root0Child0Scan3
theorem root0Child0Scan1 : scan edge (search edge 2) 3 [3, 7] 10 root0Child0Tail1 seed = seed := by
  rw [root0Child0Tail1, scan, if_neg (by decide), root0Child0Candidates1_checked]
  change scan edge (search edge 2) 3 [3, 7] 10 root0Child0Tail2
    (if 2 ≤ root0Child0Candidates1.length then search edge 2 [3, 7, 229] root0Child0Candidates1 239 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root0Child0Scan2
theorem root0Child0Scan0 : scan edge (search edge 2) 3 [3, 7] 10 root0Child0Tail0 seed = seed := by
  rw [root0Child0Tail0, scan, if_neg (by decide), root0Child0Candidates0_checked]
  change scan edge (search edge 2) 3 [3, 7] 10 root0Child0Tail1
    (if 2 ≤ root0Child0Candidates0.length then search edge 2 [3, 7, 109] root0Child0Candidates0 119 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root0Child0Scan1
theorem root0Child0 : search edge 3 [3, 7] root0Candidates0 10 seed = seed := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 2) 3 [3, 7] 10 root0Child0Tail0 seed = seed
  exact root0Child0Scan0
theorem root0Step0 : scan edge (search edge 3) 4 [3] 3 root0Tail0 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail1 seed := by
  rw [root0Tail0, scan, if_neg (by decide), root0Candidates0_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail1
    (if 3 ≤ root0Candidates0.length then search edge 3 [3, 7] root0Candidates0 10 seed else seed) = _
  rw [if_pos (by decide), root0Child0]
def root0Candidates1 : List Nat := [701, 2069, 2213, 2297, 2843, 5099, 8219, 8747, 16937, 17489]
def root0Candidates1Block0 : List Nat := [17, 31, 37, 59, 67, 73, 109, 137, 191, 229, 271, 331, 359, 373, 449, 467, 499, 541, 557, 607, 613, 617, 673, 701, 719, 733, 739, 823, 929, 947, 1013, 1019, 1033, 1051, 1181, 1193, 1237, 1481, 1531, 1607, 1627, 1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713]
set_option maxRecDepth 4096 in
theorem root0Candidates1Block0_shape : root0Tail2 = root0Candidates1Block0 ++ root0Tail130 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates1Block0_checked : root0Candidates1Block0.filter (SortedNatTree.contains FastRows.row11) = [701, 2069, 2213, 2297, 2843, 5099, 8219] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates1Block0_step : candidates edge 4 3 11 root0Tail2 seed.bound =
    [701, 2069, 2213, 2297, 2843, 5099, 8219] ++ candidates edge 4 3 11 root0Tail130 seed.bound := by
  rw [root0Candidates1Block0_shape, candidates_append_of_within edge 4 3 11 root0Candidates1Block0 root0Tail130 seed.bound (by decide)]
  have row : edge 11 = (SortedNatTree.contains FastRows.row11) := by
    change Data.row11.contains = (SortedNatTree.contains FastRows.row11)
    exact FastRows.row11_checked.symm
  rw [row, root0Candidates1Block0_checked]
def root0Candidates1Block1 : List Nat := [8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107]
set_option maxRecDepth 4096 in
theorem root0Candidates1Block1_shape : root0Tail130 = root0Candidates1Block1 ++ root0Tail258 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates1Block1_checked : root0Candidates1Block1.filter (SortedNatTree.contains FastRows.row11) = [8747, 16937, 17489] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates1Block1_step : candidates edge 4 3 11 root0Tail130 seed.bound =
    [8747, 16937, 17489] ++ candidates edge 4 3 11 root0Tail258 seed.bound := by
  rw [root0Candidates1Block1_shape, candidates_append_of_within edge 4 3 11 root0Candidates1Block1 root0Tail258 seed.bound (by decide)]
  have row : edge 11 = (SortedNatTree.contains FastRows.row11) := by
    change Data.row11.contains = (SortedNatTree.contains FastRows.row11)
    exact FastRows.row11_checked.symm
  rw [row, root0Candidates1Block1_checked]
def root0Candidates1Block2 : List Nat := [20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239, 24439, 24473, 24517, 24763, 24781, 24799, 24847, 24889, 24953, 25153, 25189, 25219, 25229, 25301, 25343, 25349, 25541, 25747, 25999]
set_option maxRecDepth 4096 in
theorem root0Candidates1Block2_shape : root0Tail258 = root0Candidates1Block2 ++ root0Tail336 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates1Block2_checked : root0Candidates1Block2.filter (SortedNatTree.contains FastRows.row11) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates1Block2_step : candidates edge 4 3 11 root0Tail258 seed.bound =
    [] ++ candidates edge 4 3 11 root0Tail336 seed.bound := by
  rw [root0Candidates1Block2_shape, candidates_append_of_within edge 4 3 11 root0Candidates1Block2 root0Tail336 seed.bound (by decide)]
  have row : edge 11 = (SortedNatTree.contains FastRows.row11) := by
    change Data.row11.contains = (SortedNatTree.contains FastRows.row11)
    exact FastRows.row11_checked.symm
  rw [row, root0Candidates1Block2_checked]
theorem root0Candidates1_end : candidates edge 4 3 11 root0Tail336 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates1_checked : candidates edge 4 3 11 root0Tail2 seed.bound = root0Candidates1 := by
  rw [root0Candidates1Block0_step, root0Candidates1Block1_step, root0Candidates1Block2_step, root0Candidates1_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child1 : search edge 3 [3, 11] root0Candidates1 14 seed = seed := by decide
theorem root0Step1 : scan edge (search edge 3) 4 [3] 3 root0Tail1 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail2 seed := by
  rw [root0Tail1, scan, if_neg (by decide), root0Candidates1_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail2
    (if 3 ≤ root0Candidates1.length then search edge 3 [3, 11] root0Candidates1 14 seed else seed) = _
  rw [if_pos (by decide), root0Child1]
def root0Candidates2 : List Nat := [449, 2069, 2213, 2297, 2411, 4019, 4583, 6353, 6599, 6899, 7691, 13763, 21143, 23027, 25301, 25343]
def root0Candidates2Block0 : List Nat := [31, 37, 59, 67, 73, 109, 137, 191, 229, 271, 331, 359, 373, 449, 467, 499, 541, 557, 607, 613, 617, 673, 701, 719, 733, 739, 823, 929, 947, 1013, 1019, 1033, 1051, 1181, 1193, 1237, 1481, 1531, 1607, 1627, 1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747]
set_option maxRecDepth 4096 in
theorem root0Candidates2Block0_shape : root0Tail3 = root0Candidates2Block0 ++ root0Tail131 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates2Block0_checked : root0Candidates2Block0.filter (SortedNatTree.contains FastRows.row17) = [449, 2069, 2213, 2297, 2411, 4019, 4583, 6353, 6599, 6899, 7691] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates2Block0_step : candidates edge 4 3 17 root0Tail3 seed.bound =
    [449, 2069, 2213, 2297, 2411, 4019, 4583, 6353, 6599, 6899, 7691] ++ candidates edge 4 3 17 root0Tail131 seed.bound := by
  rw [root0Candidates2Block0_shape, candidates_append_of_within edge 4 3 17 root0Candidates2Block0 root0Tail131 seed.bound (by decide)]
  have row : edge 17 = (SortedNatTree.contains FastRows.row17) := by
    change Data.row17.contains = (SortedNatTree.contains FastRows.row17)
    exact FastRows.row17_checked.symm
  rw [row, root0Candidates2Block0_checked]
def root0Candidates2Block1 : List Nat := [8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149]
set_option maxRecDepth 4096 in
theorem root0Candidates2Block1_shape : root0Tail131 = root0Candidates2Block1 ++ root0Tail259 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates2Block1_checked : root0Candidates2Block1.filter (SortedNatTree.contains FastRows.row17) = [13763] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates2Block1_step : candidates edge 4 3 17 root0Tail131 seed.bound =
    [13763] ++ candidates edge 4 3 17 root0Tail259 seed.bound := by
  rw [root0Candidates2Block1_shape, candidates_append_of_within edge 4 3 17 root0Candidates2Block1 root0Tail259 seed.bound (by decide)]
  have row : edge 17 = (SortedNatTree.contains FastRows.row17) := by
    change Data.row17.contains = (SortedNatTree.contains FastRows.row17)
    exact FastRows.row17_checked.symm
  rw [row, root0Candidates2Block1_checked]
def root0Candidates2Block2 : List Nat := [20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239, 24439, 24473, 24517, 24763, 24781, 24799, 24847, 24889, 24953, 25153, 25189, 25219, 25229, 25301, 25343, 25349, 25541, 25747, 25999]
set_option maxRecDepth 4096 in
theorem root0Candidates2Block2_shape : root0Tail259 = root0Candidates2Block2 ++ root0Tail336 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates2Block2_checked : root0Candidates2Block2.filter (SortedNatTree.contains FastRows.row17) = [21143, 23027, 25301, 25343] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates2Block2_step : candidates edge 4 3 17 root0Tail259 seed.bound =
    [21143, 23027, 25301, 25343] ++ candidates edge 4 3 17 root0Tail336 seed.bound := by
  rw [root0Candidates2Block2_shape, candidates_append_of_within edge 4 3 17 root0Candidates2Block2 root0Tail336 seed.bound (by decide)]
  have row : edge 17 = (SortedNatTree.contains FastRows.row17) := by
    change Data.row17.contains = (SortedNatTree.contains FastRows.row17)
    exact FastRows.row17_checked.symm
  rw [row, root0Candidates2Block2_checked]
theorem root0Candidates2_end : candidates edge 4 3 17 root0Tail336 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates2_checked : candidates edge 4 3 17 root0Tail3 seed.bound = root0Candidates2 := by
  rw [root0Candidates2Block0_step, root0Candidates2Block1_step, root0Candidates2Block2_step, root0Candidates2_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child2 : search edge 3 [3, 17] root0Candidates2 20 seed = seed := by decide
theorem root0Step2 : scan edge (search edge 3) 4 [3] 3 root0Tail2 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail3 seed := by
  rw [root0Tail2, scan, if_neg (by decide), root0Candidates2_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail3
    (if 3 ≤ root0Candidates2.length then search edge 3 [3, 17] root0Candidates2 20 seed else seed) = _
  rw [if_pos (by decide), root0Child2]
def root0Candidates3 : List Nat := [1237, 3301, 3637, 4159, 5281, 6037, 6073, 6571, 7591, 8377, 8431, 12253, 13399, 15907, 17209, 17923]
def root0Candidates3Block0 : List Nat := [37, 59, 67, 73, 109, 137, 191, 229, 271, 331, 359, 373, 449, 467, 499, 541, 557, 607, 613, 617, 673, 701, 719, 733, 739, 823, 929, 947, 1013, 1019, 1033, 1051, 1181, 1193, 1237, 1481, 1531, 1607, 1627, 1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783]
set_option maxRecDepth 4096 in
theorem root0Candidates3Block0_shape : root0Tail4 = root0Candidates3Block0 ++ root0Tail132 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates3Block0_checked : root0Candidates3Block0.filter (SortedNatTree.contains FastRows.row31) = [1237, 3301, 3637, 4159, 5281, 6037, 6073, 6571, 7591, 8377, 8431] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates3Block0_step : candidates edge 4 3 31 root0Tail4 seed.bound =
    [1237, 3301, 3637, 4159, 5281, 6037, 6073, 6571, 7591, 8377, 8431] ++ candidates edge 4 3 31 root0Tail132 seed.bound := by
  rw [root0Candidates3Block0_shape, candidates_append_of_within edge 4 3 31 root0Candidates3Block0 root0Tail132 seed.bound (by decide)]
  have row : edge 31 = (SortedNatTree.contains FastRows.row31) := by
    change Data.row31.contains = (SortedNatTree.contains FastRows.row31)
    exact FastRows.row31_checked.symm
  rw [row, root0Candidates3Block0_checked]
def root0Candidates3Block1 : List Nat := [9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269]
set_option maxRecDepth 4096 in
theorem root0Candidates3Block1_shape : root0Tail132 = root0Candidates3Block1 ++ root0Tail260 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates3Block1_checked : root0Candidates3Block1.filter (SortedNatTree.contains FastRows.row31) = [12253, 13399, 15907, 17209, 17923] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates3Block1_step : candidates edge 4 3 31 root0Tail132 seed.bound =
    [12253, 13399, 15907, 17209, 17923] ++ candidates edge 4 3 31 root0Tail260 seed.bound := by
  rw [root0Candidates3Block1_shape, candidates_append_of_within edge 4 3 31 root0Candidates3Block1 root0Tail260 seed.bound (by decide)]
  have row : edge 31 = (SortedNatTree.contains FastRows.row31) := by
    change Data.row31.contains = (SortedNatTree.contains FastRows.row31)
    exact FastRows.row31_checked.symm
  rw [row, root0Candidates3Block1_checked]
def root0Candidates3Block2 : List Nat := [20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239, 24439, 24473, 24517, 24763, 24781, 24799, 24847, 24889, 24953, 25153, 25189, 25219, 25229, 25301, 25343, 25349, 25541, 25747, 25999]
set_option maxRecDepth 4096 in
theorem root0Candidates3Block2_shape : root0Tail260 = root0Candidates3Block2 ++ root0Tail336 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates3Block2_checked : root0Candidates3Block2.filter (SortedNatTree.contains FastRows.row31) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates3Block2_step : candidates edge 4 3 31 root0Tail260 seed.bound =
    [] ++ candidates edge 4 3 31 root0Tail336 seed.bound := by
  rw [root0Candidates3Block2_shape, candidates_append_of_within edge 4 3 31 root0Candidates3Block2 root0Tail336 seed.bound (by decide)]
  have row : edge 31 = (SortedNatTree.contains FastRows.row31) := by
    change Data.row31.contains = (SortedNatTree.contains FastRows.row31)
    exact FastRows.row31_checked.symm
  rw [row, root0Candidates3Block2_checked]
theorem root0Candidates3_end : candidates edge 4 3 31 root0Tail336 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates3_checked : candidates edge 4 3 31 root0Tail4 seed.bound = root0Candidates3 := by
  rw [root0Candidates3Block0_step, root0Candidates3Block1_step, root0Candidates3Block2_step, root0Candidates3_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child3 : search edge 3 [3, 31] root0Candidates3 34 seed = seed := by decide
theorem root0Step3 : scan edge (search edge 3) 4 [3] 3 root0Tail3 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail4 seed := by
  rw [root0Tail3, scan, if_neg (by decide), root0Candidates3_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail4
    (if 3 ≤ root0Candidates3.length then search edge 3 [3, 31] root0Candidates3 34 seed else seed) = _
  rw [if_pos (by decide), root0Child3]
def root0Candidates4 : List Nat := [67, 607, 1237, 1699, 2377, 2707, 3049, 4159, 4729, 5281, 5449, 5923, 6469, 6529, 7369, 7963, 8713, 9157, 10243, 14779, 15373, 15739, 16759, 17209, 22549, 23581]
def root0Candidates4Block0 : List Nat := [59, 67, 73, 109, 137, 191, 229, 271, 331, 359, 373, 449, 467, 499, 541, 557, 607, 613, 617, 673, 701, 719, 733, 739, 823, 929, 947, 1013, 1019, 1033, 1051, 1181, 1193, 1237, 1481, 1531, 1607, 1627, 1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103]
set_option maxRecDepth 4096 in
theorem root0Candidates4Block0_shape : root0Tail5 = root0Candidates4Block0 ++ root0Tail133 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates4Block0_checked : root0Candidates4Block0.filter (SortedNatTree.contains FastRows.row37) = [67, 607, 1237, 1699, 2377, 2707, 3049, 4159, 4729, 5281, 5449, 5923, 6469, 6529, 7369, 7963, 8713] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates4Block0_step : candidates edge 4 3 37 root0Tail5 seed.bound =
    [67, 607, 1237, 1699, 2377, 2707, 3049, 4159, 4729, 5281, 5449, 5923, 6469, 6529, 7369, 7963, 8713] ++ candidates edge 4 3 37 root0Tail133 seed.bound := by
  rw [root0Candidates4Block0_shape, candidates_append_of_within edge 4 3 37 root0Candidates4Block0 root0Tail133 seed.bound (by decide)]
  have row : edge 37 = (SortedNatTree.contains FastRows.row37) := by
    change Data.row37.contains = (SortedNatTree.contains FastRows.row37)
    exact FastRows.row37_checked.symm
  rw [row, root0Candidates4Block0_checked]
def root0Candidates4Block1 : List Nat := [9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521]
set_option maxRecDepth 4096 in
theorem root0Candidates4Block1_shape : root0Tail133 = root0Candidates4Block1 ++ root0Tail261 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates4Block1_checked : root0Candidates4Block1.filter (SortedNatTree.contains FastRows.row37) = [9157, 10243, 14779, 15373, 15739, 16759, 17209] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates4Block1_step : candidates edge 4 3 37 root0Tail133 seed.bound =
    [9157, 10243, 14779, 15373, 15739, 16759, 17209] ++ candidates edge 4 3 37 root0Tail261 seed.bound := by
  rw [root0Candidates4Block1_shape, candidates_append_of_within edge 4 3 37 root0Candidates4Block1 root0Tail261 seed.bound (by decide)]
  have row : edge 37 = (SortedNatTree.contains FastRows.row37) := by
    change Data.row37.contains = (SortedNatTree.contains FastRows.row37)
    exact FastRows.row37_checked.symm
  rw [row, root0Candidates4Block1_checked]
def root0Candidates4Block2 : List Nat := [20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239, 24439, 24473, 24517, 24763, 24781, 24799, 24847, 24889, 24953, 25153, 25189, 25219, 25229, 25301, 25343, 25349, 25541, 25747]
set_option maxRecDepth 4096 in
theorem root0Candidates4Block2_shape : root0Tail261 = root0Candidates4Block2 ++ root0Tail335 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates4Block2_checked : root0Candidates4Block2.filter (SortedNatTree.contains FastRows.row37) = [22549, 23581] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates4Block2_step : candidates edge 4 3 37 root0Tail261 seed.bound =
    [22549, 23581] ++ candidates edge 4 3 37 root0Tail335 seed.bound := by
  rw [root0Candidates4Block2_shape, candidates_append_of_within edge 4 3 37 root0Candidates4Block2 root0Tail335 seed.bound (by decide)]
  have row : edge 37 = (SortedNatTree.contains FastRows.row37) := by
    change Data.row37.contains = (SortedNatTree.contains FastRows.row37)
    exact FastRows.row37_checked.symm
  rw [row, root0Candidates4Block2_checked]
theorem root0Candidates4_end : candidates edge 4 3 37 root0Tail335 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates4_checked : candidates edge 4 3 37 root0Tail5 seed.bound = root0Candidates4 := by
  rw [root0Candidates4Block0_step, root0Candidates4Block1_step, root0Candidates4Block2_step, root0Candidates4_end]
  rfl
def root0Child4Tail26 : List Nat := []
def root0Child4Tail25 : List Nat := 23581 :: root0Child4Tail26
def root0Child4Tail24 : List Nat := 22549 :: root0Child4Tail25
def root0Child4Tail23 : List Nat := 17209 :: root0Child4Tail24
def root0Child4Tail22 : List Nat := 16759 :: root0Child4Tail23
def root0Child4Tail21 : List Nat := 15739 :: root0Child4Tail22
def root0Child4Tail20 : List Nat := 15373 :: root0Child4Tail21
def root0Child4Tail19 : List Nat := 14779 :: root0Child4Tail20
def root0Child4Tail18 : List Nat := 10243 :: root0Child4Tail19
def root0Child4Tail17 : List Nat := 9157 :: root0Child4Tail18
def root0Child4Tail16 : List Nat := 8713 :: root0Child4Tail17
def root0Child4Tail15 : List Nat := 7963 :: root0Child4Tail16
def root0Child4Tail14 : List Nat := 7369 :: root0Child4Tail15
def root0Child4Tail13 : List Nat := 6529 :: root0Child4Tail14
def root0Child4Tail12 : List Nat := 6469 :: root0Child4Tail13
def root0Child4Tail11 : List Nat := 5923 :: root0Child4Tail12
def root0Child4Tail10 : List Nat := 5449 :: root0Child4Tail11
def root0Child4Tail9 : List Nat := 5281 :: root0Child4Tail10
def root0Child4Tail8 : List Nat := 4729 :: root0Child4Tail9
def root0Child4Tail7 : List Nat := 4159 :: root0Child4Tail8
def root0Child4Tail6 : List Nat := 3049 :: root0Child4Tail7
def root0Child4Tail5 : List Nat := 2707 :: root0Child4Tail6
def root0Child4Tail4 : List Nat := 2377 :: root0Child4Tail5
def root0Child4Tail3 : List Nat := 1699 :: root0Child4Tail4
def root0Child4Tail2 : List Nat := 1237 :: root0Child4Tail3
def root0Child4Tail1 : List Nat := 607 :: root0Child4Tail2
def root0Child4Tail0 : List Nat := 67 :: root0Child4Tail1
def root0Child4Candidates0 : List Nat := [2377, 5923]
set_option maxRecDepth 8192 in
theorem root0Child4Candidates0_checked : candidates edge 3 40 67 root0Child4Tail1 seed.bound = root0Child4Candidates0 := by
  unfold candidates
  have row : edge 67 = (SortedNatTree.contains FastRows.row67) := by
    change Data.row67.contains = (SortedNatTree.contains FastRows.row67)
    exact FastRows.row67_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root0Child4Child0 : search edge 2 [3, 37, 67] root0Child4Candidates0 107 seed = seed := by decide
def root0Child4Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root0Child4Candidates1_checked : candidates edge 3 40 607 root0Child4Tail2 seed.bound = root0Child4Candidates1 := by
  unfold candidates
  have row : edge 607 = (SortedNatTree.contains FastRows.row607) := by
    change Data.row607.contains = (SortedNatTree.contains FastRows.row607)
    exact FastRows.row607_checked.symm
  rw [row]
  decide
def root0Child4Candidates2 : List Nat := [8713]
set_option maxRecDepth 8192 in
theorem root0Child4Candidates2_checked : candidates edge 3 40 1237 root0Child4Tail3 seed.bound = root0Child4Candidates2 := by
  unfold candidates
  have row : edge 1237 = (SortedNatTree.contains FastRows.row1237) := by
    change Data.row1237.contains = (SortedNatTree.contains FastRows.row1237)
    exact FastRows.row1237_checked.symm
  rw [row]
  decide
def root0Child4Candidates3 : List Nat := [8713]
set_option maxRecDepth 8192 in
theorem root0Child4Candidates3_checked : candidates edge 3 40 1699 root0Child4Tail4 seed.bound = root0Child4Candidates3 := by
  unfold candidates
  have row : edge 1699 = (SortedNatTree.contains FastRows.row1699) := by
    change Data.row1699.contains = (SortedNatTree.contains FastRows.row1699)
    exact FastRows.row1699_checked.symm
  rw [row]
  decide
def root0Child4Candidates4 : List Nat := [4159, 10243]
set_option maxRecDepth 8192 in
theorem root0Child4Candidates4_checked : candidates edge 3 40 2377 root0Child4Tail5 seed.bound = root0Child4Candidates4 := by
  unfold candidates
  have row : edge 2377 = (SortedNatTree.contains FastRows.row2377) := by
    change Data.row2377.contains = (SortedNatTree.contains FastRows.row2377)
    exact FastRows.row2377_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root0Child4Child4 : search edge 2 [3, 37, 2377] root0Child4Candidates4 2417 seed = seed := by decide
def root0Child4Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root0Child4Candidates5_checked : candidates edge 3 40 2707 root0Child4Tail6 seed.bound = root0Child4Candidates5 := by
  unfold candidates
  have row : edge 2707 = (SortedNatTree.contains FastRows.row2707) := by
    change Data.row2707.contains = (SortedNatTree.contains FastRows.row2707)
    exact FastRows.row2707_checked.symm
  rw [row]
  decide
def root0Child4Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root0Child4Candidates6_checked : candidates edge 3 40 3049 root0Child4Tail7 seed.bound = root0Child4Candidates6 := by
  unfold candidates
  have row : edge 3049 = (SortedNatTree.contains FastRows.row3049) := by
    change Data.row3049.contains = (SortedNatTree.contains FastRows.row3049)
    exact FastRows.row3049_checked.symm
  rw [row]
  decide
def root0Child4Candidates7 : List Nat := [16759, 17209]
set_option maxRecDepth 8192 in
theorem root0Child4Candidates7_checked : candidates edge 3 40 4159 root0Child4Tail8 seed.bound = root0Child4Candidates7 := by
  unfold candidates
  have row : edge 4159 = (SortedNatTree.contains FastRows.row4159) := by
    change Data.row4159.contains = (SortedNatTree.contains FastRows.row4159)
    exact FastRows.row4159_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root0Child4Child7 : search edge 2 [3, 37, 4159] root0Child4Candidates7 4199 seed = seed := by decide
def root0Child4Candidates8 : List Nat := [7963]
set_option maxRecDepth 8192 in
theorem root0Child4Candidates8_checked : candidates edge 3 40 4729 root0Child4Tail9 seed.bound = root0Child4Candidates8 := by
  unfold candidates
  have row : edge 4729 = (SortedNatTree.contains FastRows.row4729) := by
    change Data.row4729.contains = (SortedNatTree.contains FastRows.row4729)
    exact FastRows.row4729_checked.symm
  rw [row]
  decide
def root0Child4Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root0Child4Candidates9_checked : candidates edge 3 40 5281 root0Child4Tail10 seed.bound = root0Child4Candidates9 := by
  unfold candidates
  have row : edge 5281 = (SortedNatTree.contains FastRows.row5281) := by
    change Data.row5281.contains = (SortedNatTree.contains FastRows.row5281)
    exact FastRows.row5281_checked.symm
  rw [row]
  decide
def root0Child4Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root0Child4Candidates10_checked : candidates edge 3 40 5449 root0Child4Tail11 seed.bound = root0Child4Candidates10 := by
  unfold candidates
  have row : edge 5449 = (SortedNatTree.contains FastRows.row5449) := by
    change Data.row5449.contains = (SortedNatTree.contains FastRows.row5449)
    exact FastRows.row5449_checked.symm
  rw [row]
  decide
def root0Child4Candidates11 : List Nat := [7963]
set_option maxRecDepth 8192 in
theorem root0Child4Candidates11_checked : candidates edge 3 40 5923 root0Child4Tail12 seed.bound = root0Child4Candidates11 := by
  unfold candidates
  have row : edge 5923 = (SortedNatTree.contains FastRows.row5923) := by
    change Data.row5923.contains = (SortedNatTree.contains FastRows.row5923)
    exact FastRows.row5923_checked.symm
  rw [row]
  decide
def root0Child4Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root0Child4Candidates12_checked : candidates edge 3 40 6469 root0Child4Tail13 seed.bound = root0Child4Candidates12 := by
  unfold candidates
  have row : edge 6469 = (SortedNatTree.contains FastRows.row6469) := by
    change Data.row6469.contains = (SortedNatTree.contains FastRows.row6469)
    exact FastRows.row6469_checked.symm
  rw [row]
  decide
def root0Child4Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root0Child4Candidates13_checked : candidates edge 3 40 6529 root0Child4Tail14 seed.bound = root0Child4Candidates13 := by
  unfold candidates
  have row : edge 6529 = (SortedNatTree.contains FastRows.row6529) := by
    change Data.row6529.contains = (SortedNatTree.contains FastRows.row6529)
    exact FastRows.row6529_checked.symm
  rw [row]
  decide
def root0Child4Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root0Child4Candidates14_checked : candidates edge 3 40 7369 root0Child4Tail15 seed.bound = root0Child4Candidates14 := by
  unfold candidates
  have row : edge 7369 = (SortedNatTree.contains FastRows.row7369) := by
    change Data.row7369.contains = (SortedNatTree.contains FastRows.row7369)
    exact FastRows.row7369_checked.symm
  rw [row]
  decide
def root0Child4Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root0Child4Candidates15_checked : candidates edge 3 40 7963 root0Child4Tail16 seed.bound = root0Child4Candidates15 := by
  unfold candidates
  have row : edge 7963 = (SortedNatTree.contains FastRows.row7963) := by
    change Data.row7963.contains = (SortedNatTree.contains FastRows.row7963)
    exact FastRows.row7963_checked.symm
  rw [row]
  decide
theorem root0Child4Scan16 : scan edge (search edge 2) 3 [3, 37] 40 root0Child4Tail16 seed = seed := by rw [root0Child4Tail16, scan, if_pos (by decide)]
theorem root0Child4Scan15 : scan edge (search edge 2) 3 [3, 37] 40 root0Child4Tail15 seed = seed := by
  rw [root0Child4Tail15, scan, if_neg (by decide), root0Child4Candidates15_checked]
  change scan edge (search edge 2) 3 [3, 37] 40 root0Child4Tail16
    (if 2 ≤ root0Child4Candidates15.length then search edge 2 [3, 37, 7963] root0Child4Candidates15 8003 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root0Child4Scan16
theorem root0Child4Scan14 : scan edge (search edge 2) 3 [3, 37] 40 root0Child4Tail14 seed = seed := by
  rw [root0Child4Tail14, scan, if_neg (by decide), root0Child4Candidates14_checked]
  change scan edge (search edge 2) 3 [3, 37] 40 root0Child4Tail15
    (if 2 ≤ root0Child4Candidates14.length then search edge 2 [3, 37, 7369] root0Child4Candidates14 7409 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root0Child4Scan15
theorem root0Child4Scan13 : scan edge (search edge 2) 3 [3, 37] 40 root0Child4Tail13 seed = seed := by
  rw [root0Child4Tail13, scan, if_neg (by decide), root0Child4Candidates13_checked]
  change scan edge (search edge 2) 3 [3, 37] 40 root0Child4Tail14
    (if 2 ≤ root0Child4Candidates13.length then search edge 2 [3, 37, 6529] root0Child4Candidates13 6569 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root0Child4Scan14
theorem root0Child4Scan12 : scan edge (search edge 2) 3 [3, 37] 40 root0Child4Tail12 seed = seed := by
  rw [root0Child4Tail12, scan, if_neg (by decide), root0Child4Candidates12_checked]
  change scan edge (search edge 2) 3 [3, 37] 40 root0Child4Tail13
    (if 2 ≤ root0Child4Candidates12.length then search edge 2 [3, 37, 6469] root0Child4Candidates12 6509 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root0Child4Scan13
theorem root0Child4Scan11 : scan edge (search edge 2) 3 [3, 37] 40 root0Child4Tail11 seed = seed := by
  rw [root0Child4Tail11, scan, if_neg (by decide), root0Child4Candidates11_checked]
  change scan edge (search edge 2) 3 [3, 37] 40 root0Child4Tail12
    (if 2 ≤ root0Child4Candidates11.length then search edge 2 [3, 37, 5923] root0Child4Candidates11 5963 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root0Child4Scan12
theorem root0Child4Scan10 : scan edge (search edge 2) 3 [3, 37] 40 root0Child4Tail10 seed = seed := by
  rw [root0Child4Tail10, scan, if_neg (by decide), root0Child4Candidates10_checked]
  change scan edge (search edge 2) 3 [3, 37] 40 root0Child4Tail11
    (if 2 ≤ root0Child4Candidates10.length then search edge 2 [3, 37, 5449] root0Child4Candidates10 5489 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root0Child4Scan11
theorem root0Child4Scan9 : scan edge (search edge 2) 3 [3, 37] 40 root0Child4Tail9 seed = seed := by
  rw [root0Child4Tail9, scan, if_neg (by decide), root0Child4Candidates9_checked]
  change scan edge (search edge 2) 3 [3, 37] 40 root0Child4Tail10
    (if 2 ≤ root0Child4Candidates9.length then search edge 2 [3, 37, 5281] root0Child4Candidates9 5321 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root0Child4Scan10
theorem root0Child4Scan8 : scan edge (search edge 2) 3 [3, 37] 40 root0Child4Tail8 seed = seed := by
  rw [root0Child4Tail8, scan, if_neg (by decide), root0Child4Candidates8_checked]
  change scan edge (search edge 2) 3 [3, 37] 40 root0Child4Tail9
    (if 2 ≤ root0Child4Candidates8.length then search edge 2 [3, 37, 4729] root0Child4Candidates8 4769 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root0Child4Scan9
theorem root0Child4Scan7 : scan edge (search edge 2) 3 [3, 37] 40 root0Child4Tail7 seed = seed := by
  rw [root0Child4Tail7, scan, if_neg (by decide), root0Child4Candidates7_checked]
  change scan edge (search edge 2) 3 [3, 37] 40 root0Child4Tail8
    (if 2 ≤ root0Child4Candidates7.length then search edge 2 [3, 37, 4159] root0Child4Candidates7 4199 seed else seed) = seed
  rw [if_pos (by decide), root0Child4Child7]
  exact root0Child4Scan8
theorem root0Child4Scan6 : scan edge (search edge 2) 3 [3, 37] 40 root0Child4Tail6 seed = seed := by
  rw [root0Child4Tail6, scan, if_neg (by decide), root0Child4Candidates6_checked]
  change scan edge (search edge 2) 3 [3, 37] 40 root0Child4Tail7
    (if 2 ≤ root0Child4Candidates6.length then search edge 2 [3, 37, 3049] root0Child4Candidates6 3089 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root0Child4Scan7
theorem root0Child4Scan5 : scan edge (search edge 2) 3 [3, 37] 40 root0Child4Tail5 seed = seed := by
  rw [root0Child4Tail5, scan, if_neg (by decide), root0Child4Candidates5_checked]
  change scan edge (search edge 2) 3 [3, 37] 40 root0Child4Tail6
    (if 2 ≤ root0Child4Candidates5.length then search edge 2 [3, 37, 2707] root0Child4Candidates5 2747 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root0Child4Scan6
theorem root0Child4Scan4 : scan edge (search edge 2) 3 [3, 37] 40 root0Child4Tail4 seed = seed := by
  rw [root0Child4Tail4, scan, if_neg (by decide), root0Child4Candidates4_checked]
  change scan edge (search edge 2) 3 [3, 37] 40 root0Child4Tail5
    (if 2 ≤ root0Child4Candidates4.length then search edge 2 [3, 37, 2377] root0Child4Candidates4 2417 seed else seed) = seed
  rw [if_pos (by decide), root0Child4Child4]
  exact root0Child4Scan5
theorem root0Child4Scan3 : scan edge (search edge 2) 3 [3, 37] 40 root0Child4Tail3 seed = seed := by
  rw [root0Child4Tail3, scan, if_neg (by decide), root0Child4Candidates3_checked]
  change scan edge (search edge 2) 3 [3, 37] 40 root0Child4Tail4
    (if 2 ≤ root0Child4Candidates3.length then search edge 2 [3, 37, 1699] root0Child4Candidates3 1739 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root0Child4Scan4
theorem root0Child4Scan2 : scan edge (search edge 2) 3 [3, 37] 40 root0Child4Tail2 seed = seed := by
  rw [root0Child4Tail2, scan, if_neg (by decide), root0Child4Candidates2_checked]
  change scan edge (search edge 2) 3 [3, 37] 40 root0Child4Tail3
    (if 2 ≤ root0Child4Candidates2.length then search edge 2 [3, 37, 1237] root0Child4Candidates2 1277 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root0Child4Scan3
theorem root0Child4Scan1 : scan edge (search edge 2) 3 [3, 37] 40 root0Child4Tail1 seed = seed := by
  rw [root0Child4Tail1, scan, if_neg (by decide), root0Child4Candidates1_checked]
  change scan edge (search edge 2) 3 [3, 37] 40 root0Child4Tail2
    (if 2 ≤ root0Child4Candidates1.length then search edge 2 [3, 37, 607] root0Child4Candidates1 647 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root0Child4Scan2
theorem root0Child4Scan0 : scan edge (search edge 2) 3 [3, 37] 40 root0Child4Tail0 seed = seed := by
  rw [root0Child4Tail0, scan, if_neg (by decide), root0Child4Candidates0_checked]
  change scan edge (search edge 2) 3 [3, 37] 40 root0Child4Tail1
    (if 2 ≤ root0Child4Candidates0.length then search edge 2 [3, 37, 67] root0Child4Candidates0 107 seed else seed) = seed
  rw [if_pos (by decide), root0Child4Child0]
  exact root0Child4Scan1
theorem root0Child4 : search edge 3 [3, 37] root0Candidates4 40 seed = seed := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 2) 3 [3, 37] 40 root0Child4Tail0 seed = seed
  exact root0Child4Scan0
theorem root0Step4 : scan edge (search edge 3) 4 [3] 3 root0Tail4 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail5 seed := by
  rw [root0Tail4, scan, if_neg (by decide), root0Candidates4_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail5
    (if 3 ≤ root0Candidates4.length then search edge 3 [3, 37] root0Candidates4 40 seed else seed) = _
  rw [if_pos (by decide), root0Child4]
def root0Candidates5 : List Nat := [929, 1193, 5381, 6653, 6899, 7253, 8783, 9857, 13763, 13931, 14627, 17903, 19031, 19937, 22067, 22349]
def root0Candidates5Block0 : List Nat := [67, 73, 109, 137, 191, 229, 271, 331, 359, 373, 449, 467, 499, 541, 557, 607, 613, 617, 673, 701, 719, 733, 739, 823, 929, 947, 1013, 1019, 1033, 1051, 1181, 1193, 1237, 1481, 1531, 1607, 1627, 1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157]
set_option maxRecDepth 4096 in
theorem root0Candidates5Block0_shape : root0Tail6 = root0Candidates5Block0 ++ root0Tail134 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates5Block0_checked : root0Candidates5Block0.filter (SortedNatTree.contains FastRows.row59) = [929, 1193, 5381, 6653, 6899, 7253, 8783] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates5Block0_step : candidates edge 4 3 59 root0Tail6 seed.bound =
    [929, 1193, 5381, 6653, 6899, 7253, 8783] ++ candidates edge 4 3 59 root0Tail134 seed.bound := by
  rw [root0Candidates5Block0_shape, candidates_append_of_within edge 4 3 59 root0Candidates5Block0 root0Tail134 seed.bound (by decide)]
  have row : edge 59 = (SortedNatTree.contains FastRows.row59) := by
    change Data.row59.contains = (SortedNatTree.contains FastRows.row59)
    exact FastRows.row59_checked.symm
  rw [row, root0Candidates5Block0_checked]
def root0Candidates5Block1 : List Nat := [9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563]
set_option maxRecDepth 4096 in
theorem root0Candidates5Block1_shape : root0Tail134 = root0Candidates5Block1 ++ root0Tail262 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates5Block1_checked : root0Candidates5Block1.filter (SortedNatTree.contains FastRows.row59) = [9857, 13763, 13931, 14627, 17903, 19031, 19937] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates5Block1_step : candidates edge 4 3 59 root0Tail134 seed.bound =
    [9857, 13763, 13931, 14627, 17903, 19031, 19937] ++ candidates edge 4 3 59 root0Tail262 seed.bound := by
  rw [root0Candidates5Block1_shape, candidates_append_of_within edge 4 3 59 root0Candidates5Block1 root0Tail262 seed.bound (by decide)]
  have row : edge 59 = (SortedNatTree.contains FastRows.row59) := by
    change Data.row59.contains = (SortedNatTree.contains FastRows.row59)
    exact FastRows.row59_checked.symm
  rw [row, root0Candidates5Block1_checked]
def root0Candidates5Block2 : List Nat := [20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239, 24439, 24473, 24517, 24763, 24781, 24799, 24847, 24889, 24953, 25153, 25189, 25219, 25229, 25301, 25343, 25349, 25541, 25747]
set_option maxRecDepth 4096 in
theorem root0Candidates5Block2_shape : root0Tail262 = root0Candidates5Block2 ++ root0Tail335 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates5Block2_checked : root0Candidates5Block2.filter (SortedNatTree.contains FastRows.row59) = [22067, 22349] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates5Block2_step : candidates edge 4 3 59 root0Tail262 seed.bound =
    [22067, 22349] ++ candidates edge 4 3 59 root0Tail335 seed.bound := by
  rw [root0Candidates5Block2_shape, candidates_append_of_within edge 4 3 59 root0Candidates5Block2 root0Tail335 seed.bound (by decide)]
  have row : edge 59 = (SortedNatTree.contains FastRows.row59) := by
    change Data.row59.contains = (SortedNatTree.contains FastRows.row59)
    exact FastRows.row59_checked.symm
  rw [row, root0Candidates5Block2_checked]
theorem root0Candidates5_end : candidates edge 4 3 59 root0Tail335 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates5_checked : candidates edge 4 3 59 root0Tail6 seed.bound = root0Candidates5 := by
  rw [root0Candidates5Block0_step, root0Candidates5Block1_step, root0Candidates5Block2_step, root0Candidates5_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child5 : search edge 3 [3, 59] root0Candidates5 62 seed = seed := by decide
theorem root0Step5 : scan edge (search edge 3) 4 [3] 3 root0Tail5 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail6 seed := by
  rw [root0Tail5, scan, if_neg (by decide), root0Candidates5_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail6
    (if 3 ≤ root0Candidates5.length then search edge 3 [3, 59] root0Candidates5 62 seed else seed) = _
  rw [if_pos (by decide), root0Child5]
def root0Candidates6 : List Nat := [2377, 3637, 5419, 5923, 14107, 16363, 17077, 17959, 21961, 25153]
def root0Candidates6Block0 : List Nat := [73, 109, 137, 191, 229, 271, 331, 359, 373, 449, 467, 499, 541, 557, 607, 613, 617, 673, 701, 719, 733, 739, 823, 929, 947, 1013, 1019, 1033, 1051, 1181, 1193, 1237, 1481, 1531, 1607, 1627, 1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181]
set_option maxRecDepth 4096 in
theorem root0Candidates6Block0_shape : root0Tail7 = root0Candidates6Block0 ++ root0Tail135 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates6Block0_checked : root0Candidates6Block0.filter (SortedNatTree.contains FastRows.row67) = [2377, 3637, 5419, 5923] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates6Block0_step : candidates edge 4 3 67 root0Tail7 seed.bound =
    [2377, 3637, 5419, 5923] ++ candidates edge 4 3 67 root0Tail135 seed.bound := by
  rw [root0Candidates6Block0_shape, candidates_append_of_within edge 4 3 67 root0Candidates6Block0 root0Tail135 seed.bound (by decide)]
  have row : edge 67 = (SortedNatTree.contains FastRows.row67) := by
    change Data.row67.contains = (SortedNatTree.contains FastRows.row67)
    exact FastRows.row67_checked.symm
  rw [row, root0Candidates6Block0_checked]
def root0Candidates6Block1 : List Nat := [9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627]
set_option maxRecDepth 4096 in
theorem root0Candidates6Block1_shape : root0Tail135 = root0Candidates6Block1 ++ root0Tail263 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates6Block1_checked : root0Candidates6Block1.filter (SortedNatTree.contains FastRows.row67) = [14107, 16363, 17077, 17959] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates6Block1_step : candidates edge 4 3 67 root0Tail135 seed.bound =
    [14107, 16363, 17077, 17959] ++ candidates edge 4 3 67 root0Tail263 seed.bound := by
  rw [root0Candidates6Block1_shape, candidates_append_of_within edge 4 3 67 root0Candidates6Block1 root0Tail263 seed.bound (by decide)]
  have row : edge 67 = (SortedNatTree.contains FastRows.row67) := by
    change Data.row67.contains = (SortedNatTree.contains FastRows.row67)
    exact FastRows.row67_checked.symm
  rw [row, root0Candidates6Block1_checked]
def root0Candidates6Block2 : List Nat := [20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239, 24439, 24473, 24517, 24763, 24781, 24799, 24847, 24889, 24953, 25153, 25189, 25219, 25229, 25301, 25343, 25349, 25541, 25747]
set_option maxRecDepth 4096 in
theorem root0Candidates6Block2_shape : root0Tail263 = root0Candidates6Block2 ++ root0Tail335 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates6Block2_checked : root0Candidates6Block2.filter (SortedNatTree.contains FastRows.row67) = [21961, 25153] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates6Block2_step : candidates edge 4 3 67 root0Tail263 seed.bound =
    [21961, 25153] ++ candidates edge 4 3 67 root0Tail335 seed.bound := by
  rw [root0Candidates6Block2_shape, candidates_append_of_within edge 4 3 67 root0Candidates6Block2 root0Tail335 seed.bound (by decide)]
  have row : edge 67 = (SortedNatTree.contains FastRows.row67) := by
    change Data.row67.contains = (SortedNatTree.contains FastRows.row67)
    exact FastRows.row67_checked.symm
  rw [row, root0Candidates6Block2_checked]
theorem root0Candidates6_end : candidates edge 4 3 67 root0Tail335 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates6_checked : candidates edge 4 3 67 root0Tail7 seed.bound = root0Candidates6 := by
  rw [root0Candidates6Block0_step, root0Candidates6Block1_step, root0Candidates6Block2_step, root0Candidates6_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child6 : search edge 3 [3, 67] root0Candidates6 70 seed = seed := by decide
theorem root0Step6 : scan edge (search edge 3) 4 [3] 3 root0Tail6 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail7 seed := by
  rw [root0Tail6, scan, if_neg (by decide), root0Candidates6_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail7
    (if 3 ≤ root0Candidates6.length then search edge 3 [3, 67] root0Candidates6 70 seed else seed) = _
  rw [if_pos (by decide), root0Child6]
def root0Candidates7 : List Nat := [607, 823, 6277, 6793, 7159, 7591, 9241, 12979, 15313, 15739, 16567, 18523, 19387, 21757]
def root0Candidates7Block0 : List Nat := [109, 137, 191, 229, 271, 331, 359, 373, 449, 467, 499, 541, 557, 607, 613, 617, 673, 701, 719, 733, 739, 823, 929, 947, 1013, 1019, 1033, 1051, 1181, 1193, 1237, 1481, 1531, 1607, 1627, 1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241]
set_option maxRecDepth 4096 in
theorem root0Candidates7Block0_shape : root0Tail8 = root0Candidates7Block0 ++ root0Tail136 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates7Block0_checked : root0Candidates7Block0.filter (SortedNatTree.contains FastRows.row73) = [607, 823, 6277, 6793, 7159, 7591, 9241] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates7Block0_step : candidates edge 4 3 73 root0Tail8 seed.bound =
    [607, 823, 6277, 6793, 7159, 7591, 9241] ++ candidates edge 4 3 73 root0Tail136 seed.bound := by
  rw [root0Candidates7Block0_shape, candidates_append_of_within edge 4 3 73 root0Candidates7Block0 root0Tail136 seed.bound (by decide)]
  have row : edge 73 = (SortedNatTree.contains FastRows.row73) := by
    change Data.row73.contains = (SortedNatTree.contains FastRows.row73)
    exact FastRows.row73_checked.symm
  rw [row, root0Candidates7Block0_checked]
def root0Candidates7Block1 : List Nat := [9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693]
set_option maxRecDepth 4096 in
theorem root0Candidates7Block1_shape : root0Tail136 = root0Candidates7Block1 ++ root0Tail264 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates7Block1_checked : root0Candidates7Block1.filter (SortedNatTree.contains FastRows.row73) = [12979, 15313, 15739, 16567, 18523, 19387] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates7Block1_step : candidates edge 4 3 73 root0Tail136 seed.bound =
    [12979, 15313, 15739, 16567, 18523, 19387] ++ candidates edge 4 3 73 root0Tail264 seed.bound := by
  rw [root0Candidates7Block1_shape, candidates_append_of_within edge 4 3 73 root0Candidates7Block1 root0Tail264 seed.bound (by decide)]
  have row : edge 73 = (SortedNatTree.contains FastRows.row73) := by
    change Data.row73.contains = (SortedNatTree.contains FastRows.row73)
    exact FastRows.row73_checked.symm
  rw [row, root0Candidates7Block1_checked]
def root0Candidates7Block2 : List Nat := [20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239, 24439, 24473, 24517, 24763, 24781, 24799, 24847, 24889, 24953, 25153, 25189, 25219, 25229, 25301, 25343, 25349, 25541, 25747]
set_option maxRecDepth 4096 in
theorem root0Candidates7Block2_shape : root0Tail264 = root0Candidates7Block2 ++ root0Tail335 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates7Block2_checked : root0Candidates7Block2.filter (SortedNatTree.contains FastRows.row73) = [21757] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates7Block2_step : candidates edge 4 3 73 root0Tail264 seed.bound =
    [21757] ++ candidates edge 4 3 73 root0Tail335 seed.bound := by
  rw [root0Candidates7Block2_shape, candidates_append_of_within edge 4 3 73 root0Candidates7Block2 root0Tail335 seed.bound (by decide)]
  have row : edge 73 = (SortedNatTree.contains FastRows.row73) := by
    change Data.row73.contains = (SortedNatTree.contains FastRows.row73)
    exact FastRows.row73_checked.symm
  rw [row, root0Candidates7Block2_checked]
theorem root0Candidates7_end : candidates edge 4 3 73 root0Tail335 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates7_checked : candidates edge 4 3 73 root0Tail8 seed.bound = root0Candidates7 := by
  rw [root0Candidates7Block0_step, root0Candidates7Block1_step, root0Candidates7Block2_step, root0Candidates7_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child7 : search edge 3 [3, 73] root0Candidates7 76 seed = seed := by decide
theorem root0Step7 : scan edge (search edge 3) 4 [3] 3 root0Tail7 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail8 seed := by
  rw [root0Tail7, scan, if_neg (by decide), root0Candidates7_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail8
    (if 3 ≤ root0Candidates7.length then search edge 3 [3, 73] root0Candidates7 76 seed else seed) = _
  rw [if_pos (by decide), root0Child7]
def root0Candidates8 : List Nat := [673, 14401, 15493, 18289, 21961, 25153]
def root0Candidates8Block0 : List Nat := [137, 191, 229, 271, 331, 359, 373, 449, 467, 499, 541, 557, 607, 613, 617, 673, 701, 719, 733, 739, 823, 929, 947, 1013, 1019, 1033, 1051, 1181, 1193, 1237, 1481, 1531, 1607, 1627, 1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461]
set_option maxRecDepth 4096 in
theorem root0Candidates8Block0_shape : root0Tail9 = root0Candidates8Block0 ++ root0Tail137 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates8Block0_checked : root0Candidates8Block0.filter (SortedNatTree.contains FastRows.row109) = [673] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates8Block0_step : candidates edge 4 3 109 root0Tail9 seed.bound =
    [673] ++ candidates edge 4 3 109 root0Tail137 seed.bound := by
  rw [root0Candidates8Block0_shape, candidates_append_of_within edge 4 3 109 root0Candidates8Block0 root0Tail137 seed.bound (by decide)]
  have row : edge 109 = (SortedNatTree.contains FastRows.row109) := by
    change Data.row109.contains = (SortedNatTree.contains FastRows.row109)
    exact FastRows.row109_checked.symm
  rw [row, root0Candidates8Block0_checked]
def root0Candidates8Block1 : List Nat := [9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759]
set_option maxRecDepth 4096 in
theorem root0Candidates8Block1_shape : root0Tail137 = root0Candidates8Block1 ++ root0Tail265 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates8Block1_checked : root0Candidates8Block1.filter (SortedNatTree.contains FastRows.row109) = [14401, 15493, 18289] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates8Block1_step : candidates edge 4 3 109 root0Tail137 seed.bound =
    [14401, 15493, 18289] ++ candidates edge 4 3 109 root0Tail265 seed.bound := by
  rw [root0Candidates8Block1_shape, candidates_append_of_within edge 4 3 109 root0Candidates8Block1 root0Tail265 seed.bound (by decide)]
  have row : edge 109 = (SortedNatTree.contains FastRows.row109) := by
    change Data.row109.contains = (SortedNatTree.contains FastRows.row109)
    exact FastRows.row109_checked.symm
  rw [row, root0Candidates8Block1_checked]
def root0Candidates8Block2 : List Nat := [20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239, 24439, 24473, 24517, 24763, 24781, 24799, 24847, 24889, 24953, 25153, 25189, 25219, 25229, 25301, 25343, 25349, 25541, 25747]
set_option maxRecDepth 4096 in
theorem root0Candidates8Block2_shape : root0Tail265 = root0Candidates8Block2 ++ root0Tail335 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates8Block2_checked : root0Candidates8Block2.filter (SortedNatTree.contains FastRows.row109) = [21961, 25153] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates8Block2_step : candidates edge 4 3 109 root0Tail265 seed.bound =
    [21961, 25153] ++ candidates edge 4 3 109 root0Tail335 seed.bound := by
  rw [root0Candidates8Block2_shape, candidates_append_of_within edge 4 3 109 root0Candidates8Block2 root0Tail335 seed.bound (by decide)]
  have row : edge 109 = (SortedNatTree.contains FastRows.row109) := by
    change Data.row109.contains = (SortedNatTree.contains FastRows.row109)
    exact FastRows.row109_checked.symm
  rw [row, root0Candidates8Block2_checked]
theorem root0Candidates8_end : candidates edge 4 3 109 root0Tail335 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates8_checked : candidates edge 4 3 109 root0Tail9 seed.bound = root0Candidates8 := by
  rw [root0Candidates8Block0_step, root0Candidates8Block1_step, root0Candidates8Block2_step, root0Candidates8_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child8 : search edge 3 [3, 109] root0Candidates8 112 seed = seed := by decide
theorem root0Step8 : scan edge (search edge 3) 4 [3] 3 root0Tail8 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail9 seed := by
  rw [root0Tail8, scan, if_neg (by decide), root0Candidates8_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail9
    (if 3 ≤ root0Candidates8.length then search edge 3 [3, 109] root0Candidates8 112 seed else seed) = _
  rw [if_pos (by decide), root0Child8]
def root0Candidates9 : List Nat := [191, 359, 947, 8219, 9857, 12269, 17489, 17789, 21143, 21569, 22067]
def root0Candidates9Block0 : List Nat := [191, 229, 271, 331, 359, 373, 449, 467, 499, 541, 557, 607, 613, 617, 673, 701, 719, 733, 739, 823, 929, 947, 1013, 1019, 1033, 1051, 1181, 1193, 1237, 1481, 1531, 1607, 1627, 1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521]
set_option maxRecDepth 4096 in
theorem root0Candidates9Block0_shape : root0Tail10 = root0Candidates9Block0 ++ root0Tail138 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates9Block0_checked : root0Candidates9Block0.filter (SortedNatTree.contains FastRows.row137) = [191, 359, 947, 8219] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates9Block0_step : candidates edge 4 3 137 root0Tail10 seed.bound =
    [191, 359, 947, 8219] ++ candidates edge 4 3 137 root0Tail138 seed.bound := by
  rw [root0Candidates9Block0_shape, candidates_append_of_within edge 4 3 137 root0Candidates9Block0 root0Tail138 seed.bound (by decide)]
  have row : edge 137 = (SortedNatTree.contains FastRows.row137) := by
    change Data.row137.contains = (SortedNatTree.contains FastRows.row137)
    exact FastRows.row137_checked.symm
  rw [row, root0Candidates9Block0_checked]
def root0Candidates9Block1 : List Nat := [9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899]
set_option maxRecDepth 4096 in
theorem root0Candidates9Block1_shape : root0Tail138 = root0Candidates9Block1 ++ root0Tail266 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates9Block1_checked : root0Candidates9Block1.filter (SortedNatTree.contains FastRows.row137) = [9857, 12269, 17489, 17789] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates9Block1_step : candidates edge 4 3 137 root0Tail138 seed.bound =
    [9857, 12269, 17489, 17789] ++ candidates edge 4 3 137 root0Tail266 seed.bound := by
  rw [root0Candidates9Block1_shape, candidates_append_of_within edge 4 3 137 root0Candidates9Block1 root0Tail266 seed.bound (by decide)]
  have row : edge 137 = (SortedNatTree.contains FastRows.row137) := by
    change Data.row137.contains = (SortedNatTree.contains FastRows.row137)
    exact FastRows.row137_checked.symm
  rw [row, root0Candidates9Block1_checked]
def root0Candidates9Block2 : List Nat := [20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239, 24439, 24473, 24517, 24763, 24781, 24799, 24847, 24889, 24953, 25153, 25189, 25219, 25229, 25301, 25343, 25349, 25541, 25747]
set_option maxRecDepth 4096 in
theorem root0Candidates9Block2_shape : root0Tail266 = root0Candidates9Block2 ++ root0Tail335 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates9Block2_checked : root0Candidates9Block2.filter (SortedNatTree.contains FastRows.row137) = [21143, 21569, 22067] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates9Block2_step : candidates edge 4 3 137 root0Tail266 seed.bound =
    [21143, 21569, 22067] ++ candidates edge 4 3 137 root0Tail335 seed.bound := by
  rw [root0Candidates9Block2_shape, candidates_append_of_within edge 4 3 137 root0Candidates9Block2 root0Tail335 seed.bound (by decide)]
  have row : edge 137 = (SortedNatTree.contains FastRows.row137) := by
    change Data.row137.contains = (SortedNatTree.contains FastRows.row137)
    exact FastRows.row137_checked.symm
  rw [row, root0Candidates9Block2_checked]
theorem root0Candidates9_end : candidates edge 4 3 137 root0Tail335 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates9_checked : candidates edge 4 3 137 root0Tail10 seed.bound = root0Candidates9 := by
  rw [root0Candidates9Block0_step, root0Candidates9Block1_step, root0Candidates9Block2_step, root0Candidates9_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child9 : search edge 3 [3, 137] root0Candidates9 140 seed = seed := by decide
theorem root0Step9 : scan edge (search edge 3) 4 [3] 3 root0Tail9 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail10 seed := by
  rw [root0Tail9, scan, if_neg (by decide), root0Candidates9_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail10
    (if 3 ≤ root0Candidates9.length then search edge 3 [3, 137] root0Candidates9 140 seed else seed) = _
  rw [if_pos (by decide), root0Child9]
def root0Candidates10 : List Nat := [2069, 2213, 5099, 6599, 8237, 8747, 9461, 22079, 22613, 24239]
def root0Candidates10Block0 : List Nat := [229, 271, 331, 359, 373, 449, 467, 499, 541, 557, 607, 613, 617, 673, 701, 719, 733, 739, 823, 929, 947, 1013, 1019, 1033, 1051, 1181, 1193, 1237, 1481, 1531, 1607, 1627, 1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623]
set_option maxRecDepth 4096 in
theorem root0Candidates10Block0_shape : root0Tail11 = root0Candidates10Block0 ++ root0Tail139 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates10Block0_checked : root0Candidates10Block0.filter (SortedNatTree.contains FastRows.row191) = [2069, 2213, 5099, 6599, 8237, 8747, 9461] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates10Block0_step : candidates edge 4 3 191 root0Tail11 seed.bound =
    [2069, 2213, 5099, 6599, 8237, 8747, 9461] ++ candidates edge 4 3 191 root0Tail139 seed.bound := by
  rw [root0Candidates10Block0_shape, candidates_append_of_within edge 4 3 191 root0Candidates10Block0 root0Tail139 seed.bound (by decide)]
  have row : edge 191 = (SortedNatTree.contains FastRows.row191) := by
    change Data.row191.contains = (SortedNatTree.contains FastRows.row191)
    exact FastRows.row191_checked.symm
  rw [row, root0Candidates10Block0_checked]
def root0Candidates10Block1 : List Nat := [9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939]
set_option maxRecDepth 4096 in
theorem root0Candidates10Block1_shape : root0Tail139 = root0Candidates10Block1 ++ root0Tail267 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates10Block1_checked : root0Candidates10Block1.filter (SortedNatTree.contains FastRows.row191) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates10Block1_step : candidates edge 4 3 191 root0Tail139 seed.bound =
    [] ++ candidates edge 4 3 191 root0Tail267 seed.bound := by
  rw [root0Candidates10Block1_shape, candidates_append_of_within edge 4 3 191 root0Candidates10Block1 root0Tail267 seed.bound (by decide)]
  have row : edge 191 = (SortedNatTree.contains FastRows.row191) := by
    change Data.row191.contains = (SortedNatTree.contains FastRows.row191)
    exact FastRows.row191_checked.symm
  rw [row, root0Candidates10Block1_checked]
def root0Candidates10Block2 : List Nat := [21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239, 24439, 24473, 24517, 24763, 24781, 24799, 24847, 24889, 24953, 25153, 25189, 25219, 25229, 25301, 25343, 25349, 25541, 25747]
set_option maxRecDepth 4096 in
theorem root0Candidates10Block2_shape : root0Tail267 = root0Candidates10Block2 ++ root0Tail335 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates10Block2_checked : root0Candidates10Block2.filter (SortedNatTree.contains FastRows.row191) = [22079, 22613, 24239] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates10Block2_step : candidates edge 4 3 191 root0Tail267 seed.bound =
    [22079, 22613, 24239] ++ candidates edge 4 3 191 root0Tail335 seed.bound := by
  rw [root0Candidates10Block2_shape, candidates_append_of_within edge 4 3 191 root0Candidates10Block2 root0Tail335 seed.bound (by decide)]
  have row : edge 191 = (SortedNatTree.contains FastRows.row191) := by
    change Data.row191.contains = (SortedNatTree.contains FastRows.row191)
    exact FastRows.row191_checked.symm
  rw [row, root0Candidates10Block2_checked]
theorem root0Candidates10_end : candidates edge 4 3 191 root0Tail335 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates10_checked : candidates edge 4 3 191 root0Tail11 seed.bound = root0Candidates10 := by
  rw [root0Candidates10Block0_step, root0Candidates10Block1_step, root0Candidates10Block2_step, root0Candidates10_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child10 : search edge 3 [3, 191] root0Candidates10 194 seed = seed := by decide
theorem root0Step10 : scan edge (search edge 3) 4 [3] 3 root0Tail10 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail11 seed := by
  rw [root0Tail10, scan, if_neg (by decide), root0Candidates10_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail11
    (if 3 ≤ root0Candidates10.length then search edge 3 [3, 191] root0Candidates10 194 seed else seed) = _
  rw [if_pos (by decide), root0Child10]
def root0Candidates11 : List Nat := [373, 499, 613, 3301, 5869, 7591, 10243, 16339]
def root0Candidates11Block0 : List Nat := [271, 331, 359, 373, 449, 467, 499, 541, 557, 607, 613, 617, 673, 701, 719, 733, 739, 823, 929, 947, 1013, 1019, 1033, 1051, 1181, 1193, 1237, 1481, 1531, 1607, 1627, 1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857]
set_option maxRecDepth 4096 in
theorem root0Candidates11Block0_shape : root0Tail12 = root0Candidates11Block0 ++ root0Tail140 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates11Block0_checked : root0Candidates11Block0.filter (SortedNatTree.contains FastRows.row229) = [373, 499, 613, 3301, 5869, 7591] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates11Block0_step : candidates edge 4 3 229 root0Tail12 seed.bound =
    [373, 499, 613, 3301, 5869, 7591] ++ candidates edge 4 3 229 root0Tail140 seed.bound := by
  rw [root0Candidates11Block0_shape, candidates_append_of_within edge 4 3 229 root0Candidates11Block0 root0Tail140 seed.bound (by decide)]
  have row : edge 229 = (SortedNatTree.contains FastRows.row229) := by
    change Data.row229.contains = (SortedNatTree.contains FastRows.row229)
    exact FastRows.row229_checked.symm
  rw [row, root0Candidates11Block0_checked]
def root0Candidates11Block1 : List Nat := [9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017]
set_option maxRecDepth 4096 in
theorem root0Candidates11Block1_shape : root0Tail140 = root0Candidates11Block1 ++ root0Tail268 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates11Block1_checked : root0Candidates11Block1.filter (SortedNatTree.contains FastRows.row229) = [10243, 16339] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates11Block1_step : candidates edge 4 3 229 root0Tail140 seed.bound =
    [10243, 16339] ++ candidates edge 4 3 229 root0Tail268 seed.bound := by
  rw [root0Candidates11Block1_shape, candidates_append_of_within edge 4 3 229 root0Candidates11Block1 root0Tail268 seed.bound (by decide)]
  have row : edge 229 = (SortedNatTree.contains FastRows.row229) := by
    change Data.row229.contains = (SortedNatTree.contains FastRows.row229)
    exact FastRows.row229_checked.symm
  rw [row, root0Candidates11Block1_checked]
def root0Candidates11Block2 : List Nat := [21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239, 24439, 24473, 24517, 24763, 24781, 24799, 24847, 24889, 24953, 25153, 25189, 25219, 25229, 25301, 25343, 25349, 25541, 25747]
set_option maxRecDepth 4096 in
theorem root0Candidates11Block2_shape : root0Tail268 = root0Candidates11Block2 ++ root0Tail335 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates11Block2_checked : root0Candidates11Block2.filter (SortedNatTree.contains FastRows.row229) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates11Block2_step : candidates edge 4 3 229 root0Tail268 seed.bound =
    [] ++ candidates edge 4 3 229 root0Tail335 seed.bound := by
  rw [root0Candidates11Block2_shape, candidates_append_of_within edge 4 3 229 root0Candidates11Block2 root0Tail335 seed.bound (by decide)]
  have row : edge 229 = (SortedNatTree.contains FastRows.row229) := by
    change Data.row229.contains = (SortedNatTree.contains FastRows.row229)
    exact FastRows.row229_checked.symm
  rw [row, root0Candidates11Block2_checked]
theorem root0Candidates11_end : candidates edge 4 3 229 root0Tail335 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates11_checked : candidates edge 4 3 229 root0Tail12 seed.bound = root0Candidates11 := by
  rw [root0Candidates11Block0_step, root0Candidates11Block1_step, root0Candidates11Block2_step, root0Candidates11_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child11 : search edge 3 [3, 229] root0Candidates11 232 seed = seed := by decide
theorem root0Step11 : scan edge (search edge 3) 4 [3] 3 root0Tail11 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail12 seed := by
  rw [root0Tail11, scan, if_neg (by decide), root0Candidates11_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail12
    (if 3 ≤ root0Candidates11.length then search edge 3 [3, 229] root0Candidates11 232 seed else seed) = _
  rw [if_pos (by decide), root0Child11]
def root0Candidates12 : List Nat := [6451, 12343, 16759, 19237, 22549]
def root0Candidates12Block0 : List Nat := [331, 359, 373, 449, 467, 499, 541, 557, 607, 613, 617, 673, 701, 719, 733, 739, 823, 929, 947, 1013, 1019, 1033, 1051, 1181, 1193, 1237, 1481, 1531, 1607, 1627, 1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887]
set_option maxRecDepth 4096 in
theorem root0Candidates12Block0_shape : root0Tail13 = root0Candidates12Block0 ++ root0Tail141 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates12Block0_checked : root0Candidates12Block0.filter (SortedNatTree.contains FastRows.row271) = [6451] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates12Block0_step : candidates edge 4 3 271 root0Tail13 seed.bound =
    [6451] ++ candidates edge 4 3 271 root0Tail141 seed.bound := by
  rw [root0Candidates12Block0_shape, candidates_append_of_within edge 4 3 271 root0Candidates12Block0 root0Tail141 seed.bound (by decide)]
  have row : edge 271 = (SortedNatTree.contains FastRows.row271) := by
    change Data.row271.contains = (SortedNatTree.contains FastRows.row271)
    exact FastRows.row271_checked.symm
  rw [row, root0Candidates12Block0_checked]
def root0Candidates12Block1 : List Nat := [9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143]
set_option maxRecDepth 4096 in
theorem root0Candidates12Block1_shape : root0Tail141 = root0Candidates12Block1 ++ root0Tail269 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates12Block1_checked : root0Candidates12Block1.filter (SortedNatTree.contains FastRows.row271) = [12343, 16759, 19237] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates12Block1_step : candidates edge 4 3 271 root0Tail141 seed.bound =
    [12343, 16759, 19237] ++ candidates edge 4 3 271 root0Tail269 seed.bound := by
  rw [root0Candidates12Block1_shape, candidates_append_of_within edge 4 3 271 root0Candidates12Block1 root0Tail269 seed.bound (by decide)]
  have row : edge 271 = (SortedNatTree.contains FastRows.row271) := by
    change Data.row271.contains = (SortedNatTree.contains FastRows.row271)
    exact FastRows.row271_checked.symm
  rw [row, root0Candidates12Block1_checked]
def root0Candidates12Block2 : List Nat := [21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239, 24439, 24473, 24517, 24763, 24781, 24799, 24847, 24889, 24953, 25153, 25189, 25219, 25229, 25301, 25343, 25349, 25541, 25747]
set_option maxRecDepth 4096 in
theorem root0Candidates12Block2_shape : root0Tail269 = root0Candidates12Block2 ++ root0Tail335 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates12Block2_checked : root0Candidates12Block2.filter (SortedNatTree.contains FastRows.row271) = [22549] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates12Block2_step : candidates edge 4 3 271 root0Tail269 seed.bound =
    [22549] ++ candidates edge 4 3 271 root0Tail335 seed.bound := by
  rw [root0Candidates12Block2_shape, candidates_append_of_within edge 4 3 271 root0Candidates12Block2 root0Tail335 seed.bound (by decide)]
  have row : edge 271 = (SortedNatTree.contains FastRows.row271) := by
    change Data.row271.contains = (SortedNatTree.contains FastRows.row271)
    exact FastRows.row271_checked.symm
  rw [row, root0Candidates12Block2_checked]
theorem root0Candidates12_end : candidates edge 4 3 271 root0Tail335 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates12_checked : candidates edge 4 3 271 root0Tail13 seed.bound = root0Candidates12 := by
  rw [root0Candidates12Block0_step, root0Candidates12Block1_step, root0Candidates12Block2_step, root0Candidates12_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child12 : search edge 3 [3, 271] root0Candidates12 274 seed = seed := by decide
theorem root0Step12 : scan edge (search edge 3) 4 [3] 3 root0Tail12 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail13 seed := by
  rw [root0Tail12, scan, if_neg (by decide), root0Candidates12_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail13
    (if 3 ≤ root0Candidates12.length then search edge 3 [3, 271] root0Candidates12 274 seed else seed) = _
  rw [if_pos (by decide), root0Child12]
def root0Candidates13 : List Nat := [739, 1531, 3637, 3769, 6451, 8377, 8431, 10771, 15373, 17599, 21169, 24799]
def root0Candidates13Block0 : List Nat := [359, 373, 449, 467, 499, 541, 557, 607, 613, 617, 673, 701, 719, 733, 739, 823, 929, 947, 1013, 1019, 1033, 1051, 1181, 1193, 1237, 1481, 1531, 1607, 1627, 1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901]
set_option maxRecDepth 4096 in
theorem root0Candidates13Block0_shape : root0Tail14 = root0Candidates13Block0 ++ root0Tail142 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates13Block0_checked : root0Candidates13Block0.filter (SortedNatTree.contains FastRows.row331) = [739, 1531, 3637, 3769, 6451, 8377, 8431] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates13Block0_step : candidates edge 4 3 331 root0Tail14 seed.bound =
    [739, 1531, 3637, 3769, 6451, 8377, 8431] ++ candidates edge 4 3 331 root0Tail142 seed.bound := by
  rw [root0Candidates13Block0_shape, candidates_append_of_within edge 4 3 331 root0Candidates13Block0 root0Tail142 seed.bound (by decide)]
  have row : edge 331 = (SortedNatTree.contains FastRows.row331) := by
    change Data.row331.contains = (SortedNatTree.contains FastRows.row331)
    exact FastRows.row331_checked.symm
  rw [row, root0Candidates13Block0_checked]
def root0Candidates13Block1 : List Nat := [10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169]
set_option maxRecDepth 4096 in
theorem root0Candidates13Block1_shape : root0Tail142 = root0Candidates13Block1 ++ root0Tail270 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates13Block1_checked : root0Candidates13Block1.filter (SortedNatTree.contains FastRows.row331) = [10771, 15373, 17599, 21169] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates13Block1_step : candidates edge 4 3 331 root0Tail142 seed.bound =
    [10771, 15373, 17599, 21169] ++ candidates edge 4 3 331 root0Tail270 seed.bound := by
  rw [root0Candidates13Block1_shape, candidates_append_of_within edge 4 3 331 root0Candidates13Block1 root0Tail270 seed.bound (by decide)]
  have row : edge 331 = (SortedNatTree.contains FastRows.row331) := by
    change Data.row331.contains = (SortedNatTree.contains FastRows.row331)
    exact FastRows.row331_checked.symm
  rw [row, root0Candidates13Block1_checked]
def root0Candidates13Block2 : List Nat := [21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239, 24439, 24473, 24517, 24763, 24781, 24799, 24847, 24889, 24953, 25153, 25189, 25219, 25229, 25301, 25343, 25349, 25541]
set_option maxRecDepth 4096 in
theorem root0Candidates13Block2_shape : root0Tail270 = root0Candidates13Block2 ++ root0Tail334 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates13Block2_checked : root0Candidates13Block2.filter (SortedNatTree.contains FastRows.row331) = [24799] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates13Block2_step : candidates edge 4 3 331 root0Tail270 seed.bound =
    [24799] ++ candidates edge 4 3 331 root0Tail334 seed.bound := by
  rw [root0Candidates13Block2_shape, candidates_append_of_within edge 4 3 331 root0Candidates13Block2 root0Tail334 seed.bound (by decide)]
  have row : edge 331 = (SortedNatTree.contains FastRows.row331) := by
    change Data.row331.contains = (SortedNatTree.contains FastRows.row331)
    exact FastRows.row331_checked.symm
  rw [row, root0Candidates13Block2_checked]
theorem root0Candidates13_end : candidates edge 4 3 331 root0Tail334 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates13_checked : candidates edge 4 3 331 root0Tail14 seed.bound = root0Candidates13 := by
  rw [root0Candidates13Block0_step, root0Candidates13Block1_step, root0Candidates13Block2_step, root0Candidates13_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child13 : search edge 3 [3, 331] root0Candidates13 334 seed = seed := by decide
theorem root0Step13 : scan edge (search edge 3) 4 [3] 3 root0Tail13 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail14 seed := by
  rw [root0Tail13, scan, if_neg (by decide), root0Candidates13_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail14
    (if 3 ≤ root0Candidates13.length then search edge 3 [3, 331] root0Candidates13 334 seed else seed) = _
  rw [if_pos (by decide), root0Child13]
def root0Candidates14 : List Nat := [701, 1667, 9887, 10169, 11177, 13763, 23201]
def root0Candidates14Block0 : List Nat := [373, 449, 467, 499, 541, 557, 607, 613, 617, 673, 701, 719, 733, 739, 823, 929, 947, 1013, 1019, 1033, 1051, 1181, 1193, 1237, 1481, 1531, 1607, 1627, 1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091]
set_option maxRecDepth 4096 in
theorem root0Candidates14Block0_shape : root0Tail15 = root0Candidates14Block0 ++ root0Tail143 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates14Block0_checked : root0Candidates14Block0.filter (SortedNatTree.contains FastRows.row359) = [701, 1667, 9887] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates14Block0_step : candidates edge 4 3 359 root0Tail15 seed.bound =
    [701, 1667, 9887] ++ candidates edge 4 3 359 root0Tail143 seed.bound := by
  rw [root0Candidates14Block0_shape, candidates_append_of_within edge 4 3 359 root0Candidates14Block0 root0Tail143 seed.bound (by decide)]
  have row : edge 359 = (SortedNatTree.contains FastRows.row359) := by
    change Data.row359.contains = (SortedNatTree.contains FastRows.row359)
    exact FastRows.row359_checked.symm
  rw [row, root0Candidates14Block0_checked]
def root0Candidates14Block1 : List Nat := [10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187]
set_option maxRecDepth 4096 in
theorem root0Candidates14Block1_shape : root0Tail143 = root0Candidates14Block1 ++ root0Tail271 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates14Block1_checked : root0Candidates14Block1.filter (SortedNatTree.contains FastRows.row359) = [10169, 11177, 13763] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates14Block1_step : candidates edge 4 3 359 root0Tail143 seed.bound =
    [10169, 11177, 13763] ++ candidates edge 4 3 359 root0Tail271 seed.bound := by
  rw [root0Candidates14Block1_shape, candidates_append_of_within edge 4 3 359 root0Candidates14Block1 root0Tail271 seed.bound (by decide)]
  have row : edge 359 = (SortedNatTree.contains FastRows.row359) := by
    change Data.row359.contains = (SortedNatTree.contains FastRows.row359)
    exact FastRows.row359_checked.symm
  rw [row, root0Candidates14Block1_checked]
def root0Candidates14Block2 : List Nat := [21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239, 24439, 24473, 24517, 24763, 24781, 24799, 24847, 24889, 24953, 25153, 25189, 25219, 25229, 25301, 25343, 25349, 25541]
set_option maxRecDepth 4096 in
theorem root0Candidates14Block2_shape : root0Tail271 = root0Candidates14Block2 ++ root0Tail334 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates14Block2_checked : root0Candidates14Block2.filter (SortedNatTree.contains FastRows.row359) = [23201] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates14Block2_step : candidates edge 4 3 359 root0Tail271 seed.bound =
    [23201] ++ candidates edge 4 3 359 root0Tail334 seed.bound := by
  rw [root0Candidates14Block2_shape, candidates_append_of_within edge 4 3 359 root0Candidates14Block2 root0Tail334 seed.bound (by decide)]
  have row : edge 359 = (SortedNatTree.contains FastRows.row359) := by
    change Data.row359.contains = (SortedNatTree.contains FastRows.row359)
    exact FastRows.row359_checked.symm
  rw [row, root0Candidates14Block2_checked]
theorem root0Candidates14_end : candidates edge 4 3 359 root0Tail334 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates14_checked : candidates edge 4 3 359 root0Tail15 seed.bound = root0Candidates14 := by
  rw [root0Candidates14Block0_step, root0Candidates14Block1_step, root0Candidates14Block2_step, root0Candidates14_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child14 : search edge 3 [3, 359] root0Candidates14 362 seed = seed := by decide
theorem root0Step14 : scan edge (search edge 3) 4 [3] 3 root0Tail14 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail15 seed := by
  rw [root0Tail14, scan, if_neg (by decide), root0Candidates14_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail15
    (if 3 ≤ root0Candidates14.length then search edge 3 [3, 359] root0Candidates14 362 seed else seed) = _
  rw [if_pos (by decide), root0Child14]
def root0Candidates15 : List Nat := [823, 1237, 1531, 5449, 6793, 7057, 10459, 15559, 18289, 19993, 23371, 24799]
def root0Candidates15Block0 : List Nat := [449, 467, 499, 541, 557, 607, 613, 617, 673, 701, 719, 733, 739, 823, 929, 947, 1013, 1019, 1033, 1051, 1181, 1193, 1237, 1481, 1531, 1607, 1627, 1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111]
set_option maxRecDepth 4096 in
theorem root0Candidates15Block0_shape : root0Tail16 = root0Candidates15Block0 ++ root0Tail144 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates15Block0_checked : root0Candidates15Block0.filter (SortedNatTree.contains FastRows.row373) = [823, 1237, 1531, 5449, 6793, 7057] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates15Block0_step : candidates edge 4 3 373 root0Tail16 seed.bound =
    [823, 1237, 1531, 5449, 6793, 7057] ++ candidates edge 4 3 373 root0Tail144 seed.bound := by
  rw [root0Candidates15Block0_shape, candidates_append_of_within edge 4 3 373 root0Candidates15Block0 root0Tail144 seed.bound (by decide)]
  have row : edge 373 = (SortedNatTree.contains FastRows.row373) := by
    change Data.row373.contains = (SortedNatTree.contains FastRows.row373)
    exact FastRows.row373_checked.symm
  rw [row, root0Candidates15Block0_checked]
def root0Candidates15Block1 : List Nat := [10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193]
set_option maxRecDepth 4096 in
theorem root0Candidates15Block1_shape : root0Tail144 = root0Candidates15Block1 ++ root0Tail272 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates15Block1_checked : root0Candidates15Block1.filter (SortedNatTree.contains FastRows.row373) = [10459, 15559, 18289, 19993] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates15Block1_step : candidates edge 4 3 373 root0Tail144 seed.bound =
    [10459, 15559, 18289, 19993] ++ candidates edge 4 3 373 root0Tail272 seed.bound := by
  rw [root0Candidates15Block1_shape, candidates_append_of_within edge 4 3 373 root0Candidates15Block1 root0Tail272 seed.bound (by decide)]
  have row : edge 373 = (SortedNatTree.contains FastRows.row373) := by
    change Data.row373.contains = (SortedNatTree.contains FastRows.row373)
    exact FastRows.row373_checked.symm
  rw [row, root0Candidates15Block1_checked]
def root0Candidates15Block2 : List Nat := [21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239, 24439, 24473, 24517, 24763, 24781, 24799, 24847, 24889, 24953, 25153, 25189, 25219, 25229, 25301, 25343, 25349, 25541]
set_option maxRecDepth 4096 in
theorem root0Candidates15Block2_shape : root0Tail272 = root0Candidates15Block2 ++ root0Tail334 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates15Block2_checked : root0Candidates15Block2.filter (SortedNatTree.contains FastRows.row373) = [23371, 24799] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates15Block2_step : candidates edge 4 3 373 root0Tail272 seed.bound =
    [23371, 24799] ++ candidates edge 4 3 373 root0Tail334 seed.bound := by
  rw [root0Candidates15Block2_shape, candidates_append_of_within edge 4 3 373 root0Candidates15Block2 root0Tail334 seed.bound (by decide)]
  have row : edge 373 = (SortedNatTree.contains FastRows.row373) := by
    change Data.row373.contains = (SortedNatTree.contains FastRows.row373)
    exact FastRows.row373_checked.symm
  rw [row, root0Candidates15Block2_checked]
theorem root0Candidates15_end : candidates edge 4 3 373 root0Tail334 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates15_checked : candidates edge 4 3 373 root0Tail16 seed.bound = root0Candidates15 := by
  rw [root0Candidates15Block0_step, root0Candidates15Block1_step, root0Candidates15Block2_step, root0Candidates15_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child15 : search edge 3 [3, 373] root0Candidates15 376 seed = seed := by decide
theorem root0Step15 : scan edge (search edge 3) 4 [3] 3 root0Tail15 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail16 seed := by
  rw [root0Tail15, scan, if_neg (by decide), root0Candidates15_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail16
    (if 3 ≤ root0Candidates15.length then search edge 3 [3, 373] root0Candidates15 376 seed else seed) = _
  rw [if_pos (by decide), root0Child15]
def root0Candidates16 : List Nat := [557, 2069, 2441, 5171, 5801, 6353, 6599, 10169, 12899, 14627, 24953]
def root0Candidates16Block0 : List Nat := [467, 499, 541, 557, 607, 613, 617, 673, 701, 719, 733, 739, 823, 929, 947, 1013, 1019, 1033, 1051, 1181, 1193, 1237, 1481, 1531, 1607, 1627, 1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169]
set_option maxRecDepth 4096 in
theorem root0Candidates16Block0_shape : root0Tail17 = root0Candidates16Block0 ++ root0Tail145 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates16Block0_checked : root0Candidates16Block0.filter (SortedNatTree.contains FastRows.row449) = [557, 2069, 2441, 5171, 5801, 6353, 6599, 10169] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates16Block0_step : candidates edge 4 3 449 root0Tail17 seed.bound =
    [557, 2069, 2441, 5171, 5801, 6353, 6599, 10169] ++ candidates edge 4 3 449 root0Tail145 seed.bound := by
  rw [root0Candidates16Block0_shape, candidates_append_of_within edge 4 3 449 root0Candidates16Block0 root0Tail145 seed.bound (by decide)]
  have row : edge 449 = (SortedNatTree.contains FastRows.row449) := by
    change Data.row449.contains = (SortedNatTree.contains FastRows.row449)
    exact FastRows.row449_checked.symm
  rw [row, root0Candidates16Block0_checked]
def root0Candidates16Block1 : List Nat := [10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313]
set_option maxRecDepth 4096 in
theorem root0Candidates16Block1_shape : root0Tail145 = root0Candidates16Block1 ++ root0Tail273 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates16Block1_checked : root0Candidates16Block1.filter (SortedNatTree.contains FastRows.row449) = [12899, 14627] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates16Block1_step : candidates edge 4 3 449 root0Tail145 seed.bound =
    [12899, 14627] ++ candidates edge 4 3 449 root0Tail273 seed.bound := by
  rw [root0Candidates16Block1_shape, candidates_append_of_within edge 4 3 449 root0Candidates16Block1 root0Tail273 seed.bound (by decide)]
  have row : edge 449 = (SortedNatTree.contains FastRows.row449) := by
    change Data.row449.contains = (SortedNatTree.contains FastRows.row449)
    exact FastRows.row449_checked.symm
  rw [row, root0Candidates16Block1_checked]
def root0Candidates16Block2 : List Nat := [21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239, 24439, 24473, 24517, 24763, 24781, 24799, 24847, 24889, 24953, 25153, 25189, 25219, 25229, 25301, 25343, 25349, 25541]
set_option maxRecDepth 4096 in
theorem root0Candidates16Block2_shape : root0Tail273 = root0Candidates16Block2 ++ root0Tail334 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates16Block2_checked : root0Candidates16Block2.filter (SortedNatTree.contains FastRows.row449) = [24953] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates16Block2_step : candidates edge 4 3 449 root0Tail273 seed.bound =
    [24953] ++ candidates edge 4 3 449 root0Tail334 seed.bound := by
  rw [root0Candidates16Block2_shape, candidates_append_of_within edge 4 3 449 root0Candidates16Block2 root0Tail334 seed.bound (by decide)]
  have row : edge 449 = (SortedNatTree.contains FastRows.row449) := by
    change Data.row449.contains = (SortedNatTree.contains FastRows.row449)
    exact FastRows.row449_checked.symm
  rw [row, root0Candidates16Block2_checked]
theorem root0Candidates16_end : candidates edge 4 3 449 root0Tail334 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates16_checked : candidates edge 4 3 449 root0Tail17 seed.bound = root0Candidates16 := by
  rw [root0Candidates16Block0_step, root0Candidates16Block1_step, root0Candidates16Block2_step, root0Candidates16_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child16 : search edge 3 [3, 449] root0Candidates16 452 seed = seed := by decide
theorem root0Step16 : scan edge (search edge 3) 4 [3] 3 root0Tail16 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail17 seed := by
  rw [root0Tail16, scan, if_neg (by decide), root0Candidates16_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail17
    (if 3 ≤ root0Candidates16.length then search edge 3 [3, 449] root0Candidates16 452 seed else seed) = _
  rw [if_pos (by decide), root0Child16]
def root0Candidates17 : List Nat := [617, 4253, 13553, 14543, 14627, 16661, 16937]
def root0Candidates17Block0 : List Nat := [499, 541, 557, 607, 613, 617, 673, 701, 719, 733, 739, 823, 929, 947, 1013, 1019, 1033, 1051, 1181, 1193, 1237, 1481, 1531, 1607, 1627, 1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223]
set_option maxRecDepth 4096 in
theorem root0Candidates17Block0_shape : root0Tail18 = root0Candidates17Block0 ++ root0Tail146 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates17Block0_checked : root0Candidates17Block0.filter (SortedNatTree.contains FastRows.row467) = [617, 4253] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates17Block0_step : candidates edge 4 3 467 root0Tail18 seed.bound =
    [617, 4253] ++ candidates edge 4 3 467 root0Tail146 seed.bound := by
  rw [root0Candidates17Block0_shape, candidates_append_of_within edge 4 3 467 root0Candidates17Block0 root0Tail146 seed.bound (by decide)]
  have row : edge 467 = (SortedNatTree.contains FastRows.row467) := by
    change Data.row467.contains = (SortedNatTree.contains FastRows.row467)
    exact FastRows.row467_checked.symm
  rw [row, root0Candidates17Block0_checked]
def root0Candidates17Block1 : List Nat := [10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319]
set_option maxRecDepth 4096 in
theorem root0Candidates17Block1_shape : root0Tail146 = root0Candidates17Block1 ++ root0Tail274 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates17Block1_checked : root0Candidates17Block1.filter (SortedNatTree.contains FastRows.row467) = [13553, 14543, 14627, 16661, 16937] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates17Block1_step : candidates edge 4 3 467 root0Tail146 seed.bound =
    [13553, 14543, 14627, 16661, 16937] ++ candidates edge 4 3 467 root0Tail274 seed.bound := by
  rw [root0Candidates17Block1_shape, candidates_append_of_within edge 4 3 467 root0Candidates17Block1 root0Tail274 seed.bound (by decide)]
  have row : edge 467 = (SortedNatTree.contains FastRows.row467) := by
    change Data.row467.contains = (SortedNatTree.contains FastRows.row467)
    exact FastRows.row467_checked.symm
  rw [row, root0Candidates17Block1_checked]
def root0Candidates17Block2 : List Nat := [21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239, 24439, 24473, 24517, 24763, 24781, 24799, 24847, 24889, 24953, 25153, 25189, 25219, 25229, 25301, 25343, 25349, 25541]
set_option maxRecDepth 4096 in
theorem root0Candidates17Block2_shape : root0Tail274 = root0Candidates17Block2 ++ root0Tail334 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates17Block2_checked : root0Candidates17Block2.filter (SortedNatTree.contains FastRows.row467) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates17Block2_step : candidates edge 4 3 467 root0Tail274 seed.bound =
    [] ++ candidates edge 4 3 467 root0Tail334 seed.bound := by
  rw [root0Candidates17Block2_shape, candidates_append_of_within edge 4 3 467 root0Candidates17Block2 root0Tail334 seed.bound (by decide)]
  have row : edge 467 = (SortedNatTree.contains FastRows.row467) := by
    change Data.row467.contains = (SortedNatTree.contains FastRows.row467)
    exact FastRows.row467_checked.symm
  rw [row, root0Candidates17Block2_checked]
theorem root0Candidates17_end : candidates edge 4 3 467 root0Tail334 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates17_checked : candidates edge 4 3 467 root0Tail18 seed.bound = root0Candidates17 := by
  rw [root0Candidates17Block0_step, root0Candidates17Block1_step, root0Candidates17Block2_step, root0Candidates17_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child17 : search edge 3 [3, 467] root0Candidates17 470 seed = seed := by decide
theorem root0Step17 : scan edge (search edge 3) 4 [3] 3 root0Tail17 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail18 seed := by
  rw [root0Tail17, scan, if_neg (by decide), root0Candidates17_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail18
    (if 3 ≤ root0Candidates17.length then search edge 3 [3, 467] root0Candidates17 470 seed else seed) = _
  rw [if_pos (by decide), root0Child17]
def root0Candidates18 : List Nat := [673, 1033, 1237, 2707, 5281, 9103, 23371, 25189, 25219]
def root0Candidates18Block0 : List Nat := [541, 557, 607, 613, 617, 673, 701, 719, 733, 739, 823, 929, 947, 1013, 1019, 1033, 1051, 1181, 1193, 1237, 1481, 1531, 1607, 1627, 1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243]
set_option maxRecDepth 4096 in
theorem root0Candidates18Block0_shape : root0Tail19 = root0Candidates18Block0 ++ root0Tail147 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates18Block0_checked : root0Candidates18Block0.filter (SortedNatTree.contains FastRows.row499) = [673, 1033, 1237, 2707, 5281, 9103] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates18Block0_step : candidates edge 4 3 499 root0Tail19 seed.bound =
    [673, 1033, 1237, 2707, 5281, 9103] ++ candidates edge 4 3 499 root0Tail147 seed.bound := by
  rw [root0Candidates18Block0_shape, candidates_append_of_within edge 4 3 499 root0Candidates18Block0 root0Tail147 seed.bound (by decide)]
  have row : edge 499 = (SortedNatTree.contains FastRows.row499) := by
    change Data.row499.contains = (SortedNatTree.contains FastRows.row499)
    exact FastRows.row499_checked.symm
  rw [row, root0Candidates18Block0_checked]
def root0Candidates18Block1 : List Nat := [10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383]
set_option maxRecDepth 4096 in
theorem root0Candidates18Block1_shape : root0Tail147 = root0Candidates18Block1 ++ root0Tail275 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates18Block1_checked : root0Candidates18Block1.filter (SortedNatTree.contains FastRows.row499) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates18Block1_step : candidates edge 4 3 499 root0Tail147 seed.bound =
    [] ++ candidates edge 4 3 499 root0Tail275 seed.bound := by
  rw [root0Candidates18Block1_shape, candidates_append_of_within edge 4 3 499 root0Candidates18Block1 root0Tail275 seed.bound (by decide)]
  have row : edge 499 = (SortedNatTree.contains FastRows.row499) := by
    change Data.row499.contains = (SortedNatTree.contains FastRows.row499)
    exact FastRows.row499_checked.symm
  rw [row, root0Candidates18Block1_checked]
def root0Candidates18Block2 : List Nat := [21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239, 24439, 24473, 24517, 24763, 24781, 24799, 24847, 24889, 24953, 25153, 25189, 25219, 25229, 25301, 25343, 25349]
set_option maxRecDepth 4096 in
theorem root0Candidates18Block2_shape : root0Tail275 = root0Candidates18Block2 ++ root0Tail333 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates18Block2_checked : root0Candidates18Block2.filter (SortedNatTree.contains FastRows.row499) = [23371, 25189, 25219] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates18Block2_step : candidates edge 4 3 499 root0Tail275 seed.bound =
    [23371, 25189, 25219] ++ candidates edge 4 3 499 root0Tail333 seed.bound := by
  rw [root0Candidates18Block2_shape, candidates_append_of_within edge 4 3 499 root0Candidates18Block2 root0Tail333 seed.bound (by decide)]
  have row : edge 499 = (SortedNatTree.contains FastRows.row499) := by
    change Data.row499.contains = (SortedNatTree.contains FastRows.row499)
    exact FastRows.row499_checked.symm
  rw [row, root0Candidates18Block2_checked]
theorem root0Candidates18_end : candidates edge 4 3 499 root0Tail333 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates18_checked : candidates edge 4 3 499 root0Tail19 seed.bound = root0Candidates18 := by
  rw [root0Candidates18Block0_step, root0Candidates18Block1_step, root0Candidates18Block2_step, root0Candidates18_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child18 : search edge 3 [3, 499] root0Candidates18 502 seed = seed := by decide
theorem root0Step18 : scan edge (search edge 3) 4 [3] 3 root0Tail18 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail19 seed := by
  rw [root0Tail18, scan, if_neg (by decide), root0Candidates18_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail19
    (if 3 ≤ root0Candidates18.length then search edge 3 [3, 499] root0Candidates18 502 seed else seed) = _
  rw [if_pos (by decide), root0Child18]
def root0Candidates19 : List Nat := [1033, 3301, 4159, 7369, 9157, 17209, 21169, 21313, 22573, 24781]
def root0Candidates19Block0 : List Nat := [557, 607, 613, 617, 673, 701, 719, 733, 739, 823, 929, 947, 1013, 1019, 1033, 1051, 1181, 1193, 1237, 1481, 1531, 1607, 1627, 1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333]
set_option maxRecDepth 4096 in
theorem root0Candidates19Block0_shape : root0Tail20 = root0Candidates19Block0 ++ root0Tail148 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates19Block0_checked : root0Candidates19Block0.filter (SortedNatTree.contains FastRows.row541) = [1033, 3301, 4159, 7369, 9157] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates19Block0_step : candidates edge 4 3 541 root0Tail20 seed.bound =
    [1033, 3301, 4159, 7369, 9157] ++ candidates edge 4 3 541 root0Tail148 seed.bound := by
  rw [root0Candidates19Block0_shape, candidates_append_of_within edge 4 3 541 root0Candidates19Block0 root0Tail148 seed.bound (by decide)]
  have row : edge 541 = (SortedNatTree.contains FastRows.row541) := by
    change Data.row541.contains = (SortedNatTree.contains FastRows.row541)
    exact FastRows.row541_checked.symm
  rw [row, root0Candidates19Block0_checked]
def root0Candidates19Block1 : List Nat := [10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397]
set_option maxRecDepth 4096 in
theorem root0Candidates19Block1_shape : root0Tail148 = root0Candidates19Block1 ++ root0Tail276 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates19Block1_checked : root0Candidates19Block1.filter (SortedNatTree.contains FastRows.row541) = [17209, 21169, 21313] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates19Block1_step : candidates edge 4 3 541 root0Tail148 seed.bound =
    [17209, 21169, 21313] ++ candidates edge 4 3 541 root0Tail276 seed.bound := by
  rw [root0Candidates19Block1_shape, candidates_append_of_within edge 4 3 541 root0Candidates19Block1 root0Tail276 seed.bound (by decide)]
  have row : edge 541 = (SortedNatTree.contains FastRows.row541) := by
    change Data.row541.contains = (SortedNatTree.contains FastRows.row541)
    exact FastRows.row541_checked.symm
  rw [row, root0Candidates19Block1_checked]
def root0Candidates19Block2 : List Nat := [21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239, 24439, 24473, 24517, 24763, 24781, 24799, 24847, 24889, 24953, 25153, 25189, 25219, 25229, 25301, 25343, 25349]
set_option maxRecDepth 4096 in
theorem root0Candidates19Block2_shape : root0Tail276 = root0Candidates19Block2 ++ root0Tail333 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates19Block2_checked : root0Candidates19Block2.filter (SortedNatTree.contains FastRows.row541) = [22573, 24781] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates19Block2_step : candidates edge 4 3 541 root0Tail276 seed.bound =
    [22573, 24781] ++ candidates edge 4 3 541 root0Tail333 seed.bound := by
  rw [root0Candidates19Block2_shape, candidates_append_of_within edge 4 3 541 root0Candidates19Block2 root0Tail333 seed.bound (by decide)]
  have row : edge 541 = (SortedNatTree.contains FastRows.row541) := by
    change Data.row541.contains = (SortedNatTree.contains FastRows.row541)
    exact FastRows.row541_checked.symm
  rw [row, root0Candidates19Block2_checked]
theorem root0Candidates19_end : candidates edge 4 3 541 root0Tail333 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates19_checked : candidates edge 4 3 541 root0Tail20 seed.bound = root0Candidates19 := by
  rw [root0Candidates19Block0_step, root0Candidates19Block1_step, root0Candidates19Block2_step, root0Candidates19_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child19 : search edge 3 [3, 541] root0Candidates19 544 seed = seed := by decide
theorem root0Step19 : scan edge (search edge 3) 4 [3] 3 root0Tail19 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail20 seed := by
  rw [root0Tail19, scan, if_neg (by decide), root0Candidates19_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail20
    (if 3 ≤ root0Candidates19.length then search edge 3 [3, 541] root0Candidates19 544 seed else seed) = _
  rw [if_pos (by decide), root0Child19]
def root0Candidates20 : List Nat := [20627]
def root0Candidates20Block0 : List Nat := [607, 613, 617, 673, 701, 719, 733, 739, 823, 929, 947, 1013, 1019, 1033, 1051, 1181, 1193, 1237, 1481, 1531, 1607, 1627, 1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357]
set_option maxRecDepth 4096 in
theorem root0Candidates20Block0_shape : root0Tail21 = root0Candidates20Block0 ++ root0Tail149 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates20Block0_checked : root0Candidates20Block0.filter (SortedNatTree.contains FastRows.row557) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates20Block0_step : candidates edge 4 3 557 root0Tail21 seed.bound =
    [] ++ candidates edge 4 3 557 root0Tail149 seed.bound := by
  rw [root0Candidates20Block0_shape, candidates_append_of_within edge 4 3 557 root0Candidates20Block0 root0Tail149 seed.bound (by decide)]
  have row : edge 557 = (SortedNatTree.contains FastRows.row557) := by
    change Data.row557.contains = (SortedNatTree.contains FastRows.row557)
    exact FastRows.row557_checked.symm
  rw [row, root0Candidates20Block0_checked]
def root0Candidates20Block1 : List Nat := [10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467]
set_option maxRecDepth 4096 in
theorem root0Candidates20Block1_shape : root0Tail149 = root0Candidates20Block1 ++ root0Tail277 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates20Block1_checked : root0Candidates20Block1.filter (SortedNatTree.contains FastRows.row557) = [20627] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates20Block1_step : candidates edge 4 3 557 root0Tail149 seed.bound =
    [20627] ++ candidates edge 4 3 557 root0Tail277 seed.bound := by
  rw [root0Candidates20Block1_shape, candidates_append_of_within edge 4 3 557 root0Candidates20Block1 root0Tail277 seed.bound (by decide)]
  have row : edge 557 = (SortedNatTree.contains FastRows.row557) := by
    change Data.row557.contains = (SortedNatTree.contains FastRows.row557)
    exact FastRows.row557_checked.symm
  rw [row, root0Candidates20Block1_checked]
def root0Candidates20Block2 : List Nat := [21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239, 24439, 24473, 24517, 24763, 24781, 24799, 24847, 24889, 24953, 25153, 25189, 25219, 25229, 25301, 25343, 25349]
set_option maxRecDepth 4096 in
theorem root0Candidates20Block2_shape : root0Tail277 = root0Candidates20Block2 ++ root0Tail333 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates20Block2_checked : root0Candidates20Block2.filter (SortedNatTree.contains FastRows.row557) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates20Block2_step : candidates edge 4 3 557 root0Tail277 seed.bound =
    [] ++ candidates edge 4 3 557 root0Tail333 seed.bound := by
  rw [root0Candidates20Block2_shape, candidates_append_of_within edge 4 3 557 root0Candidates20Block2 root0Tail333 seed.bound (by decide)]
  have row : edge 557 = (SortedNatTree.contains FastRows.row557) := by
    change Data.row557.contains = (SortedNatTree.contains FastRows.row557)
    exact FastRows.row557_checked.symm
  rw [row, root0Candidates20Block2_checked]
theorem root0Candidates20_end : candidates edge 4 3 557 root0Tail333 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates20_checked : candidates edge 4 3 557 root0Tail21 seed.bound = root0Candidates20 := by
  rw [root0Candidates20Block0_step, root0Candidates20Block1_step, root0Candidates20Block2_step, root0Candidates20_end]
  rfl
theorem root0Step20 : scan edge (search edge 3) 4 [3] 3 root0Tail20 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail21 seed := by
  rw [root0Tail20, scan, if_neg (by decide), root0Candidates20_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail21
    (if 3 ≤ root0Candidates20.length then search edge 3 [3, 557] root0Candidates20 560 seed else seed) = _
  rw [if_neg (by decide)]
def root0Candidates21 : List Nat := [7039, 10333, 11197, 18523, 19819, 21319, 23251, 24151, 25153]
def root0Candidates21Block0 : List Nat := [613, 617, 673, 701, 719, 733, 739, 823, 929, 947, 1013, 1019, 1033, 1051, 1181, 1193, 1237, 1481, 1531, 1607, 1627, 1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459]
set_option maxRecDepth 4096 in
theorem root0Candidates21Block0_shape : root0Tail22 = root0Candidates21Block0 ++ root0Tail150 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates21Block0_checked : root0Candidates21Block0.filter (SortedNatTree.contains FastRows.row607) = [7039, 10333] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates21Block0_step : candidates edge 4 3 607 root0Tail22 seed.bound =
    [7039, 10333] ++ candidates edge 4 3 607 root0Tail150 seed.bound := by
  rw [root0Candidates21Block0_shape, candidates_append_of_within edge 4 3 607 root0Candidates21Block0 root0Tail150 seed.bound (by decide)]
  have row : edge 607 = (SortedNatTree.contains FastRows.row607) := by
    change Data.row607.contains = (SortedNatTree.contains FastRows.row607)
    exact FastRows.row607_checked.symm
  rw [row, root0Candidates21Block0_checked]
def root0Candidates21Block1 : List Nat := [10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569]
set_option maxRecDepth 4096 in
theorem root0Candidates21Block1_shape : root0Tail150 = root0Candidates21Block1 ++ root0Tail278 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates21Block1_checked : root0Candidates21Block1.filter (SortedNatTree.contains FastRows.row607) = [11197, 18523, 19819, 21319] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates21Block1_step : candidates edge 4 3 607 root0Tail150 seed.bound =
    [11197, 18523, 19819, 21319] ++ candidates edge 4 3 607 root0Tail278 seed.bound := by
  rw [root0Candidates21Block1_shape, candidates_append_of_within edge 4 3 607 root0Candidates21Block1 root0Tail278 seed.bound (by decide)]
  have row : edge 607 = (SortedNatTree.contains FastRows.row607) := by
    change Data.row607.contains = (SortedNatTree.contains FastRows.row607)
    exact FastRows.row607_checked.symm
  rw [row, root0Candidates21Block1_checked]
def root0Candidates21Block2 : List Nat := [21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239, 24439, 24473, 24517, 24763, 24781, 24799, 24847, 24889, 24953, 25153, 25189, 25219, 25229, 25301, 25343, 25349]
set_option maxRecDepth 4096 in
theorem root0Candidates21Block2_shape : root0Tail278 = root0Candidates21Block2 ++ root0Tail333 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates21Block2_checked : root0Candidates21Block2.filter (SortedNatTree.contains FastRows.row607) = [23251, 24151, 25153] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates21Block2_step : candidates edge 4 3 607 root0Tail278 seed.bound =
    [23251, 24151, 25153] ++ candidates edge 4 3 607 root0Tail333 seed.bound := by
  rw [root0Candidates21Block2_shape, candidates_append_of_within edge 4 3 607 root0Candidates21Block2 root0Tail333 seed.bound (by decide)]
  have row : edge 607 = (SortedNatTree.contains FastRows.row607) := by
    change Data.row607.contains = (SortedNatTree.contains FastRows.row607)
    exact FastRows.row607_checked.symm
  rw [row, root0Candidates21Block2_checked]
theorem root0Candidates21_end : candidates edge 4 3 607 root0Tail333 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates21_checked : candidates edge 4 3 607 root0Tail22 seed.bound = root0Candidates21 := by
  rw [root0Candidates21Block0_step, root0Candidates21Block1_step, root0Candidates21Block2_step, root0Candidates21_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child21 : search edge 3 [3, 607] root0Candidates21 610 seed = seed := by decide
theorem root0Step21 : scan edge (search edge 3) 4 [3] 3 root0Tail21 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail22 seed := by
  rw [root0Tail21, scan, if_neg (by decide), root0Candidates21_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail22
    (if 3 ≤ root0Candidates21.length then search edge 3 [3, 607] root0Candidates21 610 seed else seed) = _
  rw [if_pos (by decide), root0Child21]
def root0Candidates22 : List Nat := [673, 3769, 9901, 17923, 18679]
def root0Candidates22Block0 : List Nat := [617, 673, 701, 719, 733, 739, 823, 929, 947, 1013, 1019, 1033, 1051, 1181, 1193, 1237, 1481, 1531, 1607, 1627, 1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567]
set_option maxRecDepth 4096 in
theorem root0Candidates22Block0_shape : root0Tail23 = root0Candidates22Block0 ++ root0Tail151 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates22Block0_checked : root0Candidates22Block0.filter (SortedNatTree.contains FastRows.row613) = [673, 3769, 9901] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates22Block0_step : candidates edge 4 3 613 root0Tail23 seed.bound =
    [673, 3769, 9901] ++ candidates edge 4 3 613 root0Tail151 seed.bound := by
  rw [root0Candidates22Block0_shape, candidates_append_of_within edge 4 3 613 root0Candidates22Block0 root0Tail151 seed.bound (by decide)]
  have row : edge 613 = (SortedNatTree.contains FastRows.row613) := by
    change Data.row613.contains = (SortedNatTree.contains FastRows.row613)
    exact FastRows.row613_checked.symm
  rw [row, root0Candidates22Block0_checked]
def root0Candidates22Block1 : List Nat := [10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611]
set_option maxRecDepth 4096 in
theorem root0Candidates22Block1_shape : root0Tail151 = root0Candidates22Block1 ++ root0Tail279 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates22Block1_checked : root0Candidates22Block1.filter (SortedNatTree.contains FastRows.row613) = [17923, 18679] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates22Block1_step : candidates edge 4 3 613 root0Tail151 seed.bound =
    [17923, 18679] ++ candidates edge 4 3 613 root0Tail279 seed.bound := by
  rw [root0Candidates22Block1_shape, candidates_append_of_within edge 4 3 613 root0Candidates22Block1 root0Tail279 seed.bound (by decide)]
  have row : edge 613 = (SortedNatTree.contains FastRows.row613) := by
    change Data.row613.contains = (SortedNatTree.contains FastRows.row613)
    exact FastRows.row613_checked.symm
  rw [row, root0Candidates22Block1_checked]
def root0Candidates22Block2 : List Nat := [21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239, 24439, 24473, 24517, 24763, 24781, 24799, 24847, 24889, 24953, 25153, 25189, 25219, 25229, 25301, 25343, 25349]
set_option maxRecDepth 4096 in
theorem root0Candidates22Block2_shape : root0Tail279 = root0Candidates22Block2 ++ root0Tail333 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates22Block2_checked : root0Candidates22Block2.filter (SortedNatTree.contains FastRows.row613) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates22Block2_step : candidates edge 4 3 613 root0Tail279 seed.bound =
    [] ++ candidates edge 4 3 613 root0Tail333 seed.bound := by
  rw [root0Candidates22Block2_shape, candidates_append_of_within edge 4 3 613 root0Candidates22Block2 root0Tail333 seed.bound (by decide)]
  have row : edge 613 = (SortedNatTree.contains FastRows.row613) := by
    change Data.row613.contains = (SortedNatTree.contains FastRows.row613)
    exact FastRows.row613_checked.symm
  rw [row, root0Candidates22Block2_checked]
theorem root0Candidates22_end : candidates edge 4 3 613 root0Tail333 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates22_checked : candidates edge 4 3 613 root0Tail23 seed.bound = root0Candidates22 := by
  rw [root0Candidates22Block0_step, root0Candidates22Block1_step, root0Candidates22Block2_step, root0Candidates22_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child22 : search edge 3 [3, 613] root0Candidates22 616 seed = seed := by decide
theorem root0Step22 : scan edge (search edge 3) 4 [3] 3 root0Tail22 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail23 seed := by
  rw [root0Tail22, scan, if_neg (by decide), root0Candidates22_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail23
    (if 3 ≤ root0Candidates22.length then search edge 3 [3, 613] root0Candidates22 616 seed else seed) = _
  rw [if_pos (by decide), root0Child22]
def root0Candidates23 : List Nat := [4253, 8747, 10889, 18077, 18233, 19211, 19919, 22109]
def root0Candidates23Block0 : List Nat := [673, 701, 719, 733, 739, 823, 929, 947, 1013, 1019, 1033, 1051, 1181, 1193, 1237, 1481, 1531, 1607, 1627, 1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627]
set_option maxRecDepth 4096 in
theorem root0Candidates23Block0_shape : root0Tail24 = root0Candidates23Block0 ++ root0Tail152 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates23Block0_checked : root0Candidates23Block0.filter (SortedNatTree.contains FastRows.row617) = [4253, 8747] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates23Block0_step : candidates edge 4 3 617 root0Tail24 seed.bound =
    [4253, 8747] ++ candidates edge 4 3 617 root0Tail152 seed.bound := by
  rw [root0Candidates23Block0_shape, candidates_append_of_within edge 4 3 617 root0Candidates23Block0 root0Tail152 seed.bound (by decide)]
  have row : edge 617 = (SortedNatTree.contains FastRows.row617) := by
    change Data.row617.contains = (SortedNatTree.contains FastRows.row617)
    exact FastRows.row617_checked.symm
  rw [row, root0Candidates23Block0_checked]
def root0Candidates23Block1 : List Nat := [10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617]
set_option maxRecDepth 4096 in
theorem root0Candidates23Block1_shape : root0Tail152 = root0Candidates23Block1 ++ root0Tail280 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates23Block1_checked : root0Candidates23Block1.filter (SortedNatTree.contains FastRows.row617) = [10889, 18077, 18233, 19211, 19919] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates23Block1_step : candidates edge 4 3 617 root0Tail152 seed.bound =
    [10889, 18077, 18233, 19211, 19919] ++ candidates edge 4 3 617 root0Tail280 seed.bound := by
  rw [root0Candidates23Block1_shape, candidates_append_of_within edge 4 3 617 root0Candidates23Block1 root0Tail280 seed.bound (by decide)]
  have row : edge 617 = (SortedNatTree.contains FastRows.row617) := by
    change Data.row617.contains = (SortedNatTree.contains FastRows.row617)
    exact FastRows.row617_checked.symm
  rw [row, root0Candidates23Block1_checked]
def root0Candidates23Block2 : List Nat := [21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239, 24439, 24473, 24517, 24763, 24781, 24799, 24847, 24889, 24953, 25153, 25189, 25219, 25229, 25301, 25343, 25349]
set_option maxRecDepth 4096 in
theorem root0Candidates23Block2_shape : root0Tail280 = root0Candidates23Block2 ++ root0Tail333 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates23Block2_checked : root0Candidates23Block2.filter (SortedNatTree.contains FastRows.row617) = [22109] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates23Block2_step : candidates edge 4 3 617 root0Tail280 seed.bound =
    [22109] ++ candidates edge 4 3 617 root0Tail333 seed.bound := by
  rw [root0Candidates23Block2_shape, candidates_append_of_within edge 4 3 617 root0Candidates23Block2 root0Tail333 seed.bound (by decide)]
  have row : edge 617 = (SortedNatTree.contains FastRows.row617) := by
    change Data.row617.contains = (SortedNatTree.contains FastRows.row617)
    exact FastRows.row617_checked.symm
  rw [row, root0Candidates23Block2_checked]
theorem root0Candidates23_end : candidates edge 4 3 617 root0Tail333 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates23_checked : candidates edge 4 3 617 root0Tail24 seed.bound = root0Candidates23 := by
  rw [root0Candidates23Block0_step, root0Candidates23Block1_step, root0Candidates23Block2_step, root0Candidates23_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child23 : search edge 3 [3, 617] root0Candidates23 620 seed = seed := by decide
theorem root0Step23 : scan edge (search edge 3) 4 [3] 3 root0Tail23 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail24 seed := by
  rw [root0Tail23, scan, if_neg (by decide), root0Candidates23_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail24
    (if 3 ≤ root0Candidates23.length then search edge 3 [3, 617] root0Candidates23 620 seed else seed) = _
  rw [if_pos (by decide), root0Child23]
def root0Candidates24 : List Nat := [1657, 11821, 22921]
def root0Candidates24Block0 : List Nat := [701, 719, 733, 739, 823, 929, 947, 1013, 1019, 1033, 1051, 1181, 1193, 1237, 1481, 1531, 1607, 1627, 1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771]
set_option maxRecDepth 4096 in
theorem root0Candidates24Block0_shape : root0Tail25 = root0Candidates24Block0 ++ root0Tail153 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates24Block0_checked : root0Candidates24Block0.filter (SortedNatTree.contains FastRows.row673) = [1657] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates24Block0_step : candidates edge 4 3 673 root0Tail25 seed.bound =
    [1657] ++ candidates edge 4 3 673 root0Tail153 seed.bound := by
  rw [root0Candidates24Block0_shape, candidates_append_of_within edge 4 3 673 root0Candidates24Block0 root0Tail153 seed.bound (by decide)]
  have row : edge 673 = (SortedNatTree.contains FastRows.row673) := by
    change Data.row673.contains = (SortedNatTree.contains FastRows.row673)
    exact FastRows.row673_checked.symm
  rw [row, root0Candidates24Block0_checked]
def root0Candidates24Block1 : List Nat := [10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757]
set_option maxRecDepth 4096 in
theorem root0Candidates24Block1_shape : root0Tail153 = root0Candidates24Block1 ++ root0Tail281 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates24Block1_checked : root0Candidates24Block1.filter (SortedNatTree.contains FastRows.row673) = [11821] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates24Block1_step : candidates edge 4 3 673 root0Tail153 seed.bound =
    [11821] ++ candidates edge 4 3 673 root0Tail281 seed.bound := by
  rw [root0Candidates24Block1_shape, candidates_append_of_within edge 4 3 673 root0Candidates24Block1 root0Tail281 seed.bound (by decide)]
  have row : edge 673 = (SortedNatTree.contains FastRows.row673) := by
    change Data.row673.contains = (SortedNatTree.contains FastRows.row673)
    exact FastRows.row673_checked.symm
  rw [row, root0Candidates24Block1_checked]
def root0Candidates24Block2 : List Nat := [21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239, 24439, 24473, 24517, 24763, 24781, 24799, 24847, 24889, 24953, 25153, 25189, 25219, 25229, 25301, 25343, 25349]
set_option maxRecDepth 4096 in
theorem root0Candidates24Block2_shape : root0Tail281 = root0Candidates24Block2 ++ root0Tail333 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates24Block2_checked : root0Candidates24Block2.filter (SortedNatTree.contains FastRows.row673) = [22921] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates24Block2_step : candidates edge 4 3 673 root0Tail281 seed.bound =
    [22921] ++ candidates edge 4 3 673 root0Tail333 seed.bound := by
  rw [root0Candidates24Block2_shape, candidates_append_of_within edge 4 3 673 root0Candidates24Block2 root0Tail333 seed.bound (by decide)]
  have row : edge 673 = (SortedNatTree.contains FastRows.row673) := by
    change Data.row673.contains = (SortedNatTree.contains FastRows.row673)
    exact FastRows.row673_checked.symm
  rw [row, root0Candidates24Block2_checked]
theorem root0Candidates24_end : candidates edge 4 3 673 root0Tail333 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates24_checked : candidates edge 4 3 673 root0Tail25 seed.bound = root0Candidates24 := by
  rw [root0Candidates24Block0_step, root0Candidates24Block1_step, root0Candidates24Block2_step, root0Candidates24_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child24 : search edge 3 [3, 673] root0Candidates24 676 seed = seed := by decide
theorem root0Step24 : scan edge (search edge 3) 4 [3] 3 root0Tail24 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail25 seed := by
  rw [root0Tail24, scan, if_neg (by decide), root0Candidates24_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail25
    (if 3 ≤ root0Candidates24.length then search edge 3 [3, 673] root0Candidates24 676 seed else seed) = _
  rw [if_pos (by decide), root0Child24]
def root0Candidates25 : List Nat := [1181, 1607, 4583, 8747, 18233, 21017]
def root0Candidates25Block0 : List Nat := [719, 733, 739, 823, 929, 947, 1013, 1019, 1033, 1051, 1181, 1193, 1237, 1481, 1531, 1607, 1627, 1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889]
set_option maxRecDepth 4096 in
theorem root0Candidates25Block0_shape : root0Tail26 = root0Candidates25Block0 ++ root0Tail154 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates25Block0_checked : root0Candidates25Block0.filter (SortedNatTree.contains FastRows.row701) = [1181, 1607, 4583, 8747] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates25Block0_step : candidates edge 4 3 701 root0Tail26 seed.bound =
    [1181, 1607, 4583, 8747] ++ candidates edge 4 3 701 root0Tail154 seed.bound := by
  rw [root0Candidates25Block0_shape, candidates_append_of_within edge 4 3 701 root0Candidates25Block0 root0Tail154 seed.bound (by decide)]
  have row : edge 701 = (SortedNatTree.contains FastRows.row701) := by
    change Data.row701.contains = (SortedNatTree.contains FastRows.row701)
    exact FastRows.row701_checked.symm
  rw [row, root0Candidates25Block0_checked]
def root0Candidates25Block1 : List Nat := [10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821]
set_option maxRecDepth 4096 in
theorem root0Candidates25Block1_shape : root0Tail154 = root0Candidates25Block1 ++ root0Tail282 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates25Block1_checked : root0Candidates25Block1.filter (SortedNatTree.contains FastRows.row701) = [18233, 21017] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates25Block1_step : candidates edge 4 3 701 root0Tail154 seed.bound =
    [18233, 21017] ++ candidates edge 4 3 701 root0Tail282 seed.bound := by
  rw [root0Candidates25Block1_shape, candidates_append_of_within edge 4 3 701 root0Candidates25Block1 root0Tail282 seed.bound (by decide)]
  have row : edge 701 = (SortedNatTree.contains FastRows.row701) := by
    change Data.row701.contains = (SortedNatTree.contains FastRows.row701)
    exact FastRows.row701_checked.symm
  rw [row, root0Candidates25Block1_checked]
def root0Candidates25Block2 : List Nat := [21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239, 24439, 24473, 24517, 24763, 24781, 24799, 24847, 24889, 24953, 25153, 25189, 25219, 25229, 25301]
set_option maxRecDepth 4096 in
theorem root0Candidates25Block2_shape : root0Tail282 = root0Candidates25Block2 ++ root0Tail331 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates25Block2_checked : root0Candidates25Block2.filter (SortedNatTree.contains FastRows.row701) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates25Block2_step : candidates edge 4 3 701 root0Tail282 seed.bound =
    [] ++ candidates edge 4 3 701 root0Tail331 seed.bound := by
  rw [root0Candidates25Block2_shape, candidates_append_of_within edge 4 3 701 root0Candidates25Block2 root0Tail331 seed.bound (by decide)]
  have row : edge 701 = (SortedNatTree.contains FastRows.row701) := by
    change Data.row701.contains = (SortedNatTree.contains FastRows.row701)
    exact FastRows.row701_checked.symm
  rw [row, root0Candidates25Block2_checked]
theorem root0Candidates25_end : candidates edge 4 3 701 root0Tail331 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates25_checked : candidates edge 4 3 701 root0Tail26 seed.bound = root0Candidates25 := by
  rw [root0Candidates25Block0_step, root0Candidates25Block1_step, root0Candidates25Block2_step, root0Candidates25_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child25 : search edge 3 [3, 701] root0Candidates25 704 seed = seed := by decide
theorem root0Step25 : scan edge (search edge 3) 4 [3] 3 root0Tail25 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail26 seed := by
  rw [root0Tail25, scan, if_neg (by decide), root0Candidates25_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail26
    (if 3 ≤ root0Candidates25.length then search edge 3 [3, 701] root0Candidates25 704 seed else seed) = _
  rw [if_pos (by decide), root0Child25]
def root0Candidates26 : List Nat := [947, 2579, 12413, 20759, 20939]
def root0Candidates26Block0 : List Nat := [733, 739, 823, 929, 947, 1013, 1019, 1033, 1051, 1181, 1193, 1237, 1481, 1531, 1607, 1627, 1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987]
set_option maxRecDepth 4096 in
theorem root0Candidates26Block0_shape : root0Tail27 = root0Candidates26Block0 ++ root0Tail155 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates26Block0_checked : root0Candidates26Block0.filter (SortedNatTree.contains FastRows.row719) = [947, 2579] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates26Block0_step : candidates edge 4 3 719 root0Tail27 seed.bound =
    [947, 2579] ++ candidates edge 4 3 719 root0Tail155 seed.bound := by
  rw [root0Candidates26Block0_shape, candidates_append_of_within edge 4 3 719 root0Candidates26Block0 root0Tail155 seed.bound (by decide)]
  have row : edge 719 = (SortedNatTree.contains FastRows.row719) := by
    change Data.row719.contains = (SortedNatTree.contains FastRows.row719)
    exact FastRows.row719_checked.symm
  rw [row, root0Candidates26Block0_checked]
def root0Candidates26Block1 : List Nat := [11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851]
set_option maxRecDepth 4096 in
theorem root0Candidates26Block1_shape : root0Tail155 = root0Candidates26Block1 ++ root0Tail283 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates26Block1_checked : root0Candidates26Block1.filter (SortedNatTree.contains FastRows.row719) = [12413, 20759, 20939] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates26Block1_step : candidates edge 4 3 719 root0Tail155 seed.bound =
    [12413, 20759, 20939] ++ candidates edge 4 3 719 root0Tail283 seed.bound := by
  rw [root0Candidates26Block1_shape, candidates_append_of_within edge 4 3 719 root0Candidates26Block1 root0Tail283 seed.bound (by decide)]
  have row : edge 719 = (SortedNatTree.contains FastRows.row719) := by
    change Data.row719.contains = (SortedNatTree.contains FastRows.row719)
    exact FastRows.row719_checked.symm
  rw [row, root0Candidates26Block1_checked]
def root0Candidates26Block2 : List Nat := [21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239, 24439, 24473, 24517, 24763, 24781, 24799, 24847, 24889, 24953, 25153, 25189, 25219, 25229, 25301]
set_option maxRecDepth 4096 in
theorem root0Candidates26Block2_shape : root0Tail283 = root0Candidates26Block2 ++ root0Tail331 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates26Block2_checked : root0Candidates26Block2.filter (SortedNatTree.contains FastRows.row719) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates26Block2_step : candidates edge 4 3 719 root0Tail283 seed.bound =
    [] ++ candidates edge 4 3 719 root0Tail331 seed.bound := by
  rw [root0Candidates26Block2_shape, candidates_append_of_within edge 4 3 719 root0Candidates26Block2 root0Tail331 seed.bound (by decide)]
  have row : edge 719 = (SortedNatTree.contains FastRows.row719) := by
    change Data.row719.contains = (SortedNatTree.contains FastRows.row719)
    exact FastRows.row719_checked.symm
  rw [row, root0Candidates26Block2_checked]
theorem root0Candidates26_end : candidates edge 4 3 719 root0Tail331 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates26_checked : candidates edge 4 3 719 root0Tail27 seed.bound = root0Candidates26 := by
  rw [root0Candidates26Block0_step, root0Candidates26Block1_step, root0Candidates26Block2_step, root0Candidates26_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child26 : search edge 3 [3, 719] root0Candidates26 722 seed = seed := by decide
theorem root0Step26 : scan edge (search edge 3) 4 [3] 3 root0Tail26 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail27 seed := by
  rw [root0Tail26, scan, if_neg (by decide), root0Candidates26_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail27
    (if 3 ≤ root0Candidates26.length then search edge 3 [3, 719] root0Candidates26 722 seed else seed) = _
  rw [if_pos (by decide), root0Child26]
def root0Candidates27 : List Nat := [8431, 14401, 19819, 20899, 21757]
def root0Candidates27Block0 : List Nat := [739, 823, 929, 947, 1013, 1019, 1033, 1051, 1181, 1193, 1237, 1481, 1531, 1607, 1627, 1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027]
set_option maxRecDepth 4096 in
theorem root0Candidates27Block0_shape : root0Tail28 = root0Candidates27Block0 ++ root0Tail156 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates27Block0_checked : root0Candidates27Block0.filter (SortedNatTree.contains FastRows.row733) = [8431] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates27Block0_step : candidates edge 4 3 733 root0Tail28 seed.bound =
    [8431] ++ candidates edge 4 3 733 root0Tail156 seed.bound := by
  rw [root0Candidates27Block0_shape, candidates_append_of_within edge 4 3 733 root0Candidates27Block0 root0Tail156 seed.bound (by decide)]
  have row : edge 733 = (SortedNatTree.contains FastRows.row733) := by
    change Data.row733.contains = (SortedNatTree.contains FastRows.row733)
    exact FastRows.row733_checked.symm
  rw [row, root0Candidates27Block0_checked]
def root0Candidates27Block1 : List Nat := [11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961]
set_option maxRecDepth 4096 in
theorem root0Candidates27Block1_shape : root0Tail156 = root0Candidates27Block1 ++ root0Tail284 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates27Block1_checked : root0Candidates27Block1.filter (SortedNatTree.contains FastRows.row733) = [14401, 19819, 20899, 21757] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates27Block1_step : candidates edge 4 3 733 root0Tail156 seed.bound =
    [14401, 19819, 20899, 21757] ++ candidates edge 4 3 733 root0Tail284 seed.bound := by
  rw [root0Candidates27Block1_shape, candidates_append_of_within edge 4 3 733 root0Candidates27Block1 root0Tail284 seed.bound (by decide)]
  have row : edge 733 = (SortedNatTree.contains FastRows.row733) := by
    change Data.row733.contains = (SortedNatTree.contains FastRows.row733)
    exact FastRows.row733_checked.symm
  rw [row, root0Candidates27Block1_checked]
def root0Candidates27Block2 : List Nat := [22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239, 24439, 24473, 24517, 24763, 24781, 24799, 24847, 24889, 24953, 25153, 25189, 25219, 25229]
set_option maxRecDepth 4096 in
theorem root0Candidates27Block2_shape : root0Tail284 = root0Candidates27Block2 ++ root0Tail330 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates27Block2_checked : root0Candidates27Block2.filter (SortedNatTree.contains FastRows.row733) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates27Block2_step : candidates edge 4 3 733 root0Tail284 seed.bound =
    [] ++ candidates edge 4 3 733 root0Tail330 seed.bound := by
  rw [root0Candidates27Block2_shape, candidates_append_of_within edge 4 3 733 root0Candidates27Block2 root0Tail330 seed.bound (by decide)]
  have row : edge 733 = (SortedNatTree.contains FastRows.row733) := by
    change Data.row733.contains = (SortedNatTree.contains FastRows.row733)
    exact FastRows.row733_checked.symm
  rw [row, root0Candidates27Block2_checked]
theorem root0Candidates27_end : candidates edge 4 3 733 root0Tail330 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates27_checked : candidates edge 4 3 733 root0Tail28 seed.bound = root0Candidates27 := by
  rw [root0Candidates27Block0_step, root0Candidates27Block1_step, root0Candidates27Block2_step, root0Candidates27_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child27 : search edge 3 [3, 733] root0Candidates27 736 seed = seed := by decide
theorem root0Step27 : scan edge (search edge 3) 4 [3] 3 root0Tail27 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail28 seed := by
  rw [root0Tail27, scan, if_neg (by decide), root0Candidates27_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail28
    (if 3 ≤ root0Candidates27.length then search edge 3 [3, 733] root0Candidates27 736 seed else seed) = _
  rw [if_pos (by decide), root0Child27]
def root0Candidates28 : List Nat := [6793, 8431, 17209, 18181]
def root0Candidates28Block0 : List Nat := [823, 929, 947, 1013, 1019, 1033, 1051, 1181, 1193, 1237, 1481, 1531, 1607, 1627, 1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173]
set_option maxRecDepth 4096 in
theorem root0Candidates28Block0_shape : root0Tail29 = root0Candidates28Block0 ++ root0Tail157 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates28Block0_checked : root0Candidates28Block0.filter (SortedNatTree.contains FastRows.row739) = [6793, 8431] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates28Block0_step : candidates edge 4 3 739 root0Tail29 seed.bound =
    [6793, 8431] ++ candidates edge 4 3 739 root0Tail157 seed.bound := by
  rw [root0Candidates28Block0_shape, candidates_append_of_within edge 4 3 739 root0Candidates28Block0 root0Tail157 seed.bound (by decide)]
  have row : edge 739 = (SortedNatTree.contains FastRows.row739) := by
    change Data.row739.contains = (SortedNatTree.contains FastRows.row739)
    exact FastRows.row739_checked.symm
  rw [row, root0Candidates28Block0_checked]
def root0Candidates28Block1 : List Nat := [11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037]
set_option maxRecDepth 4096 in
theorem root0Candidates28Block1_shape : root0Tail157 = root0Candidates28Block1 ++ root0Tail285 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates28Block1_checked : root0Candidates28Block1.filter (SortedNatTree.contains FastRows.row739) = [17209, 18181] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates28Block1_step : candidates edge 4 3 739 root0Tail157 seed.bound =
    [17209, 18181] ++ candidates edge 4 3 739 root0Tail285 seed.bound := by
  rw [root0Candidates28Block1_shape, candidates_append_of_within edge 4 3 739 root0Candidates28Block1 root0Tail285 seed.bound (by decide)]
  have row : edge 739 = (SortedNatTree.contains FastRows.row739) := by
    change Data.row739.contains = (SortedNatTree.contains FastRows.row739)
    exact FastRows.row739_checked.symm
  rw [row, root0Candidates28Block1_checked]
def root0Candidates28Block2 : List Nat := [22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239, 24439, 24473, 24517, 24763, 24781, 24799, 24847, 24889, 24953, 25153, 25189, 25219, 25229]
set_option maxRecDepth 4096 in
theorem root0Candidates28Block2_shape : root0Tail285 = root0Candidates28Block2 ++ root0Tail330 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates28Block2_checked : root0Candidates28Block2.filter (SortedNatTree.contains FastRows.row739) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates28Block2_step : candidates edge 4 3 739 root0Tail285 seed.bound =
    [] ++ candidates edge 4 3 739 root0Tail330 seed.bound := by
  rw [root0Candidates28Block2_shape, candidates_append_of_within edge 4 3 739 root0Candidates28Block2 root0Tail330 seed.bound (by decide)]
  have row : edge 739 = (SortedNatTree.contains FastRows.row739) := by
    change Data.row739.contains = (SortedNatTree.contains FastRows.row739)
    exact FastRows.row739_checked.symm
  rw [row, root0Candidates28Block2_checked]
theorem root0Candidates28_end : candidates edge 4 3 739 root0Tail330 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates28_checked : candidates edge 4 3 739 root0Tail29 seed.bound = root0Candidates28 := by
  rw [root0Candidates28Block0_step, root0Candidates28Block1_step, root0Candidates28Block2_step, root0Candidates28_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child28 : search edge 3 [3, 739] root0Candidates28 742 seed = seed := by decide
theorem root0Step28 : scan edge (search edge 3) 4 [3] 3 root0Tail28 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail29 seed := by
  rw [root0Tail28, scan, if_neg (by decide), root0Candidates28_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail29
    (if 3 ≤ root0Candidates28.length then search edge 3 [3, 739] root0Candidates28 742 seed else seed) = _
  rw [if_pos (by decide), root0Child28]
def root0Candidates29 : List Nat := [5281, 5323, 15559, 20107]
def root0Candidates29Block0 : List Nat := [929, 947, 1013, 1019, 1033, 1051, 1181, 1193, 1237, 1481, 1531, 1607, 1627, 1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177]
set_option maxRecDepth 4096 in
theorem root0Candidates29Block0_shape : root0Tail30 = root0Candidates29Block0 ++ root0Tail158 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates29Block0_checked : root0Candidates29Block0.filter (SortedNatTree.contains FastRows.row823) = [5281, 5323] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates29Block0_step : candidates edge 4 3 823 root0Tail30 seed.bound =
    [5281, 5323] ++ candidates edge 4 3 823 root0Tail158 seed.bound := by
  rw [root0Candidates29Block0_shape, candidates_append_of_within edge 4 3 823 root0Candidates29Block0 root0Tail158 seed.bound (by decide)]
  have row : edge 823 = (SortedNatTree.contains FastRows.row823) := by
    change Data.row823.contains = (SortedNatTree.contains FastRows.row823)
    exact FastRows.row823_checked.symm
  rw [row, root0Candidates29Block0_checked]
def root0Candidates29Block1 : List Nat := [11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051]
set_option maxRecDepth 4096 in
theorem root0Candidates29Block1_shape : root0Tail158 = root0Candidates29Block1 ++ root0Tail286 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates29Block1_checked : root0Candidates29Block1.filter (SortedNatTree.contains FastRows.row823) = [15559, 20107] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates29Block1_step : candidates edge 4 3 823 root0Tail158 seed.bound =
    [15559, 20107] ++ candidates edge 4 3 823 root0Tail286 seed.bound := by
  rw [root0Candidates29Block1_shape, candidates_append_of_within edge 4 3 823 root0Candidates29Block1 root0Tail286 seed.bound (by decide)]
  have row : edge 823 = (SortedNatTree.contains FastRows.row823) := by
    change Data.row823.contains = (SortedNatTree.contains FastRows.row823)
    exact FastRows.row823_checked.symm
  rw [row, root0Candidates29Block1_checked]
def root0Candidates29Block2 : List Nat := [22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239, 24439, 24473, 24517, 24763, 24781, 24799, 24847, 24889, 24953, 25153, 25189]
set_option maxRecDepth 4096 in
theorem root0Candidates29Block2_shape : root0Tail286 = root0Candidates29Block2 ++ root0Tail328 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates29Block2_checked : root0Candidates29Block2.filter (SortedNatTree.contains FastRows.row823) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates29Block2_step : candidates edge 4 3 823 root0Tail286 seed.bound =
    [] ++ candidates edge 4 3 823 root0Tail328 seed.bound := by
  rw [root0Candidates29Block2_shape, candidates_append_of_within edge 4 3 823 root0Candidates29Block2 root0Tail328 seed.bound (by decide)]
  have row : edge 823 = (SortedNatTree.contains FastRows.row823) := by
    change Data.row823.contains = (SortedNatTree.contains FastRows.row823)
    exact FastRows.row823_checked.symm
  rw [row, root0Candidates29Block2_checked]
theorem root0Candidates29_end : candidates edge 4 3 823 root0Tail328 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates29_checked : candidates edge 4 3 823 root0Tail30 seed.bound = root0Candidates29 := by
  rw [root0Candidates29Block0_step, root0Candidates29Block1_step, root0Candidates29Block2_step, root0Candidates29_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child29 : search edge 3 [3, 823] root0Candidates29 826 seed = seed := by decide
theorem root0Step29 : scan edge (search edge 3) 4 [3] 3 root0Tail29 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail30 seed := by
  rw [root0Tail29, scan, if_neg (by decide), root0Candidates29_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail30
    (if 3 ≤ root0Candidates29.length then search edge 3 [3, 823] root0Candidates29 826 seed else seed) = _
  rw [if_pos (by decide), root0Child29]
def root0Candidates30 : List Nat := [6563, 8747, 11393]
def root0Candidates30Block0 : List Nat := [947, 1013, 1019, 1033, 1051, 1181, 1193, 1237, 1481, 1531, 1607, 1627, 1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197]
set_option maxRecDepth 4096 in
theorem root0Candidates30Block0_shape : root0Tail31 = root0Candidates30Block0 ++ root0Tail159 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates30Block0_checked : root0Candidates30Block0.filter (SortedNatTree.contains FastRows.row929) = [6563, 8747] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates30Block0_step : candidates edge 4 3 929 root0Tail31 seed.bound =
    [6563, 8747] ++ candidates edge 4 3 929 root0Tail159 seed.bound := by
  rw [root0Candidates30Block0_shape, candidates_append_of_within edge 4 3 929 root0Candidates30Block0 root0Tail159 seed.bound (by decide)]
  have row : edge 929 = (SortedNatTree.contains FastRows.row929) := by
    change Data.row929.contains = (SortedNatTree.contains FastRows.row929)
    exact FastRows.row929_checked.symm
  rw [row, root0Candidates30Block0_checked]
def root0Candidates30Block1 : List Nat := [11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067]
set_option maxRecDepth 4096 in
theorem root0Candidates30Block1_shape : root0Tail159 = root0Candidates30Block1 ++ root0Tail287 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates30Block1_checked : root0Candidates30Block1.filter (SortedNatTree.contains FastRows.row929) = [11393] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates30Block1_step : candidates edge 4 3 929 root0Tail159 seed.bound =
    [11393] ++ candidates edge 4 3 929 root0Tail287 seed.bound := by
  rw [root0Candidates30Block1_shape, candidates_append_of_within edge 4 3 929 root0Candidates30Block1 root0Tail287 seed.bound (by decide)]
  have row : edge 929 = (SortedNatTree.contains FastRows.row929) := by
    change Data.row929.contains = (SortedNatTree.contains FastRows.row929)
    exact FastRows.row929_checked.symm
  rw [row, root0Candidates30Block1_checked]
def root0Candidates30Block2 : List Nat := [22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239, 24439, 24473, 24517, 24763, 24781, 24799, 24847, 24889, 24953]
set_option maxRecDepth 4096 in
theorem root0Candidates30Block2_shape : root0Tail287 = root0Candidates30Block2 ++ root0Tail326 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates30Block2_checked : root0Candidates30Block2.filter (SortedNatTree.contains FastRows.row929) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates30Block2_step : candidates edge 4 3 929 root0Tail287 seed.bound =
    [] ++ candidates edge 4 3 929 root0Tail326 seed.bound := by
  rw [root0Candidates30Block2_shape, candidates_append_of_within edge 4 3 929 root0Candidates30Block2 root0Tail326 seed.bound (by decide)]
  have row : edge 929 = (SortedNatTree.contains FastRows.row929) := by
    change Data.row929.contains = (SortedNatTree.contains FastRows.row929)
    exact FastRows.row929_checked.symm
  rw [row, root0Candidates30Block2_checked]
theorem root0Candidates30_end : candidates edge 4 3 929 root0Tail326 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates30_checked : candidates edge 4 3 929 root0Tail31 seed.bound = root0Candidates30 := by
  rw [root0Candidates30Block0_step, root0Candidates30Block1_step, root0Candidates30Block2_step, root0Candidates30_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child30 : search edge 3 [3, 929] root0Candidates30 932 seed = seed := by decide
theorem root0Step30 : scan edge (search edge 3) 4 [3] 3 root0Tail30 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail31 seed := by
  rw [root0Tail30, scan, if_neg (by decide), root0Candidates30_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail31
    (if 3 ≤ root0Candidates30.length then search edge 3 [3, 929] root0Candidates30 932 seed else seed) = _
  rw [if_pos (by decide), root0Child30]
def root0Candidates31 : List Nat := [5051, 6947, 13763, 18233]
def root0Candidates31Block0 : List Nat := [1013, 1019, 1033, 1051, 1181, 1193, 1237, 1481, 1531, 1607, 1627, 1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393]
set_option maxRecDepth 4096 in
theorem root0Candidates31Block0_shape : root0Tail32 = root0Candidates31Block0 ++ root0Tail160 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates31Block0_checked : root0Candidates31Block0.filter (SortedNatTree.contains FastRows.row947) = [5051, 6947] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates31Block0_step : candidates edge 4 3 947 root0Tail32 seed.bound =
    [5051, 6947] ++ candidates edge 4 3 947 root0Tail160 seed.bound := by
  rw [root0Candidates31Block0_shape, candidates_append_of_within edge 4 3 947 root0Candidates31Block0 root0Tail160 seed.bound (by decide)]
  have row : edge 947 = (SortedNatTree.contains FastRows.row947) := by
    change Data.row947.contains = (SortedNatTree.contains FastRows.row947)
    exact FastRows.row947_checked.symm
  rw [row, root0Candidates31Block0_checked]
def root0Candidates31Block1 : List Nat := [11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079]
set_option maxRecDepth 4096 in
theorem root0Candidates31Block1_shape : root0Tail160 = root0Candidates31Block1 ++ root0Tail288 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates31Block1_checked : root0Candidates31Block1.filter (SortedNatTree.contains FastRows.row947) = [13763, 18233] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates31Block1_step : candidates edge 4 3 947 root0Tail160 seed.bound =
    [13763, 18233] ++ candidates edge 4 3 947 root0Tail288 seed.bound := by
  rw [root0Candidates31Block1_shape, candidates_append_of_within edge 4 3 947 root0Candidates31Block1 root0Tail288 seed.bound (by decide)]
  have row : edge 947 = (SortedNatTree.contains FastRows.row947) := by
    change Data.row947.contains = (SortedNatTree.contains FastRows.row947)
    exact FastRows.row947_checked.symm
  rw [row, root0Candidates31Block1_checked]
def root0Candidates31Block2 : List Nat := [22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239, 24439, 24473, 24517, 24763, 24781, 24799, 24847, 24889, 24953]
set_option maxRecDepth 4096 in
theorem root0Candidates31Block2_shape : root0Tail288 = root0Candidates31Block2 ++ root0Tail326 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates31Block2_checked : root0Candidates31Block2.filter (SortedNatTree.contains FastRows.row947) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates31Block2_step : candidates edge 4 3 947 root0Tail288 seed.bound =
    [] ++ candidates edge 4 3 947 root0Tail326 seed.bound := by
  rw [root0Candidates31Block2_shape, candidates_append_of_within edge 4 3 947 root0Candidates31Block2 root0Tail326 seed.bound (by decide)]
  have row : edge 947 = (SortedNatTree.contains FastRows.row947) := by
    change Data.row947.contains = (SortedNatTree.contains FastRows.row947)
    exact FastRows.row947_checked.symm
  rw [row, root0Candidates31Block2_checked]
theorem root0Candidates31_end : candidates edge 4 3 947 root0Tail326 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates31_checked : candidates edge 4 3 947 root0Tail32 seed.bound = root0Candidates31 := by
  rw [root0Candidates31Block0_step, root0Candidates31Block1_step, root0Candidates31Block2_step, root0Candidates31_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child31 : search edge 3 [3, 947] root0Candidates31 950 seed = seed := by decide
theorem root0Step31 : scan edge (search edge 3) 4 [3] 3 root0Tail31 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail32 seed := by
  rw [root0Tail31, scan, if_neg (by decide), root0Candidates31_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail32
    (if 3 ≤ root0Candidates31.length then search edge 3 [3, 947] root0Candidates31 950 seed else seed) = _
  rw [if_pos (by decide), root0Child31]
end ProjectEulerSolutions.PrimePairLarge.Search
