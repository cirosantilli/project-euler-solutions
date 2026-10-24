import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot000Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root0Candidates32 : List Nat := [16703, 21569, 22709, 23549]
def root0Candidates32Block0 : List Nat := [1019, 1033, 1051, 1181, 1193, 1237, 1481, 1531, 1607, 1627, 1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447]
set_option maxRecDepth 4096 in
theorem root0Candidates32Block0_shape : root0Tail33 = root0Candidates32Block0 ++ root0Tail161 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates32Block0_checked : root0Candidates32Block0.filter (SortedNatTree.contains FastRows.row1013) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates32Block0_step : candidates edge 4 3 1013 root0Tail33 seed.bound =
    [] ++ candidates edge 4 3 1013 root0Tail161 seed.bound := by
  rw [root0Candidates32Block0_shape, candidates_append_of_within edge 4 3 1013 root0Candidates32Block0 root0Tail161 seed.bound (by decide)]
  have row : edge 1013 = (SortedNatTree.contains FastRows.row1013) := by
    change Data.row1013.contains = (SortedNatTree.contains FastRows.row1013)
    exact FastRows.row1013_checked.symm
  rw [row, root0Candidates32Block0_checked]
def root0Candidates32Block1 : List Nat := [11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093]
set_option maxRecDepth 4096 in
theorem root0Candidates32Block1_shape : root0Tail161 = root0Candidates32Block1 ++ root0Tail289 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates32Block1_checked : root0Candidates32Block1.filter (SortedNatTree.contains FastRows.row1013) = [16703, 21569] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates32Block1_step : candidates edge 4 3 1013 root0Tail161 seed.bound =
    [16703, 21569] ++ candidates edge 4 3 1013 root0Tail289 seed.bound := by
  rw [root0Candidates32Block1_shape, candidates_append_of_within edge 4 3 1013 root0Candidates32Block1 root0Tail289 seed.bound (by decide)]
  have row : edge 1013 = (SortedNatTree.contains FastRows.row1013) := by
    change Data.row1013.contains = (SortedNatTree.contains FastRows.row1013)
    exact FastRows.row1013_checked.symm
  rw [row, root0Candidates32Block1_checked]
def root0Candidates32Block2 : List Nat := [22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239, 24439, 24473, 24517, 24763, 24781, 24799, 24847, 24889, 24953]
set_option maxRecDepth 4096 in
theorem root0Candidates32Block2_shape : root0Tail289 = root0Candidates32Block2 ++ root0Tail326 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates32Block2_checked : root0Candidates32Block2.filter (SortedNatTree.contains FastRows.row1013) = [22709, 23549] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates32Block2_step : candidates edge 4 3 1013 root0Tail289 seed.bound =
    [22709, 23549] ++ candidates edge 4 3 1013 root0Tail326 seed.bound := by
  rw [root0Candidates32Block2_shape, candidates_append_of_within edge 4 3 1013 root0Candidates32Block2 root0Tail326 seed.bound (by decide)]
  have row : edge 1013 = (SortedNatTree.contains FastRows.row1013) := by
    change Data.row1013.contains = (SortedNatTree.contains FastRows.row1013)
    exact FastRows.row1013_checked.symm
  rw [row, root0Candidates32Block2_checked]
theorem root0Candidates32_end : candidates edge 4 3 1013 root0Tail326 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates32_checked : candidates edge 4 3 1013 root0Tail33 seed.bound = root0Candidates32 := by
  rw [root0Candidates32Block0_step, root0Candidates32Block1_step, root0Candidates32Block2_step, root0Candidates32_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child32 : search edge 3 [3, 1013] root0Candidates32 1016 seed = seed := by decide
theorem root0Step32 : scan edge (search edge 3) 4 [3] 3 root0Tail32 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail33 seed := by
  rw [root0Tail32, scan, if_neg (by decide), root0Candidates32_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail33
    (if 3 ≤ root0Candidates32.length then search edge 3 [3, 1013] root0Candidates32 1016 seed else seed) = _
  rw [if_pos (by decide), root0Child32]
def root0Candidates33 : List Nat := [2957, 6947, 17327, 17789, 21383, 22349]
def root0Candidates33Block0 : List Nat := [1033, 1051, 1181, 1193, 1237, 1481, 1531, 1607, 1627, 1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551]
set_option maxRecDepth 4096 in
theorem root0Candidates33Block0_shape : root0Tail34 = root0Candidates33Block0 ++ root0Tail162 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates33Block0_checked : root0Candidates33Block0.filter (SortedNatTree.contains FastRows.row1019) = [2957, 6947] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates33Block0_step : candidates edge 4 3 1019 root0Tail34 seed.bound =
    [2957, 6947] ++ candidates edge 4 3 1019 root0Tail162 seed.bound := by
  rw [root0Candidates33Block0_shape, candidates_append_of_within edge 4 3 1019 root0Candidates33Block0 root0Tail162 seed.bound (by decide)]
  have row : edge 1019 = (SortedNatTree.contains FastRows.row1019) := by
    change Data.row1019.contains = (SortedNatTree.contains FastRows.row1019)
    exact FastRows.row1019_checked.symm
  rw [row, root0Candidates33Block0_checked]
def root0Candidates33Block1 : List Nat := [11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109]
set_option maxRecDepth 4096 in
theorem root0Candidates33Block1_shape : root0Tail162 = root0Candidates33Block1 ++ root0Tail290 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates33Block1_checked : root0Candidates33Block1.filter (SortedNatTree.contains FastRows.row1019) = [17327, 17789, 21383] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates33Block1_step : candidates edge 4 3 1019 root0Tail162 seed.bound =
    [17327, 17789, 21383] ++ candidates edge 4 3 1019 root0Tail290 seed.bound := by
  rw [root0Candidates33Block1_shape, candidates_append_of_within edge 4 3 1019 root0Candidates33Block1 root0Tail290 seed.bound (by decide)]
  have row : edge 1019 = (SortedNatTree.contains FastRows.row1019) := by
    change Data.row1019.contains = (SortedNatTree.contains FastRows.row1019)
    exact FastRows.row1019_checked.symm
  rw [row, root0Candidates33Block1_checked]
def root0Candidates33Block2 : List Nat := [22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239, 24439, 24473, 24517, 24763, 24781, 24799, 24847, 24889, 24953]
set_option maxRecDepth 4096 in
theorem root0Candidates33Block2_shape : root0Tail290 = root0Candidates33Block2 ++ root0Tail326 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates33Block2_checked : root0Candidates33Block2.filter (SortedNatTree.contains FastRows.row1019) = [22349] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates33Block2_step : candidates edge 4 3 1019 root0Tail290 seed.bound =
    [22349] ++ candidates edge 4 3 1019 root0Tail326 seed.bound := by
  rw [root0Candidates33Block2_shape, candidates_append_of_within edge 4 3 1019 root0Candidates33Block2 root0Tail326 seed.bound (by decide)]
  have row : edge 1019 = (SortedNatTree.contains FastRows.row1019) := by
    change Data.row1019.contains = (SortedNatTree.contains FastRows.row1019)
    exact FastRows.row1019_checked.symm
  rw [row, root0Candidates33Block2_checked]
theorem root0Candidates33_end : candidates edge 4 3 1019 root0Tail326 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates33_checked : candidates edge 4 3 1019 root0Tail34 seed.bound = root0Candidates33 := by
  rw [root0Candidates33Block0_step, root0Candidates33Block1_step, root0Candidates33Block2_step, root0Candidates33_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child33 : search edge 3 [3, 1019] root0Candidates33 1022 seed = seed := by decide
theorem root0Step33 : scan edge (search edge 3) 4 [3] 3 root0Tail33 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail34 seed := by
  rw [root0Tail33, scan, if_neg (by decide), root0Candidates33_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail34
    (if 3 ≤ root0Candidates33.length then search edge 3 [3, 1019] root0Candidates33 1022 seed else seed) = _
  rw [if_pos (by decide), root0Child33]
def root0Candidates34 : List Nat := [1663, 5869, 6451, 19441, 20521]
def root0Candidates34Block0 : List Nat := [1051, 1181, 1193, 1237, 1481, 1531, 1607, 1627, 1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821]
set_option maxRecDepth 4096 in
theorem root0Candidates34Block0_shape : root0Tail35 = root0Candidates34Block0 ++ root0Tail163 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates34Block0_checked : root0Candidates34Block0.filter (SortedNatTree.contains FastRows.row1033) = [1663, 5869, 6451] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates34Block0_step : candidates edge 4 3 1033 root0Tail35 seed.bound =
    [1663, 5869, 6451] ++ candidates edge 4 3 1033 root0Tail163 seed.bound := by
  rw [root0Candidates34Block0_shape, candidates_append_of_within edge 4 3 1033 root0Candidates34Block0 root0Tail163 seed.bound (by decide)]
  have row : edge 1033 = (SortedNatTree.contains FastRows.row1033) := by
    change Data.row1033.contains = (SortedNatTree.contains FastRows.row1033)
    exact FastRows.row1033_checked.symm
  rw [row, root0Candidates34Block0_checked]
def root0Candidates34Block1 : List Nat := [11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171]
set_option maxRecDepth 4096 in
theorem root0Candidates34Block1_shape : root0Tail163 = root0Candidates34Block1 ++ root0Tail291 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates34Block1_checked : root0Candidates34Block1.filter (SortedNatTree.contains FastRows.row1033) = [19441, 20521] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates34Block1_step : candidates edge 4 3 1033 root0Tail163 seed.bound =
    [19441, 20521] ++ candidates edge 4 3 1033 root0Tail291 seed.bound := by
  rw [root0Candidates34Block1_shape, candidates_append_of_within edge 4 3 1033 root0Candidates34Block1 root0Tail291 seed.bound (by decide)]
  have row : edge 1033 = (SortedNatTree.contains FastRows.row1033) := by
    change Data.row1033.contains = (SortedNatTree.contains FastRows.row1033)
    exact FastRows.row1033_checked.symm
  rw [row, root0Candidates34Block1_checked]
def root0Candidates34Block2 : List Nat := [22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239, 24439, 24473, 24517, 24763, 24781, 24799, 24847, 24889, 24953]
set_option maxRecDepth 4096 in
theorem root0Candidates34Block2_shape : root0Tail291 = root0Candidates34Block2 ++ root0Tail326 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates34Block2_checked : root0Candidates34Block2.filter (SortedNatTree.contains FastRows.row1033) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates34Block2_step : candidates edge 4 3 1033 root0Tail291 seed.bound =
    [] ++ candidates edge 4 3 1033 root0Tail326 seed.bound := by
  rw [root0Candidates34Block2_shape, candidates_append_of_within edge 4 3 1033 root0Candidates34Block2 root0Tail326 seed.bound (by decide)]
  have row : edge 1033 = (SortedNatTree.contains FastRows.row1033) := by
    change Data.row1033.contains = (SortedNatTree.contains FastRows.row1033)
    exact FastRows.row1033_checked.symm
  rw [row, root0Candidates34Block2_checked]
theorem root0Candidates34_end : candidates edge 4 3 1033 root0Tail326 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates34_checked : candidates edge 4 3 1033 root0Tail35 seed.bound = root0Candidates34 := by
  rw [root0Candidates34Block0_step, root0Candidates34Block1_step, root0Candidates34Block2_step, root0Candidates34_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child34 : search edge 3 [3, 1033] root0Candidates34 1036 seed = seed := by decide
theorem root0Step34 : scan edge (search edge 3) 4 [3] 3 root0Tail34 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail35 seed := by
  rw [root0Tail34, scan, if_neg (by decide), root0Candidates34_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail35
    (if 3 ≤ root0Candidates34.length then search edge 3 [3, 1033] root0Candidates34 1036 seed else seed) = _
  rw [if_pos (by decide), root0Child34]
def root0Candidates35 : List Nat := [4729, 5839, 19237, 19699, 24763, 24781]
def root0Candidates35Block0 : List Nat := [1181, 1193, 1237, 1481, 1531, 1607, 1627, 1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827]
set_option maxRecDepth 4096 in
theorem root0Candidates35Block0_shape : root0Tail36 = root0Candidates35Block0 ++ root0Tail164 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates35Block0_checked : root0Candidates35Block0.filter (SortedNatTree.contains FastRows.row1051) = [4729, 5839] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates35Block0_step : candidates edge 4 3 1051 root0Tail36 seed.bound =
    [4729, 5839] ++ candidates edge 4 3 1051 root0Tail164 seed.bound := by
  rw [root0Candidates35Block0_shape, candidates_append_of_within edge 4 3 1051 root0Candidates35Block0 root0Tail164 seed.bound (by decide)]
  have row : edge 1051 = (SortedNatTree.contains FastRows.row1051) := by
    change Data.row1051.contains = (SortedNatTree.contains FastRows.row1051)
    exact FastRows.row1051_checked.symm
  rw [row, root0Candidates35Block0_checked]
def root0Candidates35Block1 : List Nat := [11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229]
set_option maxRecDepth 4096 in
theorem root0Candidates35Block1_shape : root0Tail164 = root0Candidates35Block1 ++ root0Tail292 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates35Block1_checked : root0Candidates35Block1.filter (SortedNatTree.contains FastRows.row1051) = [19237, 19699] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates35Block1_step : candidates edge 4 3 1051 root0Tail164 seed.bound =
    [19237, 19699] ++ candidates edge 4 3 1051 root0Tail292 seed.bound := by
  rw [root0Candidates35Block1_shape, candidates_append_of_within edge 4 3 1051 root0Candidates35Block1 root0Tail292 seed.bound (by decide)]
  have row : edge 1051 = (SortedNatTree.contains FastRows.row1051) := by
    change Data.row1051.contains = (SortedNatTree.contains FastRows.row1051)
    exact FastRows.row1051_checked.symm
  rw [row, root0Candidates35Block1_checked]
def root0Candidates35Block2 : List Nat := [22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239, 24439, 24473, 24517, 24763, 24781, 24799, 24847, 24889, 24953]
set_option maxRecDepth 4096 in
theorem root0Candidates35Block2_shape : root0Tail292 = root0Candidates35Block2 ++ root0Tail326 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates35Block2_checked : root0Candidates35Block2.filter (SortedNatTree.contains FastRows.row1051) = [24763, 24781] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates35Block2_step : candidates edge 4 3 1051 root0Tail292 seed.bound =
    [24763, 24781] ++ candidates edge 4 3 1051 root0Tail326 seed.bound := by
  rw [root0Candidates35Block2_shape, candidates_append_of_within edge 4 3 1051 root0Candidates35Block2 root0Tail326 seed.bound (by decide)]
  have row : edge 1051 = (SortedNatTree.contains FastRows.row1051) := by
    change Data.row1051.contains = (SortedNatTree.contains FastRows.row1051)
    exact FastRows.row1051_checked.symm
  rw [row, root0Candidates35Block2_checked]
theorem root0Candidates35_end : candidates edge 4 3 1051 root0Tail326 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates35_checked : candidates edge 4 3 1051 root0Tail36 seed.bound = root0Candidates35 := by
  rw [root0Candidates35Block0_step, root0Candidates35Block1_step, root0Candidates35Block2_step, root0Candidates35_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child35 : search edge 3 [3, 1051] root0Candidates35 1054 seed = seed := by decide
theorem root0Step35 : scan edge (search edge 3) 4 [3] 3 root0Tail35 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail36 seed := by
  rw [root0Tail35, scan, if_neg (by decide), root0Candidates35_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail36
    (if 3 ≤ root0Candidates35.length then search edge 3 [3, 1051] root0Candidates35 1054 seed else seed) = _
  rw [if_pos (by decide), root0Child35]
def root0Candidates36 : List Nat := [5171, 8237, 8669, 17327, 20627, 22709, 23567]
def root0Candidates36Block0 : List Nat := [1193, 1237, 1481, 1531, 1607, 1627, 1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867]
set_option maxRecDepth 4096 in
theorem root0Candidates36Block0_shape : root0Tail37 = root0Candidates36Block0 ++ root0Tail165 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates36Block0_checked : root0Candidates36Block0.filter (SortedNatTree.contains FastRows.row1181) = [5171, 8237, 8669] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates36Block0_step : candidates edge 4 3 1181 root0Tail37 seed.bound =
    [5171, 8237, 8669] ++ candidates edge 4 3 1181 root0Tail165 seed.bound := by
  rw [root0Candidates36Block0_shape, candidates_append_of_within edge 4 3 1181 root0Candidates36Block0 root0Tail165 seed.bound (by decide)]
  have row : edge 1181 = (SortedNatTree.contains FastRows.row1181) := by
    change Data.row1181.contains = (SortedNatTree.contains FastRows.row1181)
    exact FastRows.row1181_checked.symm
  rw [row, root0Candidates36Block0_checked]
def root0Candidates36Block1 : List Nat := [11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271]
set_option maxRecDepth 4096 in
theorem root0Candidates36Block1_shape : root0Tail165 = root0Candidates36Block1 ++ root0Tail293 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates36Block1_checked : root0Candidates36Block1.filter (SortedNatTree.contains FastRows.row1181) = [17327, 20627] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates36Block1_step : candidates edge 4 3 1181 root0Tail165 seed.bound =
    [17327, 20627] ++ candidates edge 4 3 1181 root0Tail293 seed.bound := by
  rw [root0Candidates36Block1_shape, candidates_append_of_within edge 4 3 1181 root0Candidates36Block1 root0Tail293 seed.bound (by decide)]
  have row : edge 1181 = (SortedNatTree.contains FastRows.row1181) := by
    change Data.row1181.contains = (SortedNatTree.contains FastRows.row1181)
    exact FastRows.row1181_checked.symm
  rw [row, root0Candidates36Block1_checked]
def root0Candidates36Block2 : List Nat := [22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239, 24439, 24473, 24517, 24763, 24781, 24799, 24847]
set_option maxRecDepth 4096 in
theorem root0Candidates36Block2_shape : root0Tail293 = root0Candidates36Block2 ++ root0Tail324 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates36Block2_checked : root0Candidates36Block2.filter (SortedNatTree.contains FastRows.row1181) = [22709, 23567] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates36Block2_step : candidates edge 4 3 1181 root0Tail293 seed.bound =
    [22709, 23567] ++ candidates edge 4 3 1181 root0Tail324 seed.bound := by
  rw [root0Candidates36Block2_shape, candidates_append_of_within edge 4 3 1181 root0Candidates36Block2 root0Tail324 seed.bound (by decide)]
  have row : edge 1181 = (SortedNatTree.contains FastRows.row1181) := by
    change Data.row1181.contains = (SortedNatTree.contains FastRows.row1181)
    exact FastRows.row1181_checked.symm
  rw [row, root0Candidates36Block2_checked]
theorem root0Candidates36_end : candidates edge 4 3 1181 root0Tail324 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates36_checked : candidates edge 4 3 1181 root0Tail37 seed.bound = root0Candidates36 := by
  rw [root0Candidates36Block0_step, root0Candidates36Block1_step, root0Candidates36Block2_step, root0Candidates36_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child36 : search edge 3 [3, 1181] root0Candidates36 1184 seed = seed := by decide
theorem root0Step36 : scan edge (search edge 3) 4 [3] 3 root0Tail36 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail37 seed := by
  rw [root0Tail36, scan, if_neg (by decide), root0Candidates36_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail37
    (if 3 ≤ root0Candidates36.length then search edge 3 [3, 1181] root0Candidates36 1184 seed else seed) = _
  rw [if_pos (by decide), root0Child36]
def root0Candidates37 : List Nat := [2441, 3923, 5507, 5801, 14543, 14627]
def root0Candidates37Block0 : List Nat := [1237, 1481, 1531, 1607, 1627, 1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897]
set_option maxRecDepth 4096 in
theorem root0Candidates37Block0_shape : root0Tail38 = root0Candidates37Block0 ++ root0Tail166 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates37Block0_checked : root0Candidates37Block0.filter (SortedNatTree.contains FastRows.row1193) = [2441, 3923, 5507, 5801] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates37Block0_step : candidates edge 4 3 1193 root0Tail38 seed.bound =
    [2441, 3923, 5507, 5801] ++ candidates edge 4 3 1193 root0Tail166 seed.bound := by
  rw [root0Candidates37Block0_shape, candidates_append_of_within edge 4 3 1193 root0Candidates37Block0 root0Tail166 seed.bound (by decide)]
  have row : edge 1193 = (SortedNatTree.contains FastRows.row1193) := by
    change Data.row1193.contains = (SortedNatTree.contains FastRows.row1193)
    exact FastRows.row1193_checked.symm
  rw [row, root0Candidates37Block0_checked]
def root0Candidates37Block1 : List Nat := [11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349]
set_option maxRecDepth 4096 in
theorem root0Candidates37Block1_shape : root0Tail166 = root0Candidates37Block1 ++ root0Tail294 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates37Block1_checked : root0Candidates37Block1.filter (SortedNatTree.contains FastRows.row1193) = [14543, 14627] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates37Block1_step : candidates edge 4 3 1193 root0Tail166 seed.bound =
    [14543, 14627] ++ candidates edge 4 3 1193 root0Tail294 seed.bound := by
  rw [root0Candidates37Block1_shape, candidates_append_of_within edge 4 3 1193 root0Candidates37Block1 root0Tail294 seed.bound (by decide)]
  have row : edge 1193 = (SortedNatTree.contains FastRows.row1193) := by
    change Data.row1193.contains = (SortedNatTree.contains FastRows.row1193)
    exact FastRows.row1193_checked.symm
  rw [row, root0Candidates37Block1_checked]
def root0Candidates37Block2 : List Nat := [22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239, 24439, 24473, 24517, 24763, 24781, 24799]
set_option maxRecDepth 4096 in
theorem root0Candidates37Block2_shape : root0Tail294 = root0Candidates37Block2 ++ root0Tail323 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates37Block2_checked : root0Candidates37Block2.filter (SortedNatTree.contains FastRows.row1193) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates37Block2_step : candidates edge 4 3 1193 root0Tail294 seed.bound =
    [] ++ candidates edge 4 3 1193 root0Tail323 seed.bound := by
  rw [root0Candidates37Block2_shape, candidates_append_of_within edge 4 3 1193 root0Candidates37Block2 root0Tail323 seed.bound (by decide)]
  have row : edge 1193 = (SortedNatTree.contains FastRows.row1193) := by
    change Data.row1193.contains = (SortedNatTree.contains FastRows.row1193)
    exact FastRows.row1193_checked.symm
  rw [row, root0Candidates37Block2_checked]
theorem root0Candidates37_end : candidates edge 4 3 1193 root0Tail323 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates37_checked : candidates edge 4 3 1193 root0Tail38 seed.bound = root0Candidates37 := by
  rw [root0Candidates37Block0_step, root0Candidates37Block1_step, root0Candidates37Block2_step, root0Candidates37_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child37 : search edge 3 [3, 1193] root0Candidates37 1196 seed = seed := by decide
theorem root0Step37 : scan edge (search edge 3) 4 [3] 3 root0Tail37 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail38 seed := by
  rw [root0Tail37, scan, if_neg (by decide), root0Candidates37_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail38
    (if 3 ≤ root0Candidates37.length then search edge 3 [3, 1193] root0Candidates37 1196 seed else seed) = _
  rw [if_pos (by decide), root0Child37]
def root0Candidates38 : List Nat := [1657, 6571, 8713, 13399, 17599, 19819]
def root0Candidates38Block0 : List Nat := [1481, 1531, 1607, 1627, 1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981]
set_option maxRecDepth 4096 in
theorem root0Candidates38Block0_shape : root0Tail39 = root0Candidates38Block0 ++ root0Tail167 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates38Block0_checked : root0Candidates38Block0.filter (SortedNatTree.contains FastRows.row1237) = [1657, 6571, 8713] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates38Block0_step : candidates edge 4 3 1237 root0Tail39 seed.bound =
    [1657, 6571, 8713] ++ candidates edge 4 3 1237 root0Tail167 seed.bound := by
  rw [root0Candidates38Block0_shape, candidates_append_of_within edge 4 3 1237 root0Candidates38Block0 root0Tail167 seed.bound (by decide)]
  have row : edge 1237 = (SortedNatTree.contains FastRows.row1237) := by
    change Data.row1237.contains = (SortedNatTree.contains FastRows.row1237)
    exact FastRows.row1237_checked.symm
  rw [row, root0Candidates38Block0_checked]
def root0Candidates38Block1 : List Nat := [12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549]
set_option maxRecDepth 4096 in
theorem root0Candidates38Block1_shape : root0Tail167 = root0Candidates38Block1 ++ root0Tail295 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates38Block1_checked : root0Candidates38Block1.filter (SortedNatTree.contains FastRows.row1237) = [13399, 17599, 19819] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates38Block1_step : candidates edge 4 3 1237 root0Tail167 seed.bound =
    [13399, 17599, 19819] ++ candidates edge 4 3 1237 root0Tail295 seed.bound := by
  rw [root0Candidates38Block1_shape, candidates_append_of_within edge 4 3 1237 root0Candidates38Block1 root0Tail295 seed.bound (by decide)]
  have row : edge 1237 = (SortedNatTree.contains FastRows.row1237) := by
    change Data.row1237.contains = (SortedNatTree.contains FastRows.row1237)
    exact FastRows.row1237_checked.symm
  rw [row, root0Candidates38Block1_checked]
def root0Candidates38Block2 : List Nat := [22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239, 24439, 24473, 24517, 24763, 24781]
set_option maxRecDepth 4096 in
theorem root0Candidates38Block2_shape : root0Tail295 = root0Candidates38Block2 ++ root0Tail322 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates38Block2_checked : root0Candidates38Block2.filter (SortedNatTree.contains FastRows.row1237) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates38Block2_step : candidates edge 4 3 1237 root0Tail295 seed.bound =
    [] ++ candidates edge 4 3 1237 root0Tail322 seed.bound := by
  rw [root0Candidates38Block2_shape, candidates_append_of_within edge 4 3 1237 root0Candidates38Block2 root0Tail322 seed.bound (by decide)]
  have row : edge 1237 = (SortedNatTree.contains FastRows.row1237) := by
    change Data.row1237.contains = (SortedNatTree.contains FastRows.row1237)
    exact FastRows.row1237_checked.symm
  rw [row, root0Candidates38Block2_checked]
theorem root0Candidates38_end : candidates edge 4 3 1237 root0Tail322 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates38_checked : candidates edge 4 3 1237 root0Tail39 seed.bound = root0Candidates38 := by
  rw [root0Candidates38Block0_step, root0Candidates38Block1_step, root0Candidates38Block2_step, root0Candidates38_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child38 : search edge 3 [3, 1237] root0Candidates38 1240 seed = seed := by decide
theorem root0Step38 : scan edge (search edge 3) 4 [3] 3 root0Tail38 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail39 seed := by
  rw [root0Tail38, scan, if_neg (by decide), root0Candidates38_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail39
    (if 3 ≤ root0Candidates38.length then search edge 3 [3, 1237] root0Candidates38 1240 seed else seed) = _
  rw [if_pos (by decide), root0Child38]
def root0Candidates39 : List Nat := [13619, 22349]
def root0Candidates39Block0 : List Nat := [1531, 1607, 1627, 1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071]
set_option maxRecDepth 4096 in
theorem root0Candidates39Block0_shape : root0Tail40 = root0Candidates39Block0 ++ root0Tail168 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates39Block0_checked : root0Candidates39Block0.filter (SortedNatTree.contains FastRows.row1481) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates39Block0_step : candidates edge 4 3 1481 root0Tail40 seed.bound =
    [] ++ candidates edge 4 3 1481 root0Tail168 seed.bound := by
  rw [root0Candidates39Block0_shape, candidates_append_of_within edge 4 3 1481 root0Candidates39Block0 root0Tail168 seed.bound (by decide)]
  have row : edge 1481 = (SortedNatTree.contains FastRows.row1481) := by
    change Data.row1481.contains = (SortedNatTree.contains FastRows.row1481)
    exact FastRows.row1481_checked.symm
  rw [row, root0Candidates39Block0_checked]
def root0Candidates39Block1 : List Nat := [12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573]
set_option maxRecDepth 4096 in
theorem root0Candidates39Block1_shape : root0Tail168 = root0Candidates39Block1 ++ root0Tail296 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates39Block1_checked : root0Candidates39Block1.filter (SortedNatTree.contains FastRows.row1481) = [13619, 22349] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates39Block1_step : candidates edge 4 3 1481 root0Tail168 seed.bound =
    [13619, 22349] ++ candidates edge 4 3 1481 root0Tail296 seed.bound := by
  rw [root0Candidates39Block1_shape, candidates_append_of_within edge 4 3 1481 root0Candidates39Block1 root0Tail296 seed.bound (by decide)]
  have row : edge 1481 = (SortedNatTree.contains FastRows.row1481) := by
    change Data.row1481.contains = (SortedNatTree.contains FastRows.row1481)
    exact FastRows.row1481_checked.symm
  rw [row, root0Candidates39Block1_checked]
def root0Candidates39Block2 : List Nat := [22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239, 24439, 24473, 24517]
set_option maxRecDepth 4096 in
theorem root0Candidates39Block2_shape : root0Tail296 = root0Candidates39Block2 ++ root0Tail320 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates39Block2_checked : root0Candidates39Block2.filter (SortedNatTree.contains FastRows.row1481) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates39Block2_step : candidates edge 4 3 1481 root0Tail296 seed.bound =
    [] ++ candidates edge 4 3 1481 root0Tail320 seed.bound := by
  rw [root0Candidates39Block2_shape, candidates_append_of_within edge 4 3 1481 root0Candidates39Block2 root0Tail320 seed.bound (by decide)]
  have row : edge 1481 = (SortedNatTree.contains FastRows.row1481) := by
    change Data.row1481.contains = (SortedNatTree.contains FastRows.row1481)
    exact FastRows.row1481_checked.symm
  rw [row, root0Candidates39Block2_checked]
theorem root0Candidates39_end : candidates edge 4 3 1481 root0Tail320 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates39_checked : candidates edge 4 3 1481 root0Tail40 seed.bound = root0Candidates39 := by
  rw [root0Candidates39Block0_step, root0Candidates39Block1_step, root0Candidates39Block2_step, root0Candidates39_end]
  rfl
theorem root0Step39 : scan edge (search edge 3) 4 [3] 3 root0Tail39 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail40 seed := by
  rw [root0Tail39, scan, if_neg (by decide), root0Candidates39_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail40
    (if 3 ≤ root0Candidates39.length then search edge 3 [3, 1481] root0Candidates39 1484 seed else seed) = _
  rw [if_neg (by decide)]
def root0Candidates40 : List Nat := [5059, 5281, 11551, 12979, 22549, 23767]
def root0Candidates40Block0 : List Nat := [1607, 1627, 1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101]
set_option maxRecDepth 4096 in
theorem root0Candidates40Block0_shape : root0Tail41 = root0Candidates40Block0 ++ root0Tail169 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates40Block0_checked : root0Candidates40Block0.filter (SortedNatTree.contains FastRows.row1531) = [5059, 5281, 11551] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates40Block0_step : candidates edge 4 3 1531 root0Tail41 seed.bound =
    [5059, 5281, 11551] ++ candidates edge 4 3 1531 root0Tail169 seed.bound := by
  rw [root0Candidates40Block0_shape, candidates_append_of_within edge 4 3 1531 root0Candidates40Block0 root0Tail169 seed.bound (by decide)]
  have row : edge 1531 = (SortedNatTree.contains FastRows.row1531) := by
    change Data.row1531.contains = (SortedNatTree.contains FastRows.row1531)
    exact FastRows.row1531_checked.symm
  rw [row, root0Candidates40Block0_checked]
def root0Candidates40Block1 : List Nat := [12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613]
set_option maxRecDepth 4096 in
theorem root0Candidates40Block1_shape : root0Tail169 = root0Candidates40Block1 ++ root0Tail297 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates40Block1_checked : root0Candidates40Block1.filter (SortedNatTree.contains FastRows.row1531) = [12979, 22549] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates40Block1_step : candidates edge 4 3 1531 root0Tail169 seed.bound =
    [12979, 22549] ++ candidates edge 4 3 1531 root0Tail297 seed.bound := by
  rw [root0Candidates40Block1_shape, candidates_append_of_within edge 4 3 1531 root0Candidates40Block1 root0Tail297 seed.bound (by decide)]
  have row : edge 1531 = (SortedNatTree.contains FastRows.row1531) := by
    change Data.row1531.contains = (SortedNatTree.contains FastRows.row1531)
    exact FastRows.row1531_checked.symm
  rw [row, root0Candidates40Block1_checked]
def root0Candidates40Block2 : List Nat := [22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239, 24439, 24473]
set_option maxRecDepth 4096 in
theorem root0Candidates40Block2_shape : root0Tail297 = root0Candidates40Block2 ++ root0Tail319 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates40Block2_checked : root0Candidates40Block2.filter (SortedNatTree.contains FastRows.row1531) = [23767] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates40Block2_step : candidates edge 4 3 1531 root0Tail297 seed.bound =
    [23767] ++ candidates edge 4 3 1531 root0Tail319 seed.bound := by
  rw [root0Candidates40Block2_shape, candidates_append_of_within edge 4 3 1531 root0Candidates40Block2 root0Tail319 seed.bound (by decide)]
  have row : edge 1531 = (SortedNatTree.contains FastRows.row1531) := by
    change Data.row1531.contains = (SortedNatTree.contains FastRows.row1531)
    exact FastRows.row1531_checked.symm
  rw [row, root0Candidates40Block2_checked]
theorem root0Candidates40_end : candidates edge 4 3 1531 root0Tail319 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates40_checked : candidates edge 4 3 1531 root0Tail41 seed.bound = root0Candidates40 := by
  rw [root0Candidates40Block0_step, root0Candidates40Block1_step, root0Candidates40Block2_step, root0Candidates40_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child40 : search edge 3 [3, 1531] root0Candidates40 1534 seed = seed := by decide
theorem root0Step40 : scan edge (search edge 3) 4 [3] 3 root0Tail40 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail41 seed := by
  rw [root0Tail40, scan, if_neg (by decide), root0Candidates40_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail41
    (if 3 ≤ root0Candidates40.length then search edge 3 [3, 1531] root0Candidates40 1534 seed else seed) = _
  rw [if_pos (by decide), root0Child40]
def root0Candidates41 : List Nat := [5099, 19919, 20939]
def root0Candidates41Block0 : List Nat := [1627, 1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203]
set_option maxRecDepth 4096 in
theorem root0Candidates41Block0_shape : root0Tail42 = root0Candidates41Block0 ++ root0Tail170 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates41Block0_checked : root0Candidates41Block0.filter (SortedNatTree.contains FastRows.row1607) = [5099] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates41Block0_step : candidates edge 4 3 1607 root0Tail42 seed.bound =
    [5099] ++ candidates edge 4 3 1607 root0Tail170 seed.bound := by
  rw [root0Candidates41Block0_shape, candidates_append_of_within edge 4 3 1607 root0Candidates41Block0 root0Tail170 seed.bound (by decide)]
  have row : edge 1607 = (SortedNatTree.contains FastRows.row1607) := by
    change Data.row1607.contains = (SortedNatTree.contains FastRows.row1607)
    exact FastRows.row1607_checked.symm
  rw [row, root0Candidates41Block0_checked]
def root0Candidates41Block1 : List Nat := [12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709]
set_option maxRecDepth 4096 in
theorem root0Candidates41Block1_shape : root0Tail170 = root0Candidates41Block1 ++ root0Tail298 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates41Block1_checked : root0Candidates41Block1.filter (SortedNatTree.contains FastRows.row1607) = [19919, 20939] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates41Block1_step : candidates edge 4 3 1607 root0Tail170 seed.bound =
    [19919, 20939] ++ candidates edge 4 3 1607 root0Tail298 seed.bound := by
  rw [root0Candidates41Block1_shape, candidates_append_of_within edge 4 3 1607 root0Candidates41Block1 root0Tail298 seed.bound (by decide)]
  have row : edge 1607 = (SortedNatTree.contains FastRows.row1607) := by
    change Data.row1607.contains = (SortedNatTree.contains FastRows.row1607)
    exact FastRows.row1607_checked.symm
  rw [row, root0Candidates41Block1_checked]
def root0Candidates41Block2 : List Nat := [22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239]
set_option maxRecDepth 4096 in
theorem root0Candidates41Block2_shape : root0Tail298 = root0Candidates41Block2 ++ root0Tail317 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates41Block2_checked : root0Candidates41Block2.filter (SortedNatTree.contains FastRows.row1607) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates41Block2_step : candidates edge 4 3 1607 root0Tail298 seed.bound =
    [] ++ candidates edge 4 3 1607 root0Tail317 seed.bound := by
  rw [root0Candidates41Block2_shape, candidates_append_of_within edge 4 3 1607 root0Candidates41Block2 root0Tail317 seed.bound (by decide)]
  have row : edge 1607 = (SortedNatTree.contains FastRows.row1607) := by
    change Data.row1607.contains = (SortedNatTree.contains FastRows.row1607)
    exact FastRows.row1607_checked.symm
  rw [row, root0Candidates41Block2_checked]
theorem root0Candidates41_end : candidates edge 4 3 1607 root0Tail317 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates41_checked : candidates edge 4 3 1607 root0Tail42 seed.bound = root0Candidates41 := by
  rw [root0Candidates41Block0_step, root0Candidates41Block1_step, root0Candidates41Block2_step, root0Candidates41_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child41 : search edge 3 [3, 1607] root0Candidates41 1610 seed = seed := by decide
theorem root0Step41 : scan edge (search edge 3) 4 [3] 3 root0Tail41 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail42 seed := by
  rw [root0Tail41, scan, if_neg (by decide), root0Candidates41_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail42
    (if 3 ≤ root0Candidates41.length then search edge 3 [3, 1607] root0Candidates41 1610 seed else seed) = _
  rw [if_pos (by decide), root0Child41]
def root0Candidates42 : List Nat := [2917, 3637, 3769, 6277, 7159, 13381, 16363, 21313]
def root0Candidates42Block0 : List Nat := [1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253]
set_option maxRecDepth 4096 in
theorem root0Candidates42Block0_shape : root0Tail43 = root0Candidates42Block0 ++ root0Tail171 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates42Block0_checked : root0Candidates42Block0.filter (SortedNatTree.contains FastRows.row1627) = [2917, 3637, 3769, 6277, 7159] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates42Block0_step : candidates edge 4 3 1627 root0Tail43 seed.bound =
    [2917, 3637, 3769, 6277, 7159] ++ candidates edge 4 3 1627 root0Tail171 seed.bound := by
  rw [root0Candidates42Block0_shape, candidates_append_of_within edge 4 3 1627 root0Candidates42Block0 root0Tail171 seed.bound (by decide)]
  have row : edge 1627 = (SortedNatTree.contains FastRows.row1627) := by
    change Data.row1627.contains = (SortedNatTree.contains FastRows.row1627)
    exact FastRows.row1627_checked.symm
  rw [row, root0Candidates42Block0_checked]
def root0Candidates42Block1 : List Nat := [12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769]
set_option maxRecDepth 4096 in
theorem root0Candidates42Block1_shape : root0Tail171 = root0Candidates42Block1 ++ root0Tail299 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates42Block1_checked : root0Candidates42Block1.filter (SortedNatTree.contains FastRows.row1627) = [13381, 16363, 21313] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates42Block1_step : candidates edge 4 3 1627 root0Tail171 seed.bound =
    [13381, 16363, 21313] ++ candidates edge 4 3 1627 root0Tail299 seed.bound := by
  rw [root0Candidates42Block1_shape, candidates_append_of_within edge 4 3 1627 root0Candidates42Block1 root0Tail299 seed.bound (by decide)]
  have row : edge 1627 = (SortedNatTree.contains FastRows.row1627) := by
    change Data.row1627.contains = (SortedNatTree.contains FastRows.row1627)
    exact FastRows.row1627_checked.symm
  rw [row, root0Candidates42Block1_checked]
def root0Candidates42Block2 : List Nat := [22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239]
set_option maxRecDepth 4096 in
theorem root0Candidates42Block2_shape : root0Tail299 = root0Candidates42Block2 ++ root0Tail317 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates42Block2_checked : root0Candidates42Block2.filter (SortedNatTree.contains FastRows.row1627) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates42Block2_step : candidates edge 4 3 1627 root0Tail299 seed.bound =
    [] ++ candidates edge 4 3 1627 root0Tail317 seed.bound := by
  rw [root0Candidates42Block2_shape, candidates_append_of_within edge 4 3 1627 root0Candidates42Block2 root0Tail317 seed.bound (by decide)]
  have row : edge 1627 = (SortedNatTree.contains FastRows.row1627) := by
    change Data.row1627.contains = (SortedNatTree.contains FastRows.row1627)
    exact FastRows.row1627_checked.symm
  rw [row, root0Candidates42Block2_checked]
theorem root0Candidates42_end : candidates edge 4 3 1627 root0Tail317 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates42_checked : candidates edge 4 3 1627 root0Tail43 seed.bound = root0Candidates42 := by
  rw [root0Candidates42Block0_step, root0Candidates42Block1_step, root0Candidates42Block2_step, root0Candidates42_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child42 : search edge 3 [3, 1627] root0Candidates42 1630 seed = seed := by decide
theorem root0Step42 : scan edge (search edge 3) 4 [3] 3 root0Tail42 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail43 seed := by
  rw [root0Tail42, scan, if_neg (by decide), root0Candidates42_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail43
    (if 3 ≤ root0Candidates42.length then search edge 3 [3, 1627] root0Candidates42 1630 seed else seed) = _
  rw [if_pos (by decide), root0Child42]
def root0Candidates43 : List Nat := [1699, 2917, 5059, 10627, 17029, 22549, 23473]
def root0Candidates43Block0 : List Nat := [1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269]
set_option maxRecDepth 4096 in
theorem root0Candidates43Block0_shape : root0Tail44 = root0Candidates43Block0 ++ root0Tail172 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates43Block0_checked : root0Candidates43Block0.filter (SortedNatTree.contains FastRows.row1657) = [1699, 2917, 5059, 10627] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates43Block0_step : candidates edge 4 3 1657 root0Tail44 seed.bound =
    [1699, 2917, 5059, 10627] ++ candidates edge 4 3 1657 root0Tail172 seed.bound := by
  rw [root0Candidates43Block0_shape, candidates_append_of_within edge 4 3 1657 root0Candidates43Block0 root0Tail172 seed.bound (by decide)]
  have row : edge 1657 = (SortedNatTree.contains FastRows.row1657) := by
    change Data.row1657.contains = (SortedNatTree.contains FastRows.row1657)
    exact FastRows.row1657_checked.symm
  rw [row, root0Candidates43Block0_checked]
def root0Candidates43Block1 : List Nat := [12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859]
set_option maxRecDepth 4096 in
theorem root0Candidates43Block1_shape : root0Tail172 = root0Candidates43Block1 ++ root0Tail300 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates43Block1_checked : root0Candidates43Block1.filter (SortedNatTree.contains FastRows.row1657) = [17029, 22549] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates43Block1_step : candidates edge 4 3 1657 root0Tail172 seed.bound =
    [17029, 22549] ++ candidates edge 4 3 1657 root0Tail300 seed.bound := by
  rw [root0Candidates43Block1_shape, candidates_append_of_within edge 4 3 1657 root0Candidates43Block1 root0Tail300 seed.bound (by decide)]
  have row : edge 1657 = (SortedNatTree.contains FastRows.row1657) := by
    change Data.row1657.contains = (SortedNatTree.contains FastRows.row1657)
    exact FastRows.row1657_checked.symm
  rw [row, root0Candidates43Block1_checked]
def root0Candidates43Block2 : List Nat := [22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239]
set_option maxRecDepth 4096 in
theorem root0Candidates43Block2_shape : root0Tail300 = root0Candidates43Block2 ++ root0Tail317 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates43Block2_checked : root0Candidates43Block2.filter (SortedNatTree.contains FastRows.row1657) = [23473] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates43Block2_step : candidates edge 4 3 1657 root0Tail300 seed.bound =
    [23473] ++ candidates edge 4 3 1657 root0Tail317 seed.bound := by
  rw [root0Candidates43Block2_shape, candidates_append_of_within edge 4 3 1657 root0Candidates43Block2 root0Tail317 seed.bound (by decide)]
  have row : edge 1657 = (SortedNatTree.contains FastRows.row1657) := by
    change Data.row1657.contains = (SortedNatTree.contains FastRows.row1657)
    exact FastRows.row1657_checked.symm
  rw [row, root0Candidates43Block2_checked]
theorem root0Candidates43_end : candidates edge 4 3 1657 root0Tail317 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates43_checked : candidates edge 4 3 1657 root0Tail44 seed.bound = root0Candidates43 := by
  rw [root0Candidates43Block0_step, root0Candidates43Block1_step, root0Candidates43Block2_step, root0Candidates43_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child43 : search edge 3 [3, 1657] root0Candidates43 1660 seed = seed := by decide
theorem root0Step43 : scan edge (search edge 3) 4 [3] 3 root0Tail43 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail44 seed := by
  rw [root0Tail43, scan, if_neg (by decide), root0Candidates43_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail44
    (if 3 ≤ root0Candidates43.length then search edge 3 [3, 1657] root0Candidates43 1660 seed else seed) = _
  rw [if_pos (by decide), root0Child43]
def root0Candidates44 : List Nat := [5449, 7963, 14107, 20899, 23371]
def root0Candidates44Block0 : List Nat := [1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343]
set_option maxRecDepth 4096 in
theorem root0Candidates44Block0_shape : root0Tail45 = root0Candidates44Block0 ++ root0Tail173 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates44Block0_checked : root0Candidates44Block0.filter (SortedNatTree.contains FastRows.row1663) = [5449, 7963] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates44Block0_step : candidates edge 4 3 1663 root0Tail45 seed.bound =
    [5449, 7963] ++ candidates edge 4 3 1663 root0Tail173 seed.bound := by
  rw [root0Candidates44Block0_shape, candidates_append_of_within edge 4 3 1663 root0Candidates44Block0 root0Tail173 seed.bound (by decide)]
  have row : edge 1663 = (SortedNatTree.contains FastRows.row1663) := by
    change Data.row1663.contains = (SortedNatTree.contains FastRows.row1663)
    exact FastRows.row1663_checked.symm
  rw [row, root0Candidates44Block0_checked]
def root0Candidates44Block1 : List Nat := [12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871]
set_option maxRecDepth 4096 in
theorem root0Candidates44Block1_shape : root0Tail173 = root0Candidates44Block1 ++ root0Tail301 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates44Block1_checked : root0Candidates44Block1.filter (SortedNatTree.contains FastRows.row1663) = [14107, 20899] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates44Block1_step : candidates edge 4 3 1663 root0Tail173 seed.bound =
    [14107, 20899] ++ candidates edge 4 3 1663 root0Tail301 seed.bound := by
  rw [root0Candidates44Block1_shape, candidates_append_of_within edge 4 3 1663 root0Candidates44Block1 root0Tail301 seed.bound (by decide)]
  have row : edge 1663 = (SortedNatTree.contains FastRows.row1663) := by
    change Data.row1663.contains = (SortedNatTree.contains FastRows.row1663)
    exact FastRows.row1663_checked.symm
  rw [row, root0Candidates44Block1_checked]
def root0Candidates44Block2 : List Nat := [22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239]
set_option maxRecDepth 4096 in
theorem root0Candidates44Block2_shape : root0Tail301 = root0Candidates44Block2 ++ root0Tail317 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates44Block2_checked : root0Candidates44Block2.filter (SortedNatTree.contains FastRows.row1663) = [23371] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates44Block2_step : candidates edge 4 3 1663 root0Tail301 seed.bound =
    [23371] ++ candidates edge 4 3 1663 root0Tail317 seed.bound := by
  rw [root0Candidates44Block2_shape, candidates_append_of_within edge 4 3 1663 root0Candidates44Block2 root0Tail317 seed.bound (by decide)]
  have row : edge 1663 = (SortedNatTree.contains FastRows.row1663) := by
    change Data.row1663.contains = (SortedNatTree.contains FastRows.row1663)
    exact FastRows.row1663_checked.symm
  rw [row, root0Candidates44Block2_checked]
theorem root0Candidates44_end : candidates edge 4 3 1663 root0Tail317 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates44_checked : candidates edge 4 3 1663 root0Tail45 seed.bound = root0Candidates44 := by
  rw [root0Candidates44Block0_step, root0Candidates44Block1_step, root0Candidates44Block2_step, root0Candidates44_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child44 : search edge 3 [3, 1663] root0Candidates44 1666 seed = seed := by decide
theorem root0Step44 : scan edge (search edge 3) 4 [3] 3 root0Tail44 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail45 seed := by
  rw [root0Tail44, scan, if_neg (by decide), root0Candidates44_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail45
    (if 3 ≤ root0Candidates44.length then search edge 3 [3, 1663] root0Candidates44 1666 seed else seed) = _
  rw [if_pos (by decide), root0Child44]
def root0Candidates45 : List Nat := [5051, 5531, 5573, 6353, 10091, 22877]
def root0Candidates45Block0 : List Nat := [1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413]
set_option maxRecDepth 4096 in
theorem root0Candidates45Block0_shape : root0Tail46 = root0Candidates45Block0 ++ root0Tail174 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates45Block0_checked : root0Candidates45Block0.filter (SortedNatTree.contains FastRows.row1667) = [5051, 5531, 5573, 6353, 10091] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates45Block0_step : candidates edge 4 3 1667 root0Tail46 seed.bound =
    [5051, 5531, 5573, 6353, 10091] ++ candidates edge 4 3 1667 root0Tail174 seed.bound := by
  rw [root0Candidates45Block0_shape, candidates_append_of_within edge 4 3 1667 root0Candidates45Block0 root0Tail174 seed.bound (by decide)]
  have row : edge 1667 = (SortedNatTree.contains FastRows.row1667) := by
    change Data.row1667.contains = (SortedNatTree.contains FastRows.row1667)
    exact FastRows.row1667_checked.symm
  rw [row, root0Candidates45Block0_checked]
def root0Candidates45Block1 : List Nat := [12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877]
set_option maxRecDepth 4096 in
theorem root0Candidates45Block1_shape : root0Tail174 = root0Candidates45Block1 ++ root0Tail302 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates45Block1_checked : root0Candidates45Block1.filter (SortedNatTree.contains FastRows.row1667) = [22877] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates45Block1_step : candidates edge 4 3 1667 root0Tail174 seed.bound =
    [22877] ++ candidates edge 4 3 1667 root0Tail302 seed.bound := by
  rw [root0Candidates45Block1_shape, candidates_append_of_within edge 4 3 1667 root0Candidates45Block1 root0Tail302 seed.bound (by decide)]
  have row : edge 1667 = (SortedNatTree.contains FastRows.row1667) := by
    change Data.row1667.contains = (SortedNatTree.contains FastRows.row1667)
    exact FastRows.row1667_checked.symm
  rw [row, root0Candidates45Block1_checked]
def root0Candidates45Block2 : List Nat := [22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239]
set_option maxRecDepth 4096 in
theorem root0Candidates45Block2_shape : root0Tail302 = root0Candidates45Block2 ++ root0Tail317 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates45Block2_checked : root0Candidates45Block2.filter (SortedNatTree.contains FastRows.row1667) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates45Block2_step : candidates edge 4 3 1667 root0Tail302 seed.bound =
    [] ++ candidates edge 4 3 1667 root0Tail317 seed.bound := by
  rw [root0Candidates45Block2_shape, candidates_append_of_within edge 4 3 1667 root0Candidates45Block2 root0Tail317 seed.bound (by decide)]
  have row : edge 1667 = (SortedNatTree.contains FastRows.row1667) := by
    change Data.row1667.contains = (SortedNatTree.contains FastRows.row1667)
    exact FastRows.row1667_checked.symm
  rw [row, root0Candidates45Block2_checked]
theorem root0Candidates45_end : candidates edge 4 3 1667 root0Tail317 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates45_checked : candidates edge 4 3 1667 root0Tail46 seed.bound = root0Candidates45 := by
  rw [root0Candidates45Block0_step, root0Candidates45Block1_step, root0Candidates45Block2_step, root0Candidates45_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child45 : search edge 3 [3, 1667] root0Candidates45 1670 seed = seed := by decide
theorem root0Step45 : scan edge (search edge 3) 4 [3] 3 root0Tail45 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail46 seed := by
  rw [root0Tail45, scan, if_neg (by decide), root0Candidates45_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail46
    (if 3 ≤ root0Candidates45.length then search edge 3 [3, 1667] root0Candidates45 1670 seed else seed) = _
  rw [if_pos (by decide), root0Child45]
def root0Candidates46 : List Nat := [6571, 8713, 12343, 13399, 16339]
def root0Candidates46Block0 : List Nat := [1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451]
set_option maxRecDepth 4096 in
theorem root0Candidates46Block0_shape : root0Tail47 = root0Candidates46Block0 ++ root0Tail175 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates46Block0_checked : root0Candidates46Block0.filter (SortedNatTree.contains FastRows.row1699) = [6571, 8713, 12343] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates46Block0_step : candidates edge 4 3 1699 root0Tail47 seed.bound =
    [6571, 8713, 12343] ++ candidates edge 4 3 1699 root0Tail175 seed.bound := by
  rw [root0Candidates46Block0_shape, candidates_append_of_within edge 4 3 1699 root0Candidates46Block0 root0Tail175 seed.bound (by decide)]
  have row : edge 1699 = (SortedNatTree.contains FastRows.row1699) := by
    change Data.row1699.contains = (SortedNatTree.contains FastRows.row1699)
    exact FastRows.row1699_checked.symm
  rw [row, root0Candidates46Block0_checked]
def root0Candidates46Block1 : List Nat := [12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921]
set_option maxRecDepth 4096 in
theorem root0Candidates46Block1_shape : root0Tail175 = root0Candidates46Block1 ++ root0Tail303 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates46Block1_checked : root0Candidates46Block1.filter (SortedNatTree.contains FastRows.row1699) = [13399, 16339] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates46Block1_step : candidates edge 4 3 1699 root0Tail175 seed.bound =
    [13399, 16339] ++ candidates edge 4 3 1699 root0Tail303 seed.bound := by
  rw [root0Candidates46Block1_shape, candidates_append_of_within edge 4 3 1699 root0Candidates46Block1 root0Tail303 seed.bound (by decide)]
  have row : edge 1699 = (SortedNatTree.contains FastRows.row1699) := by
    change Data.row1699.contains = (SortedNatTree.contains FastRows.row1699)
    exact FastRows.row1699_checked.symm
  rw [row, root0Candidates46Block1_checked]
def root0Candidates46Block2 : List Nat := [23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767, 24151, 24179, 24239]
set_option maxRecDepth 4096 in
theorem root0Candidates46Block2_shape : root0Tail303 = root0Candidates46Block2 ++ root0Tail317 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates46Block2_checked : root0Candidates46Block2.filter (SortedNatTree.contains FastRows.row1699) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates46Block2_step : candidates edge 4 3 1699 root0Tail303 seed.bound =
    [] ++ candidates edge 4 3 1699 root0Tail317 seed.bound := by
  rw [root0Candidates46Block2_shape, candidates_append_of_within edge 4 3 1699 root0Candidates46Block2 root0Tail317 seed.bound (by decide)]
  have row : edge 1699 = (SortedNatTree.contains FastRows.row1699) := by
    change Data.row1699.contains = (SortedNatTree.contains FastRows.row1699)
    exact FastRows.row1699_checked.symm
  rw [row, root0Candidates46Block2_checked]
theorem root0Candidates46_end : candidates edge 4 3 1699 root0Tail317 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates46_checked : candidates edge 4 3 1699 root0Tail47 seed.bound = root0Candidates46 := by
  rw [root0Candidates46Block0_step, root0Candidates46Block1_step, root0Candidates46Block2_step, root0Candidates46_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child46 : search edge 3 [3, 1699] root0Candidates46 1702 seed = seed := by decide
theorem root0Step46 : scan edge (search edge 3) 4 [3] 3 root0Tail46 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail47 seed := by
  rw [root0Tail46, scan, if_neg (by decide), root0Candidates46_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail47
    (if 3 ≤ root0Candidates46.length then search edge 3 [3, 1699] root0Candidates46 1702 seed else seed) = _
  rw [if_pos (by decide), root0Child46]
def root0Candidates47 : List Nat := [5099, 11867, 14543, 17327, 17489, 22229, 22613]
def root0Candidates47Block0 : List Nat := [2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601]
set_option maxRecDepth 4096 in
theorem root0Candidates47Block0_shape : root0Tail48 = root0Candidates47Block0 ++ root0Tail176 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates47Block0_checked : root0Candidates47Block0.filter (SortedNatTree.contains FastRows.row1907) = [5099, 11867] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates47Block0_step : candidates edge 4 3 1907 root0Tail48 seed.bound =
    [5099, 11867] ++ candidates edge 4 3 1907 root0Tail176 seed.bound := by
  rw [root0Candidates47Block0_shape, candidates_append_of_within edge 4 3 1907 root0Candidates47Block0 root0Tail176 seed.bound (by decide)]
  have row : edge 1907 = (SortedNatTree.contains FastRows.row1907) := by
    change Data.row1907.contains = (SortedNatTree.contains FastRows.row1907)
    exact FastRows.row1907_checked.symm
  rw [row, root0Candidates47Block0_checked]
def root0Candidates47Block1 : List Nat := [12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027]
set_option maxRecDepth 4096 in
theorem root0Candidates47Block1_shape : root0Tail176 = root0Candidates47Block1 ++ root0Tail304 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates47Block1_checked : root0Candidates47Block1.filter (SortedNatTree.contains FastRows.row1907) = [14543, 17327, 17489, 22229, 22613] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates47Block1_step : candidates edge 4 3 1907 root0Tail176 seed.bound =
    [14543, 17327, 17489, 22229, 22613] ++ candidates edge 4 3 1907 root0Tail304 seed.bound := by
  rw [root0Candidates47Block1_shape, candidates_append_of_within edge 4 3 1907 root0Candidates47Block1 root0Tail304 seed.bound (by decide)]
  have row : edge 1907 = (SortedNatTree.contains FastRows.row1907) := by
    change Data.row1907.contains = (SortedNatTree.contains FastRows.row1907)
    exact FastRows.row1907_checked.symm
  rw [row, root0Candidates47Block1_checked]
def root0Candidates47Block2 : List Nat := [23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767]
set_option maxRecDepth 4096 in
theorem root0Candidates47Block2_shape : root0Tail304 = root0Candidates47Block2 ++ root0Tail314 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates47Block2_checked : root0Candidates47Block2.filter (SortedNatTree.contains FastRows.row1907) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates47Block2_step : candidates edge 4 3 1907 root0Tail304 seed.bound =
    [] ++ candidates edge 4 3 1907 root0Tail314 seed.bound := by
  rw [root0Candidates47Block2_shape, candidates_append_of_within edge 4 3 1907 root0Candidates47Block2 root0Tail314 seed.bound (by decide)]
  have row : edge 1907 = (SortedNatTree.contains FastRows.row1907) := by
    change Data.row1907.contains = (SortedNatTree.contains FastRows.row1907)
    exact FastRows.row1907_checked.symm
  rw [row, root0Candidates47Block2_checked]
theorem root0Candidates47_end : candidates edge 4 3 1907 root0Tail314 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates47_checked : candidates edge 4 3 1907 root0Tail48 seed.bound = root0Candidates47 := by
  rw [root0Candidates47Block0_step, root0Candidates47Block1_step, root0Candidates47Block2_step, root0Candidates47_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child47 : search edge 3 [3, 1907] root0Candidates47 1910 seed = seed := by decide
theorem root0Step47 : scan edge (search edge 3) 4 [3] 3 root0Tail47 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail48 seed := by
  rw [root0Tail47, scan, if_neg (by decide), root0Candidates47_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail48
    (if 3 ≤ root0Candidates47.length then search edge 3 [3, 1907] root0Candidates47 1910 seed else seed) = _
  rw [if_pos (by decide), root0Child47]
def root0Candidates48 : List Nat := [2297, 7253, 8219, 8747, 14627, 15461, 18077, 21017]
def root0Candidates48Block0 : List Nat := [2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703]
set_option maxRecDepth 4096 in
theorem root0Candidates48Block0_shape : root0Tail49 = root0Candidates48Block0 ++ root0Tail177 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates48Block0_checked : root0Candidates48Block0.filter (SortedNatTree.contains FastRows.row2069) = [2297, 7253, 8219, 8747] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates48Block0_step : candidates edge 4 3 2069 root0Tail49 seed.bound =
    [2297, 7253, 8219, 8747] ++ candidates edge 4 3 2069 root0Tail177 seed.bound := by
  rw [root0Candidates48Block0_shape, candidates_append_of_within edge 4 3 2069 root0Candidates48Block0 root0Tail177 seed.bound (by decide)]
  have row : edge 2069 = (SortedNatTree.contains FastRows.row2069) := by
    change Data.row2069.contains = (SortedNatTree.contains FastRows.row2069)
    exact FastRows.row2069_checked.symm
  rw [row, root0Candidates48Block0_checked]
def root0Candidates48Block1 : List Nat := [12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087]
set_option maxRecDepth 4096 in
theorem root0Candidates48Block1_shape : root0Tail177 = root0Candidates48Block1 ++ root0Tail305 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates48Block1_checked : root0Candidates48Block1.filter (SortedNatTree.contains FastRows.row2069) = [14627, 15461, 18077, 21017] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates48Block1_step : candidates edge 4 3 2069 root0Tail177 seed.bound =
    [14627, 15461, 18077, 21017] ++ candidates edge 4 3 2069 root0Tail305 seed.bound := by
  rw [root0Candidates48Block1_shape, candidates_append_of_within edge 4 3 2069 root0Candidates48Block1 root0Tail305 seed.bound (by decide)]
  have row : edge 2069 = (SortedNatTree.contains FastRows.row2069) := by
    change Data.row2069.contains = (SortedNatTree.contains FastRows.row2069)
    exact FastRows.row2069_checked.symm
  rw [row, root0Candidates48Block1_checked]
def root0Candidates48Block2 : List Nat := [23201, 23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767]
set_option maxRecDepth 4096 in
theorem root0Candidates48Block2_shape : root0Tail305 = root0Candidates48Block2 ++ root0Tail314 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates48Block2_checked : root0Candidates48Block2.filter (SortedNatTree.contains FastRows.row2069) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates48Block2_step : candidates edge 4 3 2069 root0Tail305 seed.bound =
    [] ++ candidates edge 4 3 2069 root0Tail314 seed.bound := by
  rw [root0Candidates48Block2_shape, candidates_append_of_within edge 4 3 2069 root0Candidates48Block2 root0Tail314 seed.bound (by decide)]
  have row : edge 2069 = (SortedNatTree.contains FastRows.row2069) := by
    change Data.row2069.contains = (SortedNatTree.contains FastRows.row2069)
    exact FastRows.row2069_checked.symm
  rw [row, root0Candidates48Block2_checked]
theorem root0Candidates48_end : candidates edge 4 3 2069 root0Tail314 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates48_checked : candidates edge 4 3 2069 root0Tail49 seed.bound = root0Candidates48 := by
  rw [root0Candidates48Block0_step, root0Candidates48Block1_step, root0Candidates48Block2_step, root0Candidates48_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child48 : search edge 3 [3, 2069] root0Candidates48 2072 seed = seed := by decide
theorem root0Step48 : scan edge (search edge 3) 4 [3] 3 root0Tail48 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail49 seed := by
  rw [root0Tail48, scan, if_neg (by decide), root0Candidates48_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail49
    (if 3 ≤ root0Candidates48.length then search edge 3 [3, 2069] root0Candidates48 2072 seed else seed) = _
  rw [if_pos (by decide), root0Child48]
def root0Candidates49 : List Nat := [6871, 13381, 13399, 18523, 20521, 23251]
def root0Candidates49Block0 : List Nat := [2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841]
set_option maxRecDepth 4096 in
theorem root0Candidates49Block0_shape : root0Tail50 = root0Candidates49Block0 ++ root0Tail178 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates49Block0_checked : root0Candidates49Block0.filter (SortedNatTree.contains FastRows.row2143) = [6871] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates49Block0_step : candidates edge 4 3 2143 root0Tail50 seed.bound =
    [6871] ++ candidates edge 4 3 2143 root0Tail178 seed.bound := by
  rw [root0Candidates49Block0_shape, candidates_append_of_within edge 4 3 2143 root0Candidates49Block0 root0Tail178 seed.bound (by decide)]
  have row : edge 2143 = (SortedNatTree.contains FastRows.row2143) := by
    change Data.row2143.contains = (SortedNatTree.contains FastRows.row2143)
    exact FastRows.row2143_checked.symm
  rw [row, root0Candidates49Block0_checked]
def root0Candidates49Block1 : List Nat := [12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201]
set_option maxRecDepth 4096 in
theorem root0Candidates49Block1_shape : root0Tail178 = root0Candidates49Block1 ++ root0Tail306 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates49Block1_checked : root0Candidates49Block1.filter (SortedNatTree.contains FastRows.row2143) = [13381, 13399, 18523, 20521] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates49Block1_step : candidates edge 4 3 2143 root0Tail178 seed.bound =
    [13381, 13399, 18523, 20521] ++ candidates edge 4 3 2143 root0Tail306 seed.bound := by
  rw [root0Candidates49Block1_shape, candidates_append_of_within edge 4 3 2143 root0Candidates49Block1 root0Tail306 seed.bound (by decide)]
  have row : edge 2143 = (SortedNatTree.contains FastRows.row2143) := by
    change Data.row2143.contains = (SortedNatTree.contains FastRows.row2143)
    exact FastRows.row2143_checked.symm
  rw [row, root0Candidates49Block1_checked]
def root0Candidates49Block2 : List Nat := [23251, 23369, 23371, 23473, 23549, 23567, 23581, 23767]
set_option maxRecDepth 4096 in
theorem root0Candidates49Block2_shape : root0Tail306 = root0Candidates49Block2 ++ root0Tail314 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates49Block2_checked : root0Candidates49Block2.filter (SortedNatTree.contains FastRows.row2143) = [23251] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates49Block2_step : candidates edge 4 3 2143 root0Tail306 seed.bound =
    [23251] ++ candidates edge 4 3 2143 root0Tail314 seed.bound := by
  rw [root0Candidates49Block2_shape, candidates_append_of_within edge 4 3 2143 root0Candidates49Block2 root0Tail314 seed.bound (by decide)]
  have row : edge 2143 = (SortedNatTree.contains FastRows.row2143) := by
    change Data.row2143.contains = (SortedNatTree.contains FastRows.row2143)
    exact FastRows.row2143_checked.symm
  rw [row, root0Candidates49Block2_checked]
theorem root0Candidates49_end : candidates edge 4 3 2143 root0Tail314 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates49_checked : candidates edge 4 3 2143 root0Tail50 seed.bound = root0Candidates49 := by
  rw [root0Candidates49Block0_step, root0Candidates49Block1_step, root0Candidates49Block2_step, root0Candidates49_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child49 : search edge 3 [3, 2143] root0Candidates49 2146 seed = seed := by decide
theorem root0Step49 : scan edge (search edge 3) 4 [3] 3 root0Tail49 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail50 seed := by
  rw [root0Tail49, scan, if_neg (by decide), root0Candidates49_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail50
    (if 3 ≤ root0Candidates49.length then search edge 3 [3, 2143] root0Candidates49 2146 seed else seed) = _
  rw [if_pos (by decide), root0Child49]
def root0Candidates50 : List Nat := [4253, 10091]
def root0Candidates50Block0 : List Nat := [2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899]
set_option maxRecDepth 4096 in
theorem root0Candidates50Block0_shape : root0Tail51 = root0Candidates50Block0 ++ root0Tail179 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates50Block0_checked : root0Candidates50Block0.filter (SortedNatTree.contains FastRows.row2213) = [4253, 10091] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates50Block0_step : candidates edge 4 3 2213 root0Tail51 seed.bound =
    [4253, 10091] ++ candidates edge 4 3 2213 root0Tail179 seed.bound := by
  rw [root0Candidates50Block0_shape, candidates_append_of_within edge 4 3 2213 root0Candidates50Block0 root0Tail179 seed.bound (by decide)]
  have row : edge 2213 = (SortedNatTree.contains FastRows.row2213) := by
    change Data.row2213.contains = (SortedNatTree.contains FastRows.row2213)
    exact FastRows.row2213_checked.symm
  rw [row, root0Candidates50Block0_checked]
def root0Candidates50Block1 : List Nat := [12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251]
set_option maxRecDepth 4096 in
theorem root0Candidates50Block1_shape : root0Tail179 = root0Candidates50Block1 ++ root0Tail307 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates50Block1_checked : root0Candidates50Block1.filter (SortedNatTree.contains FastRows.row2213) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates50Block1_step : candidates edge 4 3 2213 root0Tail179 seed.bound =
    [] ++ candidates edge 4 3 2213 root0Tail307 seed.bound := by
  rw [root0Candidates50Block1_shape, candidates_append_of_within edge 4 3 2213 root0Candidates50Block1 root0Tail307 seed.bound (by decide)]
  have row : edge 2213 = (SortedNatTree.contains FastRows.row2213) := by
    change Data.row2213.contains = (SortedNatTree.contains FastRows.row2213)
    exact FastRows.row2213_checked.symm
  rw [row, root0Candidates50Block1_checked]
def root0Candidates50Block2 : List Nat := [23369, 23371, 23473, 23549, 23567, 23581, 23767]
set_option maxRecDepth 4096 in
theorem root0Candidates50Block2_shape : root0Tail307 = root0Candidates50Block2 ++ root0Tail314 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates50Block2_checked : root0Candidates50Block2.filter (SortedNatTree.contains FastRows.row2213) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates50Block2_step : candidates edge 4 3 2213 root0Tail307 seed.bound =
    [] ++ candidates edge 4 3 2213 root0Tail314 seed.bound := by
  rw [root0Candidates50Block2_shape, candidates_append_of_within edge 4 3 2213 root0Candidates50Block2 root0Tail314 seed.bound (by decide)]
  have row : edge 2213 = (SortedNatTree.contains FastRows.row2213) := by
    change Data.row2213.contains = (SortedNatTree.contains FastRows.row2213)
    exact FastRows.row2213_checked.symm
  rw [row, root0Candidates50Block2_checked]
theorem root0Candidates50_end : candidates edge 4 3 2213 root0Tail314 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates50_checked : candidates edge 4 3 2213 root0Tail51 seed.bound = root0Candidates50 := by
  rw [root0Candidates50Block0_step, root0Candidates50Block1_step, root0Candidates50Block2_step, root0Candidates50_end]
  rfl
theorem root0Step50 : scan edge (search edge 3) 4 [3] 3 root0Tail50 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail51 seed := by
  rw [root0Tail50, scan, if_neg (by decide), root0Candidates50_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail51
    (if 3 ≤ root0Candidates50.length then search edge 3 [3, 2213] root0Candidates50 2216 seed else seed) = _
  rw [if_neg (by decide)]
def root0Candidates51 : List Nat := [2381]
def root0Candidates51Block0 : List Nat := [2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979]
set_option maxRecDepth 4096 in
theorem root0Candidates51Block0_shape : root0Tail52 = root0Candidates51Block0 ++ root0Tail180 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates51Block0_checked : root0Candidates51Block0.filter (SortedNatTree.contains FastRows.row2297) = [2381] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates51Block0_step : candidates edge 4 3 2297 root0Tail52 seed.bound =
    [2381] ++ candidates edge 4 3 2297 root0Tail180 seed.bound := by
  rw [root0Candidates51Block0_shape, candidates_append_of_within edge 4 3 2297 root0Candidates51Block0 root0Tail180 seed.bound (by decide)]
  have row : edge 2297 = (SortedNatTree.contains FastRows.row2297) := by
    change Data.row2297.contains = (SortedNatTree.contains FastRows.row2297)
    exact FastRows.row2297_checked.symm
  rw [row, root0Candidates51Block0_checked]
def root0Candidates51Block1 : List Nat := [13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369]
set_option maxRecDepth 4096 in
theorem root0Candidates51Block1_shape : root0Tail180 = root0Candidates51Block1 ++ root0Tail308 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates51Block1_checked : root0Candidates51Block1.filter (SortedNatTree.contains FastRows.row2297) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates51Block1_step : candidates edge 4 3 2297 root0Tail180 seed.bound =
    [] ++ candidates edge 4 3 2297 root0Tail308 seed.bound := by
  rw [root0Candidates51Block1_shape, candidates_append_of_within edge 4 3 2297 root0Candidates51Block1 root0Tail308 seed.bound (by decide)]
  have row : edge 2297 = (SortedNatTree.contains FastRows.row2297) := by
    change Data.row2297.contains = (SortedNatTree.contains FastRows.row2297)
    exact FastRows.row2297_checked.symm
  rw [row, root0Candidates51Block1_checked]
def root0Candidates51Block2 : List Nat := [23371, 23473, 23549, 23567, 23581]
set_option maxRecDepth 4096 in
theorem root0Candidates51Block2_shape : root0Tail308 = root0Candidates51Block2 ++ root0Tail313 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates51Block2_checked : root0Candidates51Block2.filter (SortedNatTree.contains FastRows.row2297) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates51Block2_step : candidates edge 4 3 2297 root0Tail308 seed.bound =
    [] ++ candidates edge 4 3 2297 root0Tail313 seed.bound := by
  rw [root0Candidates51Block2_shape, candidates_append_of_within edge 4 3 2297 root0Candidates51Block2 root0Tail313 seed.bound (by decide)]
  have row : edge 2297 = (SortedNatTree.contains FastRows.row2297) := by
    change Data.row2297.contains = (SortedNatTree.contains FastRows.row2297)
    exact FastRows.row2297_checked.symm
  rw [row, root0Candidates51Block2_checked]
theorem root0Candidates51_end : candidates edge 4 3 2297 root0Tail313 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates51_checked : candidates edge 4 3 2297 root0Tail52 seed.bound = root0Candidates51 := by
  rw [root0Candidates51Block0_step, root0Candidates51Block1_step, root0Candidates51Block2_step, root0Candidates51_end]
  rfl
theorem root0Step51 : scan edge (search edge 3) 4 [3] 3 root0Tail51 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail52 seed := by
  rw [root0Tail51, scan, if_neg (by decide), root0Candidates51_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail52
    (if 3 ≤ root0Candidates51.length then search edge 3 [3, 2297] root0Candidates51 2300 seed else seed) = _
  rw [if_neg (by decide)]
def root0Candidates52 : List Nat := [4159, 6073, 10243, 18181, 20521, 22171]
def root0Candidates52Block0 : List Nat := [2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381]
set_option maxRecDepth 4096 in
theorem root0Candidates52Block0_shape : root0Tail53 = root0Candidates52Block0 ++ root0Tail181 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates52Block0_checked : root0Candidates52Block0.filter (SortedNatTree.contains FastRows.row2377) = [4159, 6073, 10243] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates52Block0_step : candidates edge 4 3 2377 root0Tail53 seed.bound =
    [4159, 6073, 10243] ++ candidates edge 4 3 2377 root0Tail181 seed.bound := by
  rw [root0Candidates52Block0_shape, candidates_append_of_within edge 4 3 2377 root0Candidates52Block0 root0Tail181 seed.bound (by decide)]
  have row : edge 2377 = (SortedNatTree.contains FastRows.row2377) := by
    change Data.row2377.contains = (SortedNatTree.contains FastRows.row2377)
    exact FastRows.row2377_checked.symm
  rw [row, root0Candidates52Block0_checked]
def root0Candidates52Block1 : List Nat := [13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371]
set_option maxRecDepth 4096 in
theorem root0Candidates52Block1_shape : root0Tail181 = root0Candidates52Block1 ++ root0Tail309 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates52Block1_checked : root0Candidates52Block1.filter (SortedNatTree.contains FastRows.row2377) = [18181, 20521, 22171] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates52Block1_step : candidates edge 4 3 2377 root0Tail181 seed.bound =
    [18181, 20521, 22171] ++ candidates edge 4 3 2377 root0Tail309 seed.bound := by
  rw [root0Candidates52Block1_shape, candidates_append_of_within edge 4 3 2377 root0Candidates52Block1 root0Tail309 seed.bound (by decide)]
  have row : edge 2377 = (SortedNatTree.contains FastRows.row2377) := by
    change Data.row2377.contains = (SortedNatTree.contains FastRows.row2377)
    exact FastRows.row2377_checked.symm
  rw [row, root0Candidates52Block1_checked]
def root0Candidates52Block2 : List Nat := [23473, 23549, 23567, 23581]
set_option maxRecDepth 4096 in
theorem root0Candidates52Block2_shape : root0Tail309 = root0Candidates52Block2 ++ root0Tail313 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates52Block2_checked : root0Candidates52Block2.filter (SortedNatTree.contains FastRows.row2377) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates52Block2_step : candidates edge 4 3 2377 root0Tail309 seed.bound =
    [] ++ candidates edge 4 3 2377 root0Tail313 seed.bound := by
  rw [root0Candidates52Block2_shape, candidates_append_of_within edge 4 3 2377 root0Candidates52Block2 root0Tail313 seed.bound (by decide)]
  have row : edge 2377 = (SortedNatTree.contains FastRows.row2377) := by
    change Data.row2377.contains = (SortedNatTree.contains FastRows.row2377)
    exact FastRows.row2377_checked.symm
  rw [row, root0Candidates52Block2_checked]
theorem root0Candidates52_end : candidates edge 4 3 2377 root0Tail313 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates52_checked : candidates edge 4 3 2377 root0Tail53 seed.bound = root0Candidates52 := by
  rw [root0Candidates52Block0_step, root0Candidates52Block1_step, root0Candidates52Block2_step, root0Candidates52_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child52 : search edge 3 [3, 2377] root0Candidates52 2380 seed = seed := by decide
theorem root0Step52 : scan edge (search edge 3) 4 [3] 3 root0Tail52 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail53 seed := by
  rw [root0Tail52, scan, if_neg (by decide), root0Candidates52_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail53
    (if 3 ≤ root0Candidates52.length then search edge 3 [3, 2377] root0Candidates52 2380 seed else seed) = _
  rw [if_pos (by decide), root0Child52]
def root0Candidates53 : List Nat := [6563, 12071, 19727]
def root0Candidates53Block0 : List Nat := [2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399]
set_option maxRecDepth 4096 in
theorem root0Candidates53Block0_shape : root0Tail54 = root0Candidates53Block0 ++ root0Tail182 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates53Block0_checked : root0Candidates53Block0.filter (SortedNatTree.contains FastRows.row2381) = [6563, 12071] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates53Block0_step : candidates edge 4 3 2381 root0Tail54 seed.bound =
    [6563, 12071] ++ candidates edge 4 3 2381 root0Tail182 seed.bound := by
  rw [root0Candidates53Block0_shape, candidates_append_of_within edge 4 3 2381 root0Candidates53Block0 root0Tail182 seed.bound (by decide)]
  have row : edge 2381 = (SortedNatTree.contains FastRows.row2381) := by
    change Data.row2381.contains = (SortedNatTree.contains FastRows.row2381)
    exact FastRows.row2381_checked.symm
  rw [row, root0Candidates53Block0_checked]
def root0Candidates53Block1 : List Nat := [13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473]
set_option maxRecDepth 4096 in
theorem root0Candidates53Block1_shape : root0Tail182 = root0Candidates53Block1 ++ root0Tail310 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates53Block1_checked : root0Candidates53Block1.filter (SortedNatTree.contains FastRows.row2381) = [19727] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates53Block1_step : candidates edge 4 3 2381 root0Tail182 seed.bound =
    [19727] ++ candidates edge 4 3 2381 root0Tail310 seed.bound := by
  rw [root0Candidates53Block1_shape, candidates_append_of_within edge 4 3 2381 root0Candidates53Block1 root0Tail310 seed.bound (by decide)]
  have row : edge 2381 = (SortedNatTree.contains FastRows.row2381) := by
    change Data.row2381.contains = (SortedNatTree.contains FastRows.row2381)
    exact FastRows.row2381_checked.symm
  rw [row, root0Candidates53Block1_checked]
def root0Candidates53Block2 : List Nat := [23549, 23567, 23581]
set_option maxRecDepth 4096 in
theorem root0Candidates53Block2_shape : root0Tail310 = root0Candidates53Block2 ++ root0Tail313 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates53Block2_checked : root0Candidates53Block2.filter (SortedNatTree.contains FastRows.row2381) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates53Block2_step : candidates edge 4 3 2381 root0Tail310 seed.bound =
    [] ++ candidates edge 4 3 2381 root0Tail313 seed.bound := by
  rw [root0Candidates53Block2_shape, candidates_append_of_within edge 4 3 2381 root0Candidates53Block2 root0Tail313 seed.bound (by decide)]
  have row : edge 2381 = (SortedNatTree.contains FastRows.row2381) := by
    change Data.row2381.contains = (SortedNatTree.contains FastRows.row2381)
    exact FastRows.row2381_checked.symm
  rw [row, root0Candidates53Block2_checked]
theorem root0Candidates53_end : candidates edge 4 3 2381 root0Tail313 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates53_checked : candidates edge 4 3 2381 root0Tail54 seed.bound = root0Candidates53 := by
  rw [root0Candidates53Block0_step, root0Candidates53Block1_step, root0Candidates53Block2_step, root0Candidates53_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child53 : search edge 3 [3, 2381] root0Candidates53 2384 seed = seed := by decide
theorem root0Step53 : scan edge (search edge 3) 4 [3] 3 root0Tail53 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail54 seed := by
  rw [root0Tail53, scan, if_neg (by decide), root0Candidates53_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail54
    (if 3 ≤ root0Candidates53.length then search edge 3 [3, 2381] root0Candidates53 2384 seed else seed) = _
  rw [if_pos (by decide), root0Child53]
def root0Candidates54 : List Nat := [4253, 4919, 5171, 12899, 21821]
def root0Candidates54Block0 : List Nat := [2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553]
set_option maxRecDepth 4096 in
theorem root0Candidates54Block0_shape : root0Tail55 = root0Candidates54Block0 ++ root0Tail183 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates54Block0_checked : root0Candidates54Block0.filter (SortedNatTree.contains FastRows.row2411) = [4253, 4919, 5171, 12899] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates54Block0_step : candidates edge 4 3 2411 root0Tail55 seed.bound =
    [4253, 4919, 5171, 12899] ++ candidates edge 4 3 2411 root0Tail183 seed.bound := by
  rw [root0Candidates54Block0_shape, candidates_append_of_within edge 4 3 2411 root0Candidates54Block0 root0Tail183 seed.bound (by decide)]
  have row : edge 2411 = (SortedNatTree.contains FastRows.row2411) := by
    change Data.row2411.contains = (SortedNatTree.contains FastRows.row2411)
    exact FastRows.row2411_checked.symm
  rw [row, root0Candidates54Block0_checked]
def root0Candidates54Block1 : List Nat := [13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549]
set_option maxRecDepth 4096 in
theorem root0Candidates54Block1_shape : root0Tail183 = root0Candidates54Block1 ++ root0Tail311 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates54Block1_checked : root0Candidates54Block1.filter (SortedNatTree.contains FastRows.row2411) = [21821] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates54Block1_step : candidates edge 4 3 2411 root0Tail183 seed.bound =
    [21821] ++ candidates edge 4 3 2411 root0Tail311 seed.bound := by
  rw [root0Candidates54Block1_shape, candidates_append_of_within edge 4 3 2411 root0Candidates54Block1 root0Tail311 seed.bound (by decide)]
  have row : edge 2411 = (SortedNatTree.contains FastRows.row2411) := by
    change Data.row2411.contains = (SortedNatTree.contains FastRows.row2411)
    exact FastRows.row2411_checked.symm
  rw [row, root0Candidates54Block1_checked]
def root0Candidates54Block2 : List Nat := [23567, 23581]
set_option maxRecDepth 4096 in
theorem root0Candidates54Block2_shape : root0Tail311 = root0Candidates54Block2 ++ root0Tail313 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates54Block2_checked : root0Candidates54Block2.filter (SortedNatTree.contains FastRows.row2411) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates54Block2_step : candidates edge 4 3 2411 root0Tail311 seed.bound =
    [] ++ candidates edge 4 3 2411 root0Tail313 seed.bound := by
  rw [root0Candidates54Block2_shape, candidates_append_of_within edge 4 3 2411 root0Candidates54Block2 root0Tail313 seed.bound (by decide)]
  have row : edge 2411 = (SortedNatTree.contains FastRows.row2411) := by
    change Data.row2411.contains = (SortedNatTree.contains FastRows.row2411)
    exact FastRows.row2411_checked.symm
  rw [row, root0Candidates54Block2_checked]
theorem root0Candidates54_end : candidates edge 4 3 2411 root0Tail313 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates54_checked : candidates edge 4 3 2411 root0Tail55 seed.bound = root0Candidates54 := by
  rw [root0Candidates54Block0_step, root0Candidates54Block1_step, root0Candidates54Block2_step, root0Candidates54_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child54 : search edge 3 [3, 2411] root0Candidates54 2414 seed = seed := by decide
theorem root0Step54 : scan edge (search edge 3) 4 [3] 3 root0Tail54 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail55 seed := by
  rw [root0Tail54, scan, if_neg (by decide), root0Candidates54_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail55
    (if 3 ≤ root0Candidates54.length then search edge 3 [3, 2411] root0Candidates54 2414 seed else seed) = _
  rw [if_pos (by decide), root0Child54]
def root0Candidates55 : List Nat := [5381, 7691, 10169, 15083, 22709]
def root0Candidates55Block0 : List Nat := [2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613]
set_option maxRecDepth 4096 in
theorem root0Candidates55Block0_shape : root0Tail56 = root0Candidates55Block0 ++ root0Tail184 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates55Block0_checked : root0Candidates55Block0.filter (SortedNatTree.contains FastRows.row2441) = [5381, 7691, 10169] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates55Block0_step : candidates edge 4 3 2441 root0Tail56 seed.bound =
    [5381, 7691, 10169] ++ candidates edge 4 3 2441 root0Tail184 seed.bound := by
  rw [root0Candidates55Block0_shape, candidates_append_of_within edge 4 3 2441 root0Candidates55Block0 root0Tail184 seed.bound (by decide)]
  have row : edge 2441 = (SortedNatTree.contains FastRows.row2441) := by
    change Data.row2441.contains = (SortedNatTree.contains FastRows.row2441)
    exact FastRows.row2441_checked.symm
  rw [row, root0Candidates55Block0_checked]
def root0Candidates55Block1 : List Nat := [13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473, 23549, 23567]
set_option maxRecDepth 4096 in
theorem root0Candidates55Block1_shape : root0Tail184 = root0Candidates55Block1 ++ root0Tail312 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates55Block1_checked : root0Candidates55Block1.filter (SortedNatTree.contains FastRows.row2441) = [15083, 22709] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates55Block1_step : candidates edge 4 3 2441 root0Tail184 seed.bound =
    [15083, 22709] ++ candidates edge 4 3 2441 root0Tail312 seed.bound := by
  rw [root0Candidates55Block1_shape, candidates_append_of_within edge 4 3 2441 root0Candidates55Block1 root0Tail312 seed.bound (by decide)]
  have row : edge 2441 = (SortedNatTree.contains FastRows.row2441) := by
    change Data.row2441.contains = (SortedNatTree.contains FastRows.row2441)
    exact FastRows.row2441_checked.symm
  rw [row, root0Candidates55Block1_checked]
def root0Candidates55Block2 : List Nat := [23581]
set_option maxRecDepth 4096 in
theorem root0Candidates55Block2_shape : root0Tail312 = root0Candidates55Block2 ++ root0Tail313 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates55Block2_checked : root0Candidates55Block2.filter (SortedNatTree.contains FastRows.row2441) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates55Block2_step : candidates edge 4 3 2441 root0Tail312 seed.bound =
    [] ++ candidates edge 4 3 2441 root0Tail313 seed.bound := by
  rw [root0Candidates55Block2_shape, candidates_append_of_within edge 4 3 2441 root0Candidates55Block2 root0Tail313 seed.bound (by decide)]
  have row : edge 2441 = (SortedNatTree.contains FastRows.row2441) := by
    change Data.row2441.contains = (SortedNatTree.contains FastRows.row2441)
    exact FastRows.row2441_checked.symm
  rw [row, root0Candidates55Block2_checked]
theorem root0Candidates55_end : candidates edge 4 3 2441 root0Tail313 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates55_checked : candidates edge 4 3 2441 root0Tail56 seed.bound = root0Candidates55 := by
  rw [root0Candidates55Block0_step, root0Candidates55Block1_step, root0Candidates55Block2_step, root0Candidates55_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child55 : search edge 3 [3, 2441] root0Candidates55 2444 seed = seed := by decide
theorem root0Step55 : scan edge (search edge 3) 4 [3] 3 root0Tail55 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail56 seed := by
  rw [root0Tail55, scan, if_neg (by decide), root0Candidates55_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail56
    (if 3 ≤ root0Candidates55.length then search edge 3 [3, 2441] root0Candidates55 2444 seed else seed) = _
  rw [if_pos (by decide), root0Child55]
def root0Candidates56 : List Nat := [5281, 5869, 7591, 11827, 17923, 18181, 23251]
def root0Candidates56Block0 : List Nat := [2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619]
set_option maxRecDepth 4096 in
theorem root0Candidates56Block0_shape : root0Tail57 = root0Candidates56Block0 ++ root0Tail185 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates56Block0_checked : root0Candidates56Block0.filter (SortedNatTree.contains FastRows.row2503) = [5281, 5869, 7591, 11827] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates56Block0_step : candidates edge 4 3 2503 root0Tail57 seed.bound =
    [5281, 5869, 7591, 11827] ++ candidates edge 4 3 2503 root0Tail185 seed.bound := by
  rw [root0Candidates56Block0_shape, candidates_append_of_within edge 4 3 2503 root0Candidates56Block0 root0Tail185 seed.bound (by decide)]
  have row : edge 2503 = (SortedNatTree.contains FastRows.row2503) := by
    change Data.row2503.contains = (SortedNatTree.contains FastRows.row2503)
    exact FastRows.row2503_checked.symm
  rw [row, root0Candidates56Block0_checked]
def root0Candidates56Block1 : List Nat := [13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371, 23473]
set_option maxRecDepth 4096 in
theorem root0Candidates56Block1_shape : root0Tail185 = root0Candidates56Block1 ++ root0Tail310 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates56Block1_checked : root0Candidates56Block1.filter (SortedNatTree.contains FastRows.row2503) = [17923, 18181, 23251] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates56Block1_step : candidates edge 4 3 2503 root0Tail185 seed.bound =
    [17923, 18181, 23251] ++ candidates edge 4 3 2503 root0Tail310 seed.bound := by
  rw [root0Candidates56Block1_shape, candidates_append_of_within edge 4 3 2503 root0Candidates56Block1 root0Tail310 seed.bound (by decide)]
  have row : edge 2503 = (SortedNatTree.contains FastRows.row2503) := by
    change Data.row2503.contains = (SortedNatTree.contains FastRows.row2503)
    exact FastRows.row2503_checked.symm
  rw [row, root0Candidates56Block1_checked]
theorem root0Candidates56_end : candidates edge 4 3 2503 root0Tail310 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates56_checked : candidates edge 4 3 2503 root0Tail57 seed.bound = root0Candidates56 := by
  rw [root0Candidates56Block0_step, root0Candidates56Block1_step, root0Candidates56_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child56 : search edge 3 [3, 2503] root0Candidates56 2506 seed = seed := by decide
theorem root0Step56 : scan edge (search edge 3) 4 [3] 3 root0Tail56 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail57 seed := by
  rw [root0Tail56, scan, if_neg (by decide), root0Candidates56_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail57
    (if 3 ≤ root0Candidates56.length then search edge 3 [3, 2503] root0Candidates56 2506 seed else seed) = _
  rw [if_pos (by decide), root0Child56]
def root0Candidates57 : List Nat := [11177, 11393, 19919, 20759, 22271, 22859, 23027]
def root0Candidates57Block0 : List Nat := [2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669]
set_option maxRecDepth 4096 in
theorem root0Candidates57Block0_shape : root0Tail58 = root0Candidates57Block0 ++ root0Tail186 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates57Block0_checked : root0Candidates57Block0.filter (SortedNatTree.contains FastRows.row2579) = [11177, 11393] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates57Block0_step : candidates edge 4 3 2579 root0Tail58 seed.bound =
    [11177, 11393] ++ candidates edge 4 3 2579 root0Tail186 seed.bound := by
  rw [root0Candidates57Block0_shape, candidates_append_of_within edge 4 3 2579 root0Candidates57Block0 root0Tail186 seed.bound (by decide)]
  have row : edge 2579 = (SortedNatTree.contains FastRows.row2579) := by
    change Data.row2579.contains = (SortedNatTree.contains FastRows.row2579)
    exact FastRows.row2579_checked.symm
  rw [row, root0Candidates57Block0_checked]
def root0Candidates57Block1 : List Nat := [13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251, 23369, 23371]
set_option maxRecDepth 4096 in
theorem root0Candidates57Block1_shape : root0Tail186 = root0Candidates57Block1 ++ root0Tail309 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates57Block1_checked : root0Candidates57Block1.filter (SortedNatTree.contains FastRows.row2579) = [19919, 20759, 22271, 22859, 23027] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates57Block1_step : candidates edge 4 3 2579 root0Tail186 seed.bound =
    [19919, 20759, 22271, 22859, 23027] ++ candidates edge 4 3 2579 root0Tail309 seed.bound := by
  rw [root0Candidates57Block1_shape, candidates_append_of_within edge 4 3 2579 root0Candidates57Block1 root0Tail309 seed.bound (by decide)]
  have row : edge 2579 = (SortedNatTree.contains FastRows.row2579) := by
    change Data.row2579.contains = (SortedNatTree.contains FastRows.row2579)
    exact FastRows.row2579_checked.symm
  rw [row, root0Candidates57Block1_checked]
theorem root0Candidates57_end : candidates edge 4 3 2579 root0Tail309 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates57_checked : candidates edge 4 3 2579 root0Tail58 seed.bound = root0Candidates57 := by
  rw [root0Candidates57Block0_step, root0Candidates57Block1_step, root0Candidates57_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child57 : search edge 3 [3, 2579] root0Candidates57 2582 seed = seed := by decide
theorem root0Step57 : scan edge (search edge 3) 4 [3] 3 root0Tail57 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail58 seed := by
  rw [root0Tail57, scan, if_neg (by decide), root0Candidates57_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail58
    (if 3 ≤ root0Candidates57.length then search edge 3 [3, 2579] root0Candidates57 2582 seed else seed) = _
  rw [if_pos (by decide), root0Child57]
def root0Candidates58 : List Nat := [10333, 14107, 19819]
def root0Candidates58Block0 : List Nat := [2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763]
set_option maxRecDepth 4096 in
theorem root0Candidates58Block0_shape : root0Tail59 = root0Candidates58Block0 ++ root0Tail187 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates58Block0_checked : root0Candidates58Block0.filter (SortedNatTree.contains FastRows.row2707) = [10333] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates58Block0_step : candidates edge 4 3 2707 root0Tail59 seed.bound =
    [10333] ++ candidates edge 4 3 2707 root0Tail187 seed.bound := by
  rw [root0Candidates58Block0_shape, candidates_append_of_within edge 4 3 2707 root0Candidates58Block0 root0Tail187 seed.bound (by decide)]
  have row : edge 2707 = (SortedNatTree.contains FastRows.row2707) := by
    change Data.row2707.contains = (SortedNatTree.contains FastRows.row2707)
    exact FastRows.row2707_checked.symm
  rw [row, root0Candidates58Block0_checked]
def root0Candidates58Block1 : List Nat := [13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201, 23251]
set_option maxRecDepth 4096 in
theorem root0Candidates58Block1_shape : root0Tail187 = root0Candidates58Block1 ++ root0Tail307 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates58Block1_checked : root0Candidates58Block1.filter (SortedNatTree.contains FastRows.row2707) = [14107, 19819] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates58Block1_step : candidates edge 4 3 2707 root0Tail187 seed.bound =
    [14107, 19819] ++ candidates edge 4 3 2707 root0Tail307 seed.bound := by
  rw [root0Candidates58Block1_shape, candidates_append_of_within edge 4 3 2707 root0Candidates58Block1 root0Tail307 seed.bound (by decide)]
  have row : edge 2707 = (SortedNatTree.contains FastRows.row2707) := by
    change Data.row2707.contains = (SortedNatTree.contains FastRows.row2707)
    exact FastRows.row2707_checked.symm
  rw [row, root0Candidates58Block1_checked]
theorem root0Candidates58_end : candidates edge 4 3 2707 root0Tail307 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates58_checked : candidates edge 4 3 2707 root0Tail59 seed.bound = root0Candidates58 := by
  rw [root0Candidates58Block0_step, root0Candidates58Block1_step, root0Candidates58_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child58 : search edge 3 [3, 2707] root0Candidates58 2710 seed = seed := by decide
theorem root0Step58 : scan edge (search edge 3) 4 [3] 3 root0Tail58 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail59 seed := by
  rw [root0Tail58, scan, if_neg (by decide), root0Candidates58_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail59
    (if 3 ≤ root0Candidates58.length then search edge 3 [3, 2707] root0Candidates58 2710 seed else seed) = _
  rw [if_pos (by decide), root0Child58]
def root0Candidates59 : List Nat := [3461, 5051, 6563, 7691, 14543, 21017]
def root0Candidates59Block0 : List Nat := [2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879]
set_option maxRecDepth 4096 in
theorem root0Candidates59Block0_shape : root0Tail60 = root0Candidates59Block0 ++ root0Tail188 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates59Block0_checked : root0Candidates59Block0.filter (SortedNatTree.contains FastRows.row2789) = [3461, 5051, 6563, 7691] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates59Block0_step : candidates edge 4 3 2789 root0Tail60 seed.bound =
    [3461, 5051, 6563, 7691] ++ candidates edge 4 3 2789 root0Tail188 seed.bound := by
  rw [root0Candidates59Block0_shape, candidates_append_of_within edge 4 3 2789 root0Candidates59Block0 root0Tail188 seed.bound (by decide)]
  have row : edge 2789 = (SortedNatTree.contains FastRows.row2789) := by
    change Data.row2789.contains = (SortedNatTree.contains FastRows.row2789)
    exact FastRows.row2789_checked.symm
  rw [row, root0Candidates59Block0_checked]
def root0Candidates59Block1 : List Nat := [13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087, 23201]
set_option maxRecDepth 4096 in
theorem root0Candidates59Block1_shape : root0Tail188 = root0Candidates59Block1 ++ root0Tail306 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates59Block1_checked : root0Candidates59Block1.filter (SortedNatTree.contains FastRows.row2789) = [14543, 21017] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates59Block1_step : candidates edge 4 3 2789 root0Tail188 seed.bound =
    [14543, 21017] ++ candidates edge 4 3 2789 root0Tail306 seed.bound := by
  rw [root0Candidates59Block1_shape, candidates_append_of_within edge 4 3 2789 root0Candidates59Block1 root0Tail306 seed.bound (by decide)]
  have row : edge 2789 = (SortedNatTree.contains FastRows.row2789) := by
    change Data.row2789.contains = (SortedNatTree.contains FastRows.row2789)
    exact FastRows.row2789_checked.symm
  rw [row, root0Candidates59Block1_checked]
theorem root0Candidates59_end : candidates edge 4 3 2789 root0Tail306 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates59_checked : candidates edge 4 3 2789 root0Tail60 seed.bound = root0Candidates59 := by
  rw [root0Candidates59Block0_step, root0Candidates59Block1_step, root0Candidates59_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child59 : search edge 3 [3, 2789] root0Candidates59 2792 seed = seed := by decide
theorem root0Step59 : scan edge (search edge 3) 4 [3] 3 root0Tail59 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail60 seed := by
  rw [root0Tail59, scan, if_neg (by decide), root0Candidates59_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail60
    (if 3 ≤ root0Candidates59.length then search edge 3 [3, 2789] root0Candidates59 2792 seed else seed) = _
  rw [if_pos (by decide), root0Child59]
def root0Candidates60 : List Nat := [4253, 6899, 9857]
def root0Candidates60Block0 : List Nat := [2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931]
set_option maxRecDepth 4096 in
theorem root0Candidates60Block0_shape : root0Tail61 = root0Candidates60Block0 ++ root0Tail189 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates60Block0_checked : root0Candidates60Block0.filter (SortedNatTree.contains FastRows.row2843) = [4253, 6899, 9857] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates60Block0_step : candidates edge 4 3 2843 root0Tail61 seed.bound =
    [4253, 6899, 9857] ++ candidates edge 4 3 2843 root0Tail189 seed.bound := by
  rw [root0Candidates60Block0_shape, candidates_append_of_within edge 4 3 2843 root0Candidates60Block0 root0Tail189 seed.bound (by decide)]
  have row : edge 2843 = (SortedNatTree.contains FastRows.row2843) := by
    change Data.row2843.contains = (SortedNatTree.contains FastRows.row2843)
    exact FastRows.row2843_checked.symm
  rw [row, root0Candidates60Block0_checked]
def root0Candidates60Block1 : List Nat := [13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087]
set_option maxRecDepth 4096 in
theorem root0Candidates60Block1_shape : root0Tail189 = root0Candidates60Block1 ++ root0Tail305 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates60Block1_checked : root0Candidates60Block1.filter (SortedNatTree.contains FastRows.row2843) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates60Block1_step : candidates edge 4 3 2843 root0Tail189 seed.bound =
    [] ++ candidates edge 4 3 2843 root0Tail305 seed.bound := by
  rw [root0Candidates60Block1_shape, candidates_append_of_within edge 4 3 2843 root0Candidates60Block1 root0Tail305 seed.bound (by decide)]
  have row : edge 2843 = (SortedNatTree.contains FastRows.row2843) := by
    change Data.row2843.contains = (SortedNatTree.contains FastRows.row2843)
    exact FastRows.row2843_checked.symm
  rw [row, root0Candidates60Block1_checked]
theorem root0Candidates60_end : candidates edge 4 3 2843 root0Tail305 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates60_checked : candidates edge 4 3 2843 root0Tail61 seed.bound = root0Candidates60 := by
  rw [root0Candidates60Block0_step, root0Candidates60Block1_step, root0Candidates60_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child60 : search edge 3 [3, 2843] root0Candidates60 2846 seed = seed := by decide
theorem root0Step60 : scan edge (search edge 3) 4 [3] 3 root0Tail60 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail61 seed := by
  rw [root0Tail60, scan, if_neg (by decide), root0Candidates60_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail61
    (if 3 ≤ root0Candidates60.length then search edge 3 [3, 2843] root0Candidates60 2846 seed else seed) = _
  rw [if_pos (by decide), root0Child60]
def root0Candidates61 : List Nat := [16759]
def root0Candidates61Block0 : List Nat := [2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933]
set_option maxRecDepth 4096 in
theorem root0Candidates61Block0_shape : root0Tail62 = root0Candidates61Block0 ++ root0Tail190 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates61Block0_checked : root0Candidates61Block0.filter (SortedNatTree.contains FastRows.row2917) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates61Block0_step : candidates edge 4 3 2917 root0Tail62 seed.bound =
    [] ++ candidates edge 4 3 2917 root0Tail190 seed.bound := by
  rw [root0Candidates61Block0_shape, candidates_append_of_within edge 4 3 2917 root0Candidates61Block0 root0Tail190 seed.bound (by decide)]
  have row : edge 2917 = (SortedNatTree.contains FastRows.row2917) := by
    change Data.row2917.contains = (SortedNatTree.contains FastRows.row2917)
    exact FastRows.row2917_checked.symm
  rw [row, root0Candidates61Block0_checked]
def root0Candidates61Block1 : List Nat := [14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027, 23087]
set_option maxRecDepth 4096 in
theorem root0Candidates61Block1_shape : root0Tail190 = root0Candidates61Block1 ++ root0Tail305 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates61Block1_checked : root0Candidates61Block1.filter (SortedNatTree.contains FastRows.row2917) = [16759] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates61Block1_step : candidates edge 4 3 2917 root0Tail190 seed.bound =
    [16759] ++ candidates edge 4 3 2917 root0Tail305 seed.bound := by
  rw [root0Candidates61Block1_shape, candidates_append_of_within edge 4 3 2917 root0Candidates61Block1 root0Tail305 seed.bound (by decide)]
  have row : edge 2917 = (SortedNatTree.contains FastRows.row2917) := by
    change Data.row2917.contains = (SortedNatTree.contains FastRows.row2917)
    exact FastRows.row2917_checked.symm
  rw [row, root0Candidates61Block1_checked]
theorem root0Candidates61_end : candidates edge 4 3 2917 root0Tail305 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates61_checked : candidates edge 4 3 2917 root0Tail62 seed.bound = root0Candidates61 := by
  rw [root0Candidates61Block0_step, root0Candidates61Block1_step, root0Candidates61_end]
  rfl
theorem root0Step61 : scan edge (search edge 3) 4 [3] 3 root0Tail61 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail62 seed := by
  rw [root0Tail61, scan, if_neg (by decide), root0Candidates61_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail62
    (if 3 ≤ root0Candidates61.length then search edge 3 [3, 2917] root0Candidates61 2920 seed else seed) = _
  rw [if_neg (by decide)]
def root0Candidates62 : List Nat := [19031]
def root0Candidates62Block0 : List Nat := [3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107]
set_option maxRecDepth 4096 in
theorem root0Candidates62Block0_shape : root0Tail63 = root0Candidates62Block0 ++ root0Tail191 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates62Block0_checked : root0Candidates62Block0.filter (SortedNatTree.contains FastRows.row2957) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates62Block0_step : candidates edge 4 3 2957 root0Tail63 seed.bound =
    [] ++ candidates edge 4 3 2957 root0Tail191 seed.bound := by
  rw [root0Candidates62Block0_shape, candidates_append_of_within edge 4 3 2957 root0Candidates62Block0 root0Tail191 seed.bound (by decide)]
  have row : edge 2957 = (SortedNatTree.contains FastRows.row2957) := by
    change Data.row2957.contains = (SortedNatTree.contains FastRows.row2957)
    exact FastRows.row2957_checked.symm
  rw [row, root0Candidates62Block0_checked]
def root0Candidates62Block1 : List Nat := [14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921, 23027]
set_option maxRecDepth 4096 in
theorem root0Candidates62Block1_shape : root0Tail191 = root0Candidates62Block1 ++ root0Tail304 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates62Block1_checked : root0Candidates62Block1.filter (SortedNatTree.contains FastRows.row2957) = [19031] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates62Block1_step : candidates edge 4 3 2957 root0Tail191 seed.bound =
    [19031] ++ candidates edge 4 3 2957 root0Tail304 seed.bound := by
  rw [root0Candidates62Block1_shape, candidates_append_of_within edge 4 3 2957 root0Candidates62Block1 root0Tail304 seed.bound (by decide)]
  have row : edge 2957 = (SortedNatTree.contains FastRows.row2957) := by
    change Data.row2957.contains = (SortedNatTree.contains FastRows.row2957)
    exact FastRows.row2957_checked.symm
  rw [row, root0Candidates62Block1_checked]
theorem root0Candidates62_end : candidates edge 4 3 2957 root0Tail304 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates62_checked : candidates edge 4 3 2957 root0Tail63 seed.bound = root0Candidates62 := by
  rw [root0Candidates62Block0_step, root0Candidates62Block1_step, root0Candidates62_end]
  rfl
theorem root0Step62 : scan edge (search edge 3) 4 [3] 3 root0Tail62 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail63 seed := by
  rw [root0Tail62, scan, if_neg (by decide), root0Candidates62_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail63
    (if 3 ≤ root0Candidates62.length then search edge 3 [3, 2957] root0Candidates62 2960 seed else seed) = _
  rw [if_neg (by decide)]
def root0Candidates63 : List Nat := [10357, 19819, 20899, 21187]
def root0Candidates63Block0 : List Nat := [3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243]
set_option maxRecDepth 4096 in
theorem root0Candidates63Block0_shape : root0Tail64 = root0Candidates63Block0 ++ root0Tail192 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates63Block0_checked : root0Candidates63Block0.filter (SortedNatTree.contains FastRows.row3049) = [10357] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates63Block0_step : candidates edge 4 3 3049 root0Tail64 seed.bound =
    [10357] ++ candidates edge 4 3 3049 root0Tail192 seed.bound := by
  rw [root0Candidates63Block0_shape, candidates_append_of_within edge 4 3 3049 root0Candidates63Block0 root0Tail192 seed.bound (by decide)]
  have row : edge 3049 = (SortedNatTree.contains FastRows.row3049) := by
    change Data.row3049.contains = (SortedNatTree.contains FastRows.row3049)
    exact FastRows.row3049_checked.symm
  rw [row, root0Candidates63Block0_checked]
def root0Candidates63Block1 : List Nat := [14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877, 22921]
set_option maxRecDepth 4096 in
theorem root0Candidates63Block1_shape : root0Tail192 = root0Candidates63Block1 ++ root0Tail303 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates63Block1_checked : root0Candidates63Block1.filter (SortedNatTree.contains FastRows.row3049) = [19819, 20899, 21187] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates63Block1_step : candidates edge 4 3 3049 root0Tail192 seed.bound =
    [19819, 20899, 21187] ++ candidates edge 4 3 3049 root0Tail303 seed.bound := by
  rw [root0Candidates63Block1_shape, candidates_append_of_within edge 4 3 3049 root0Candidates63Block1 root0Tail303 seed.bound (by decide)]
  have row : edge 3049 = (SortedNatTree.contains FastRows.row3049) := by
    change Data.row3049.contains = (SortedNatTree.contains FastRows.row3049)
    exact FastRows.row3049_checked.symm
  rw [row, root0Candidates63Block1_checked]
theorem root0Candidates63_end : candidates edge 4 3 3049 root0Tail303 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates63_checked : candidates edge 4 3 3049 root0Tail64 seed.bound = root0Candidates63 := by
  rw [root0Candidates63Block0_step, root0Candidates63Block1_step, root0Candidates63_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child63 : search edge 3 [3, 3049] root0Candidates63 3052 seed = seed := by decide
theorem root0Step63 : scan edge (search edge 3) 4 [3] 3 root0Tail63 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail64 seed := by
  rw [root0Tail63, scan, if_neg (by decide), root0Candidates63_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail64
    (if 3 ≤ root0Candidates63.length then search edge 3 [3, 3049] root0Candidates63 3052 seed else seed) = _
  rw [if_pos (by decide), root0Child63]
end ProjectEulerSolutions.PrimePairLarge.Search
