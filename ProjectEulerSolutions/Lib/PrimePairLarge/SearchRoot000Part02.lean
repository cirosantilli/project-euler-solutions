import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot000Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root0Candidates64 : List Nat := [9887, 14879, 19919, 21569]
def root0Candidates64Block0 : List Nat := [3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401]
set_option maxRecDepth 4096 in
theorem root0Candidates64Block0_shape : root0Tail65 = root0Candidates64Block0 ++ root0Tail193 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates64Block0_checked : root0Candidates64Block0.filter (SortedNatTree.contains FastRows.row3119) = [9887] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates64Block0_step : candidates edge 4 3 3119 root0Tail65 seed.bound =
    [9887] ++ candidates edge 4 3 3119 root0Tail193 seed.bound := by
  rw [root0Candidates64Block0_shape, candidates_append_of_within edge 4 3 3119 root0Candidates64Block0 root0Tail193 seed.bound (by decide)]
  have row : edge 3119 = (SortedNatTree.contains FastRows.row3119) := by
    change Data.row3119.contains = (SortedNatTree.contains FastRows.row3119)
    exact FastRows.row3119_checked.symm
  rw [row, root0Candidates64Block0_checked]
def root0Candidates64Block1 : List Nat := [14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709, 22769, 22859, 22871, 22877]
set_option maxRecDepth 4096 in
theorem root0Candidates64Block1_shape : root0Tail193 = root0Candidates64Block1 ++ root0Tail302 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates64Block1_checked : root0Candidates64Block1.filter (SortedNatTree.contains FastRows.row3119) = [14879, 19919, 21569] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates64Block1_step : candidates edge 4 3 3119 root0Tail193 seed.bound =
    [14879, 19919, 21569] ++ candidates edge 4 3 3119 root0Tail302 seed.bound := by
  rw [root0Candidates64Block1_shape, candidates_append_of_within edge 4 3 3119 root0Candidates64Block1 root0Tail302 seed.bound (by decide)]
  have row : edge 3119 = (SortedNatTree.contains FastRows.row3119) := by
    change Data.row3119.contains = (SortedNatTree.contains FastRows.row3119)
    exact FastRows.row3119_checked.symm
  rw [row, root0Candidates64Block1_checked]
theorem root0Candidates64_end : candidates edge 4 3 3119 root0Tail302 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates64_checked : candidates edge 4 3 3119 root0Tail65 seed.bound = root0Candidates64 := by
  rw [root0Candidates64Block0_step, root0Candidates64Block1_step, root0Candidates64_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child64 : search edge 3 [3, 3119] root0Candidates64 3122 seed = seed := by decide
theorem root0Step64 : scan edge (search edge 3) 4 [3] 3 root0Tail64 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail65 seed := by
  rw [root0Tail64, scan, if_neg (by decide), root0Candidates64_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail65
    (if 3 ≤ root0Candidates64.length then search edge 3 [3, 3119] root0Candidates64 3122 seed else seed) = _
  rw [if_pos (by decide), root0Child64]
def root0Candidates65 : List Nat := [13879, 20521, 21193]
def root0Candidates65Block0 : List Nat := [3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407]
set_option maxRecDepth 4096 in
theorem root0Candidates65Block0_shape : root0Tail66 = root0Candidates65Block0 ++ root0Tail194 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates65Block0_checked : root0Candidates65Block0.filter (SortedNatTree.contains FastRows.row3301) = [13879] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates65Block0_step : candidates edge 4 3 3301 root0Tail66 seed.bound =
    [13879] ++ candidates edge 4 3 3301 root0Tail194 seed.bound := by
  rw [root0Candidates65Block0_shape, candidates_append_of_within edge 4 3 3301 root0Candidates65Block0 root0Tail194 seed.bound (by decide)]
  have row : edge 3301 = (SortedNatTree.contains FastRows.row3301) := by
    change Data.row3301.contains = (SortedNatTree.contains FastRows.row3301)
    exact FastRows.row3301_checked.symm
  rw [row, root0Candidates65Block0_checked]
def root0Candidates65Block1 : List Nat := [14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549, 22573, 22613, 22709]
set_option maxRecDepth 4096 in
theorem root0Candidates65Block1_shape : root0Tail194 = root0Candidates65Block1 ++ root0Tail298 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates65Block1_checked : root0Candidates65Block1.filter (SortedNatTree.contains FastRows.row3301) = [20521, 21193] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates65Block1_step : candidates edge 4 3 3301 root0Tail194 seed.bound =
    [20521, 21193] ++ candidates edge 4 3 3301 root0Tail298 seed.bound := by
  rw [root0Candidates65Block1_shape, candidates_append_of_within edge 4 3 3301 root0Candidates65Block1 root0Tail298 seed.bound (by decide)]
  have row : edge 3301 = (SortedNatTree.contains FastRows.row3301) := by
    change Data.row3301.contains = (SortedNatTree.contains FastRows.row3301)
    exact FastRows.row3301_checked.symm
  rw [row, root0Candidates65Block1_checked]
theorem root0Candidates65_end : candidates edge 4 3 3301 root0Tail298 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates65_checked : candidates edge 4 3 3301 root0Tail66 seed.bound = root0Candidates65 := by
  rw [root0Candidates65Block0_step, root0Candidates65Block1_step, root0Candidates65_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child65 : search edge 3 [3, 3301] root0Candidates65 3304 seed = seed := by decide
theorem root0Step65 : scan edge (search edge 3) 4 [3] 3 root0Tail65 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail66 seed := by
  rw [root0Tail65, scan, if_neg (by decide), root0Candidates65_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail66
    (if 3 ≤ root0Candidates65.length then search edge 3 [3, 3301] root0Candidates65 3304 seed else seed) = _
  rw [if_pos (by decide), root0Child65]
def root0Candidates66 : List Nat := [4583, 15083, 17789, 17903, 19919]
def root0Candidates66Block0 : List Nat := [3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543]
set_option maxRecDepth 4096 in
theorem root0Candidates66Block0_shape : root0Tail67 = root0Candidates66Block0 ++ root0Tail195 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates66Block0_checked : root0Candidates66Block0.filter (SortedNatTree.contains FastRows.row3461) = [4583] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates66Block0_step : candidates edge 4 3 3461 root0Tail67 seed.bound =
    [4583] ++ candidates edge 4 3 3461 root0Tail195 seed.bound := by
  rw [root0Candidates66Block0_shape, candidates_append_of_within edge 4 3 3461 root0Candidates66Block0 root0Tail195 seed.bound (by decide)]
  have row : edge 3461 = (SortedNatTree.contains FastRows.row3461) := by
    change Data.row3461.contains = (SortedNatTree.contains FastRows.row3461)
    exact FastRows.row3461_checked.symm
  rw [row, root0Candidates66Block0_checked]
def root0Candidates66Block1 : List Nat := [14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549]
set_option maxRecDepth 4096 in
theorem root0Candidates66Block1_shape : root0Tail195 = root0Candidates66Block1 ++ root0Tail295 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates66Block1_checked : root0Candidates66Block1.filter (SortedNatTree.contains FastRows.row3461) = [15083, 17789, 17903, 19919] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates66Block1_step : candidates edge 4 3 3461 root0Tail195 seed.bound =
    [15083, 17789, 17903, 19919] ++ candidates edge 4 3 3461 root0Tail295 seed.bound := by
  rw [root0Candidates66Block1_shape, candidates_append_of_within edge 4 3 3461 root0Candidates66Block1 root0Tail295 seed.bound (by decide)]
  have row : edge 3461 = (SortedNatTree.contains FastRows.row3461) := by
    change Data.row3461.contains = (SortedNatTree.contains FastRows.row3461)
    exact FastRows.row3461_checked.symm
  rw [row, root0Candidates66Block1_checked]
theorem root0Candidates66_end : candidates edge 4 3 3461 root0Tail295 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates66_checked : candidates edge 4 3 3461 root0Tail67 seed.bound = root0Candidates66 := by
  rw [root0Candidates66Block0_step, root0Candidates66Block1_step, root0Candidates66_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child66 : search edge 3 [3, 3461] root0Candidates66 3464 seed = seed := by decide
theorem root0Step66 : scan edge (search edge 3) 4 [3] 3 root0Tail66 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail67 seed := by
  rw [root0Tail66, scan, if_neg (by decide), root0Candidates66_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail67
    (if 3 ≤ root0Candidates66.length then search edge 3 [3, 3461] root0Candidates66 3464 seed else seed) = _
  rw [if_pos (by decide), root0Child66]
def root0Candidates67 : List Nat := [5839, 14107]
def root0Candidates67Block0 : List Nat := [3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627]
set_option maxRecDepth 4096 in
theorem root0Candidates67Block0_shape : root0Tail68 = root0Candidates67Block0 ++ root0Tail196 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates67Block0_checked : root0Candidates67Block0.filter (SortedNatTree.contains FastRows.row3469) = [5839, 14107] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates67Block0_step : candidates edge 4 3 3469 root0Tail68 seed.bound =
    [5839, 14107] ++ candidates edge 4 3 3469 root0Tail196 seed.bound := by
  rw [root0Candidates67Block0_shape, candidates_append_of_within edge 4 3 3469 root0Candidates67Block0 root0Tail196 seed.bound (by decide)]
  have row : edge 3469 = (SortedNatTree.contains FastRows.row3469) := by
    change Data.row3469.contains = (SortedNatTree.contains FastRows.row3469)
    exact FastRows.row3469_checked.symm
  rw [row, root0Candidates67Block0_checked]
def root0Candidates67Block1 : List Nat := [14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349, 22549]
set_option maxRecDepth 4096 in
theorem root0Candidates67Block1_shape : root0Tail196 = root0Candidates67Block1 ++ root0Tail295 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates67Block1_checked : root0Candidates67Block1.filter (SortedNatTree.contains FastRows.row3469) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates67Block1_step : candidates edge 4 3 3469 root0Tail196 seed.bound =
    [] ++ candidates edge 4 3 3469 root0Tail295 seed.bound := by
  rw [root0Candidates67Block1_shape, candidates_append_of_within edge 4 3 3469 root0Candidates67Block1 root0Tail295 seed.bound (by decide)]
  have row : edge 3469 = (SortedNatTree.contains FastRows.row3469) := by
    change Data.row3469.contains = (SortedNatTree.contains FastRows.row3469)
    exact FastRows.row3469_checked.symm
  rw [row, root0Candidates67Block1_checked]
theorem root0Candidates67_end : candidates edge 4 3 3469 root0Tail295 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates67_checked : candidates edge 4 3 3469 root0Tail68 seed.bound = root0Candidates67 := by
  rw [root0Candidates67Block0_step, root0Candidates67Block1_step, root0Candidates67_end]
  rfl
theorem root0Step67 : scan edge (search edge 3) 4 [3] 3 root0Tail67 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail68 seed := by
  rw [root0Tail67, scan, if_neg (by decide), root0Candidates67_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail68
    (if 3 ≤ root0Candidates67.length then search edge 3 [3, 3469] root0Candidates67 3472 seed else seed) = _
  rw [if_neg (by decide)]
def root0Candidates68 : List Nat := [5839, 6871, 10771, 11827, 12451, 14851]
def root0Candidates68Block0 : List Nat := [3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779]
set_option maxRecDepth 4096 in
theorem root0Candidates68Block0_shape : root0Tail69 = root0Candidates68Block0 ++ root0Tail197 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates68Block0_checked : root0Candidates68Block0.filter (SortedNatTree.contains FastRows.row3637) = [5839, 6871, 10771, 11827, 12451] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates68Block0_step : candidates edge 4 3 3637 root0Tail69 seed.bound =
    [5839, 6871, 10771, 11827, 12451] ++ candidates edge 4 3 3637 root0Tail197 seed.bound := by
  rw [root0Candidates68Block0_shape, candidates_append_of_within edge 4 3 3637 root0Candidates68Block0 root0Tail197 seed.bound (by decide)]
  have row : edge 3637 = (SortedNatTree.contains FastRows.row3637) := by
    change Data.row3637.contains = (SortedNatTree.contains FastRows.row3637)
    exact FastRows.row3637_checked.symm
  rw [row, root0Candidates68Block0_checked]
def root0Candidates68Block1 : List Nat := [14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229, 22271, 22349]
set_option maxRecDepth 4096 in
theorem root0Candidates68Block1_shape : root0Tail197 = root0Candidates68Block1 ++ root0Tail294 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates68Block1_checked : root0Candidates68Block1.filter (SortedNatTree.contains FastRows.row3637) = [14851] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates68Block1_step : candidates edge 4 3 3637 root0Tail197 seed.bound =
    [14851] ++ candidates edge 4 3 3637 root0Tail294 seed.bound := by
  rw [root0Candidates68Block1_shape, candidates_append_of_within edge 4 3 3637 root0Candidates68Block1 root0Tail294 seed.bound (by decide)]
  have row : edge 3637 = (SortedNatTree.contains FastRows.row3637) := by
    change Data.row3637.contains = (SortedNatTree.contains FastRows.row3637)
    exact FastRows.row3637_checked.symm
  rw [row, root0Candidates68Block1_checked]
theorem root0Candidates68_end : candidates edge 4 3 3637 root0Tail294 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates68_checked : candidates edge 4 3 3637 root0Tail69 seed.bound = root0Candidates68 := by
  rw [root0Candidates68Block0_step, root0Candidates68Block1_step, root0Candidates68_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child68 : search edge 3 [3, 3637] root0Candidates68 3640 seed = seed := by decide
theorem root0Step68 : scan edge (search edge 3) 4 [3] 3 root0Tail68 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail69 seed := by
  rw [root0Tail68, scan, if_neg (by decide), root0Candidates68_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail69
    (if 3 ≤ root0Candidates68.length then search edge 3 [3, 3637] root0Candidates68 3640 seed else seed) = _
  rw [if_pos (by decide), root0Child68]
def root0Candidates69 : List Nat := [14779, 19687, 21313]
def root0Candidates69Block0 : List Nat := [3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851]
set_option maxRecDepth 4096 in
theorem root0Candidates69Block0_shape : root0Tail70 = root0Candidates69Block0 ++ root0Tail198 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates69Block0_checked : root0Candidates69Block0.filter (SortedNatTree.contains FastRows.row3769) = [14779] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates69Block0_step : candidates edge 4 3 3769 root0Tail70 seed.bound =
    [14779] ++ candidates edge 4 3 3769 root0Tail198 seed.bound := by
  rw [root0Candidates69Block0_shape, candidates_append_of_within edge 4 3 3769 root0Candidates69Block0 root0Tail198 seed.bound (by decide)]
  have row : edge 3769 = (SortedNatTree.contains FastRows.row3769) := by
    change Data.row3769.contains = (SortedNatTree.contains FastRows.row3769)
    exact FastRows.row3769_checked.symm
  rw [row, root0Candidates69Block0_checked]
def root0Candidates69Block1 : List Nat := [14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109, 22171, 22229]
set_option maxRecDepth 4096 in
theorem root0Candidates69Block1_shape : root0Tail198 = root0Candidates69Block1 ++ root0Tail292 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates69Block1_checked : root0Candidates69Block1.filter (SortedNatTree.contains FastRows.row3769) = [19687, 21313] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates69Block1_step : candidates edge 4 3 3769 root0Tail198 seed.bound =
    [19687, 21313] ++ candidates edge 4 3 3769 root0Tail292 seed.bound := by
  rw [root0Candidates69Block1_shape, candidates_append_of_within edge 4 3 3769 root0Candidates69Block1 root0Tail292 seed.bound (by decide)]
  have row : edge 3769 = (SortedNatTree.contains FastRows.row3769) := by
    change Data.row3769.contains = (SortedNatTree.contains FastRows.row3769)
    exact FastRows.row3769_checked.symm
  rw [row, root0Candidates69Block1_checked]
theorem root0Candidates69_end : candidates edge 4 3 3769 root0Tail292 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates69_checked : candidates edge 4 3 3769 root0Tail70 seed.bound = root0Candidates69 := by
  rw [root0Candidates69Block0_step, root0Candidates69Block1_step, root0Candidates69_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child69 : search edge 3 [3, 3769] root0Candidates69 3772 seed = seed := by decide
theorem root0Step69 : scan edge (search edge 3) 4 [3] 3 root0Tail69 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail70 seed := by
  rw [root0Tail69, scan, if_neg (by decide), root0Candidates69_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail70
    (if 3 ≤ root0Candidates69.length then search edge 3 [3, 3769] root0Candidates69 3772 seed else seed) = _
  rw [if_pos (by decide), root0Child69]
def root0Candidates70 : List Nat := [12269]
def root0Candidates70Block0 : List Nat := [3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879]
set_option maxRecDepth 4096 in
theorem root0Candidates70Block0_shape : root0Tail71 = root0Candidates70Block0 ++ root0Tail199 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates70Block0_checked : root0Candidates70Block0.filter (SortedNatTree.contains FastRows.row3911) = [12269] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates70Block0_step : candidates edge 4 3 3911 root0Tail71 seed.bound =
    [12269] ++ candidates edge 4 3 3911 root0Tail199 seed.bound := by
  rw [root0Candidates70Block0_shape, candidates_append_of_within edge 4 3 3911 root0Candidates70Block0 root0Tail199 seed.bound (by decide)]
  have row : edge 3911 = (SortedNatTree.contains FastRows.row3911) := by
    change Data.row3911.contains = (SortedNatTree.contains FastRows.row3911)
    exact FastRows.row3911_checked.symm
  rw [row, root0Candidates70Block0_checked]
def root0Candidates70Block1 : List Nat := [15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093, 22109]
set_option maxRecDepth 4096 in
theorem root0Candidates70Block1_shape : root0Tail199 = root0Candidates70Block1 ++ root0Tail290 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates70Block1_checked : root0Candidates70Block1.filter (SortedNatTree.contains FastRows.row3911) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates70Block1_step : candidates edge 4 3 3911 root0Tail199 seed.bound =
    [] ++ candidates edge 4 3 3911 root0Tail290 seed.bound := by
  rw [root0Candidates70Block1_shape, candidates_append_of_within edge 4 3 3911 root0Candidates70Block1 root0Tail290 seed.bound (by decide)]
  have row : edge 3911 = (SortedNatTree.contains FastRows.row3911) := by
    change Data.row3911.contains = (SortedNatTree.contains FastRows.row3911)
    exact FastRows.row3911_checked.symm
  rw [row, root0Candidates70Block1_checked]
theorem root0Candidates70_end : candidates edge 4 3 3911 root0Tail290 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates70_checked : candidates edge 4 3 3911 root0Tail71 seed.bound = root0Candidates70 := by
  rw [root0Candidates70Block0_step, root0Candidates70Block1_step, root0Candidates70_end]
  rfl
theorem root0Step70 : scan edge (search edge 3) 4 [3] 3 root0Tail70 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail71 seed := by
  rw [root0Tail70, scan, if_neg (by decide), root0Candidates70_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail71
    (if 3 ≤ root0Candidates70.length then search edge 3 [3, 3911] root0Candidates70 3914 seed else seed) = _
  rw [if_neg (by decide)]
def root0Candidates71 : List Nat := [4919, 8237, 8783, 18233]
def root0Candidates71Block0 : List Nat := [3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083]
set_option maxRecDepth 4096 in
theorem root0Candidates71Block0_shape : root0Tail72 = root0Candidates71Block0 ++ root0Tail200 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates71Block0_checked : root0Candidates71Block0.filter (SortedNatTree.contains FastRows.row3923) = [4919, 8237, 8783] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates71Block0_step : candidates edge 4 3 3923 root0Tail72 seed.bound =
    [4919, 8237, 8783] ++ candidates edge 4 3 3923 root0Tail200 seed.bound := by
  rw [root0Candidates71Block0_shape, candidates_append_of_within edge 4 3 3923 root0Candidates71Block0 root0Tail200 seed.bound (by decide)]
  have row : edge 3923 = (SortedNatTree.contains FastRows.row3923) := by
    change Data.row3923.contains = (SortedNatTree.contains FastRows.row3923)
    exact FastRows.row3923_checked.symm
  rw [row, root0Candidates71Block0_checked]
def root0Candidates71Block1 : List Nat := [15313, 15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093]
set_option maxRecDepth 4096 in
theorem root0Candidates71Block1_shape : root0Tail200 = root0Candidates71Block1 ++ root0Tail289 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates71Block1_checked : root0Candidates71Block1.filter (SortedNatTree.contains FastRows.row3923) = [18233] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates71Block1_step : candidates edge 4 3 3923 root0Tail200 seed.bound =
    [18233] ++ candidates edge 4 3 3923 root0Tail289 seed.bound := by
  rw [root0Candidates71Block1_shape, candidates_append_of_within edge 4 3 3923 root0Candidates71Block1 root0Tail289 seed.bound (by decide)]
  have row : edge 3923 = (SortedNatTree.contains FastRows.row3923) := by
    change Data.row3923.contains = (SortedNatTree.contains FastRows.row3923)
    exact FastRows.row3923_checked.symm
  rw [row, root0Candidates71Block1_checked]
theorem root0Candidates71_end : candidates edge 4 3 3923 root0Tail289 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates71_checked : candidates edge 4 3 3923 root0Tail72 seed.bound = root0Candidates71 := by
  rw [root0Candidates71Block0_step, root0Candidates71Block1_step, root0Candidates71_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child71 : search edge 3 [3, 3923] root0Candidates71 3926 seed = seed := by decide
theorem root0Step71 : scan edge (search edge 3) 4 [3] 3 root0Tail71 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail72 seed := by
  rw [root0Tail71, scan, if_neg (by decide), root0Candidates71_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail72
    (if 3 ≤ root0Candidates71.length then search edge 3 [3, 3923] root0Candidates71 3926 seed else seed) = _
  rw [if_pos (by decide), root0Child71]
def root0Candidates72 : List Nat := [5449, 17029, 20149, 21397]
def root0Candidates72Block0 : List Nat := [4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313]
set_option maxRecDepth 4096 in
theorem root0Candidates72Block0_shape : root0Tail73 = root0Candidates72Block0 ++ root0Tail201 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates72Block0_checked : root0Candidates72Block0.filter (SortedNatTree.contains FastRows.row3931) = [5449] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates72Block0_step : candidates edge 4 3 3931 root0Tail73 seed.bound =
    [5449] ++ candidates edge 4 3 3931 root0Tail201 seed.bound := by
  rw [root0Candidates72Block0_shape, candidates_append_of_within edge 4 3 3931 root0Candidates72Block0 root0Tail201 seed.bound (by decide)]
  have row : edge 3931 = (SortedNatTree.contains FastRows.row3931) := by
    change Data.row3931.contains = (SortedNatTree.contains FastRows.row3931)
    exact FastRows.row3931_checked.symm
  rw [row, root0Candidates72Block0_checked]
def root0Candidates72Block1 : List Nat := [15373, 15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961, 22037, 22051, 22067, 22079, 22093]
set_option maxRecDepth 4096 in
theorem root0Candidates72Block1_shape : root0Tail201 = root0Candidates72Block1 ++ root0Tail289 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates72Block1_checked : root0Candidates72Block1.filter (SortedNatTree.contains FastRows.row3931) = [17029, 20149, 21397] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates72Block1_step : candidates edge 4 3 3931 root0Tail201 seed.bound =
    [17029, 20149, 21397] ++ candidates edge 4 3 3931 root0Tail289 seed.bound := by
  rw [root0Candidates72Block1_shape, candidates_append_of_within edge 4 3 3931 root0Candidates72Block1 root0Tail289 seed.bound (by decide)]
  have row : edge 3931 = (SortedNatTree.contains FastRows.row3931) := by
    change Data.row3931.contains = (SortedNatTree.contains FastRows.row3931)
    exact FastRows.row3931_checked.symm
  rw [row, root0Candidates72Block1_checked]
theorem root0Candidates72_end : candidates edge 4 3 3931 root0Tail289 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates72_checked : candidates edge 4 3 3931 root0Tail73 seed.bound = root0Candidates72 := by
  rw [root0Candidates72Block0_step, root0Candidates72Block1_step, root0Candidates72_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child72 : search edge 3 [3, 3931] root0Candidates72 3934 seed = seed := by decide
theorem root0Step72 : scan edge (search edge 3) 4 [3] 3 root0Tail72 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail73 seed := by
  rw [root0Tail72, scan, if_neg (by decide), root0Candidates72_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail73
    (if 3 ≤ root0Candidates72.length then search edge 3 [3, 3931] root0Candidates72 3934 seed else seed) = _
  rw [if_pos (by decide), root0Child72]
def root0Candidates73 : List Nat := [17489, 17903]
def root0Candidates73Block0 : List Nat := [4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373]
set_option maxRecDepth 4096 in
theorem root0Candidates73Block0_shape : root0Tail74 = root0Candidates73Block0 ++ root0Tail202 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates73Block0_checked : root0Candidates73Block0.filter (SortedNatTree.contains FastRows.row4019) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates73Block0_step : candidates edge 4 3 4019 root0Tail74 seed.bound =
    [] ++ candidates edge 4 3 4019 root0Tail202 seed.bound := by
  rw [root0Candidates73Block0_shape, candidates_append_of_within edge 4 3 4019 root0Candidates73Block0 root0Tail202 seed.bound (by decide)]
  have row : edge 4019 = (SortedNatTree.contains FastRows.row4019) := by
    change Data.row4019.contains = (SortedNatTree.contains FastRows.row4019)
    exact FastRows.row4019_checked.symm
  rw [row, root0Candidates73Block0_checked]
def root0Candidates73Block1 : List Nat := [15461, 15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851, 21961]
set_option maxRecDepth 4096 in
theorem root0Candidates73Block1_shape : root0Tail202 = root0Candidates73Block1 ++ root0Tail284 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates73Block1_checked : root0Candidates73Block1.filter (SortedNatTree.contains FastRows.row4019) = [17489, 17903] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates73Block1_step : candidates edge 4 3 4019 root0Tail202 seed.bound =
    [17489, 17903] ++ candidates edge 4 3 4019 root0Tail284 seed.bound := by
  rw [root0Candidates73Block1_shape, candidates_append_of_within edge 4 3 4019 root0Candidates73Block1 root0Tail284 seed.bound (by decide)]
  have row : edge 4019 = (SortedNatTree.contains FastRows.row4019) := by
    change Data.row4019.contains = (SortedNatTree.contains FastRows.row4019)
    exact FastRows.row4019_checked.symm
  rw [row, root0Candidates73Block1_checked]
theorem root0Candidates73_end : candidates edge 4 3 4019 root0Tail284 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates73_checked : candidates edge 4 3 4019 root0Tail74 seed.bound = root0Candidates73 := by
  rw [root0Candidates73Block0_step, root0Candidates73Block1_step, root0Candidates73_end]
  rfl
theorem root0Step73 : scan edge (search edge 3) 4 [3] 3 root0Tail73 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail74 seed := by
  rw [root0Tail73, scan, if_neg (by decide), root0Candidates73_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail74
    (if 3 ≤ root0Candidates73.length then search edge 3 [3, 4019] root0Candidates73 4022 seed else seed) = _
  rw [if_neg (by decide)]
def root0Candidates74 : List Nat := [16759, 17209, 17839]
def root0Candidates74Block0 : List Nat := [4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461]
set_option maxRecDepth 4096 in
theorem root0Candidates74Block0_shape : root0Tail75 = root0Candidates74Block0 ++ root0Tail203 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates74Block0_checked : root0Candidates74Block0.filter (SortedNatTree.contains FastRows.row4159) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates74Block0_step : candidates edge 4 3 4159 root0Tail75 seed.bound =
    [] ++ candidates edge 4 3 4159 root0Tail203 seed.bound := by
  rw [root0Candidates74Block0_shape, candidates_append_of_within edge 4 3 4159 root0Candidates74Block0 root0Tail203 seed.bound (by decide)]
  have row : edge 4159 = (SortedNatTree.contains FastRows.row4159) := by
    change Data.row4159.contains = (SortedNatTree.contains FastRows.row4159)
    exact FastRows.row4159_checked.symm
  rw [row, root0Candidates74Block0_checked]
def root0Candidates74Block1 : List Nat := [15493, 15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757, 21821, 21851]
set_option maxRecDepth 4096 in
theorem root0Candidates74Block1_shape : root0Tail203 = root0Candidates74Block1 ++ root0Tail283 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates74Block1_checked : root0Candidates74Block1.filter (SortedNatTree.contains FastRows.row4159) = [16759, 17209, 17839] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates74Block1_step : candidates edge 4 3 4159 root0Tail203 seed.bound =
    [16759, 17209, 17839] ++ candidates edge 4 3 4159 root0Tail283 seed.bound := by
  rw [root0Candidates74Block1_shape, candidates_append_of_within edge 4 3 4159 root0Candidates74Block1 root0Tail283 seed.bound (by decide)]
  have row : edge 4159 = (SortedNatTree.contains FastRows.row4159) := by
    change Data.row4159.contains = (SortedNatTree.contains FastRows.row4159)
    exact FastRows.row4159_checked.symm
  rw [row, root0Candidates74Block1_checked]
theorem root0Candidates74_end : candidates edge 4 3 4159 root0Tail283 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates74_checked : candidates edge 4 3 4159 root0Tail75 seed.bound = root0Candidates74 := by
  rw [root0Candidates74Block0_step, root0Candidates74Block1_step, root0Candidates74_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child74 : search edge 3 [3, 4159] root0Candidates74 4162 seed = seed := by decide
theorem root0Step74 : scan edge (search edge 3) 4 [3] 3 root0Tail74 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail75 seed := by
  rw [root0Tail74, scan, if_neg (by decide), root0Candidates74_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail75
    (if 3 ≤ root0Candidates74.length then search edge 3 [3, 4159] root0Candidates74 4162 seed else seed) = _
  rw [if_pos (by decide), root0Child74]
def root0Candidates75 : List Nat := [9521, 11867, 12101, 15461, 16937]
def root0Candidates75Block0 : List Nat := [4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493]
set_option maxRecDepth 4096 in
theorem root0Candidates75Block0_shape : root0Tail76 = root0Candidates75Block0 ++ root0Tail204 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates75Block0_checked : root0Candidates75Block0.filter (SortedNatTree.contains FastRows.row4253) = [9521, 11867, 12101, 15461] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates75Block0_step : candidates edge 4 3 4253 root0Tail76 seed.bound =
    [9521, 11867, 12101, 15461] ++ candidates edge 4 3 4253 root0Tail204 seed.bound := by
  rw [root0Candidates75Block0_shape, candidates_append_of_within edge 4 3 4253 root0Candidates75Block0 root0Tail204 seed.bound (by decide)]
  have row : edge 4253 = (SortedNatTree.contains FastRows.row4253) := by
    change Data.row4253.contains = (SortedNatTree.contains FastRows.row4253)
    exact FastRows.row4253_checked.symm
  rw [row, root0Candidates75Block0_checked]
def root0Candidates75Block1 : List Nat := [15559, 15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397, 21467, 21569, 21611, 21617, 21757]
set_option maxRecDepth 4096 in
theorem root0Candidates75Block1_shape : root0Tail204 = root0Candidates75Block1 ++ root0Tail281 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates75Block1_checked : root0Candidates75Block1.filter (SortedNatTree.contains FastRows.row4253) = [16937] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates75Block1_step : candidates edge 4 3 4253 root0Tail204 seed.bound =
    [16937] ++ candidates edge 4 3 4253 root0Tail281 seed.bound := by
  rw [root0Candidates75Block1_shape, candidates_append_of_within edge 4 3 4253 root0Candidates75Block1 root0Tail281 seed.bound (by decide)]
  have row : edge 4253 = (SortedNatTree.contains FastRows.row4253) := by
    change Data.row4253.contains = (SortedNatTree.contains FastRows.row4253)
    exact FastRows.row4253_checked.symm
  rw [row, root0Candidates75Block1_checked]
theorem root0Candidates75_end : candidates edge 4 3 4253 root0Tail281 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates75_checked : candidates edge 4 3 4253 root0Tail76 seed.bound = root0Candidates75 := by
  rw [root0Candidates75Block0_step, root0Candidates75Block1_step, root0Candidates75_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child75 : search edge 3 [3, 4253] root0Candidates75 4256 seed = seed := by decide
theorem root0Step75 : scan edge (search edge 3) 4 [3] 3 root0Tail75 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail76 seed := by
  rw [root0Tail75, scan, if_neg (by decide), root0Candidates75_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail76
    (if 3 ≤ root0Candidates75.length then search edge 3 [3, 4253] root0Candidates75 4256 seed else seed) = _
  rw [if_pos (by decide), root0Child75]
def root0Candidates76 : List Nat := [18713, 20939]
def root0Candidates76Block0 : List Nat := [4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559]
set_option maxRecDepth 4096 in
theorem root0Candidates76Block0_shape : root0Tail77 = root0Candidates76Block0 ++ root0Tail205 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates76Block0_checked : root0Candidates76Block0.filter (SortedNatTree.contains FastRows.row4583) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates76Block0_step : candidates edge 4 3 4583 root0Tail77 seed.bound =
    [] ++ candidates edge 4 3 4583 root0Tail205 seed.bound := by
  rw [root0Candidates76Block0_shape, candidates_append_of_within edge 4 3 4583 root0Candidates76Block0 root0Tail205 seed.bound (by decide)]
  have row : edge 4583 = (SortedNatTree.contains FastRows.row4583) := by
    change Data.row4583.contains = (SortedNatTree.contains FastRows.row4583)
    exact FastRows.row4583_checked.symm
  rw [row, root0Candidates76Block0_checked]
def root0Candidates76Block1 : List Nat := [15569, 15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193, 21313, 21319, 21383, 21397]
set_option maxRecDepth 4096 in
theorem root0Candidates76Block1_shape : root0Tail205 = root0Candidates76Block1 ++ root0Tail276 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates76Block1_checked : root0Candidates76Block1.filter (SortedNatTree.contains FastRows.row4583) = [18713, 20939] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates76Block1_step : candidates edge 4 3 4583 root0Tail205 seed.bound =
    [18713, 20939] ++ candidates edge 4 3 4583 root0Tail276 seed.bound := by
  rw [root0Candidates76Block1_shape, candidates_append_of_within edge 4 3 4583 root0Candidates76Block1 root0Tail276 seed.bound (by decide)]
  have row : edge 4583 = (SortedNatTree.contains FastRows.row4583) := by
    change Data.row4583.contains = (SortedNatTree.contains FastRows.row4583)
    exact FastRows.row4583_checked.symm
  rw [row, root0Candidates76Block1_checked]
theorem root0Candidates76_end : candidates edge 4 3 4583 root0Tail276 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates76_checked : candidates edge 4 3 4583 root0Tail77 seed.bound = root0Candidates76 := by
  rw [root0Candidates76Block0_step, root0Candidates76Block1_step, root0Candidates76_end]
  rfl
theorem root0Step76 : scan edge (search edge 3) 4 [3] 3 root0Tail76 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail77 seed := by
  rw [root0Tail76, scan, if_neg (by decide), root0Candidates76_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail77
    (if 3 ≤ root0Candidates76.length then search edge 3 [3, 4583] root0Candidates76 4586 seed else seed) = _
  rw [if_neg (by decide)]
def root0Candidates77 : List Nat := [7963, 8431, 9181, 15559, 18181, 18289, 19441, 20269]
def root0Candidates77Block0 : List Nat := [4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569]
set_option maxRecDepth 4096 in
theorem root0Candidates77Block0_shape : root0Tail78 = root0Candidates77Block0 ++ root0Tail206 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates77Block0_checked : root0Candidates77Block0.filter (SortedNatTree.contains FastRows.row4729) = [7963, 8431, 9181, 15559] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates77Block0_step : candidates edge 4 3 4729 root0Tail78 seed.bound =
    [7963, 8431, 9181, 15559] ++ candidates edge 4 3 4729 root0Tail206 seed.bound := by
  rw [root0Candidates77Block0_shape, candidates_append_of_within edge 4 3 4729 root0Candidates77Block0 root0Tail206 seed.bound (by decide)]
  have row : edge 4729 = (SortedNatTree.contains FastRows.row4729) := by
    change Data.row4729.contains = (SortedNatTree.contains FastRows.row4729)
    exact FastRows.row4729_checked.symm
  rw [row, root0Candidates77Block0_checked]
def root0Candidates77Block1 : List Nat := [15739, 15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017, 21143, 21169, 21187, 21193]
set_option maxRecDepth 4096 in
theorem root0Candidates77Block1_shape : root0Tail206 = root0Candidates77Block1 ++ root0Tail272 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates77Block1_checked : root0Candidates77Block1.filter (SortedNatTree.contains FastRows.row4729) = [18181, 18289, 19441, 20269] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates77Block1_step : candidates edge 4 3 4729 root0Tail206 seed.bound =
    [18181, 18289, 19441, 20269] ++ candidates edge 4 3 4729 root0Tail272 seed.bound := by
  rw [root0Candidates77Block1_shape, candidates_append_of_within edge 4 3 4729 root0Candidates77Block1 root0Tail272 seed.bound (by decide)]
  have row : edge 4729 = (SortedNatTree.contains FastRows.row4729) := by
    change Data.row4729.contains = (SortedNatTree.contains FastRows.row4729)
    exact FastRows.row4729_checked.symm
  rw [row, root0Candidates77Block1_checked]
theorem root0Candidates77_end : candidates edge 4 3 4729 root0Tail272 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates77_checked : candidates edge 4 3 4729 root0Tail78 seed.bound = root0Candidates77 := by
  rw [root0Candidates77Block0_step, root0Candidates77Block1_step, root0Candidates77_end]
  rfl
set_option maxRecDepth 4096 in
theorem root0Child77 : search edge 3 [3, 4729] root0Candidates77 4732 seed = seed := by decide
theorem root0Step77 : scan edge (search edge 3) 4 [3] 3 root0Tail77 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail78 seed := by
  rw [root0Tail77, scan, if_neg (by decide), root0Candidates77_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail78
    (if 3 ≤ root0Candidates77.length then search edge 3 [3, 4729] root0Candidates77 4732 seed else seed) = _
  rw [if_pos (by decide), root0Child77]
def root0Candidates78 : List Nat := [6599, 8783]
def root0Candidates78Block0 : List Nat := [5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377, 8431, 8629, 8669, 8699, 8713, 8747, 8783, 9103, 9157, 9181, 9241, 9461, 9521, 9623, 9857, 9887, 9901, 10091, 10111, 10169, 10223, 10243, 10333, 10357, 10459, 10567, 10627, 10771, 10889, 10987, 11027, 11173, 11177, 11197, 11393, 11447, 11551, 11821, 11827, 11867, 11897, 11981, 12071, 12101, 12203, 12253, 12269, 12343, 12413, 12451, 12601, 12703, 12841, 12899, 12979, 13381, 13399, 13553, 13613, 13619, 13669, 13763, 13879, 13931, 13933, 14107, 14243, 14401, 14407, 14543, 14627, 14779, 14851, 14879, 15083, 15313, 15373, 15461, 15493, 15559, 15569, 15739]
set_option maxRecDepth 4096 in
theorem root0Candidates78Block0_shape : root0Tail79 = root0Candidates78Block0 ++ root0Tail207 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates78Block0_checked : root0Candidates78Block0.filter (SortedNatTree.contains FastRows.row4919) = [6599, 8783] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates78Block0_step : candidates edge 4 3 4919 root0Tail79 seed.bound =
    [6599, 8783] ++ candidates edge 4 3 4919 root0Tail207 seed.bound := by
  rw [root0Candidates78Block0_shape, candidates_append_of_within edge 4 3 4919 root0Candidates78Block0 root0Tail207 seed.bound (by decide)]
  have row : edge 4919 = (SortedNatTree.contains FastRows.row4919) := by
    change Data.row4919.contains = (SortedNatTree.contains FastRows.row4919)
    exact FastRows.row4919_checked.symm
  rw [row, root0Candidates78Block0_checked]
def root0Candidates78Block1 : List Nat := [15907, 16339, 16363, 16567, 16661, 16703, 16759, 16937, 17021, 17029, 17047, 17077, 17123, 17209, 17257, 17327, 17489, 17539, 17599, 17789, 17839, 17903, 17921, 17923, 17957, 17959, 18077, 18127, 18181, 18191, 18233, 18289, 18523, 18679, 18713, 18743, 19031, 19183, 19211, 19237, 19387, 19441, 19541, 19687, 19699, 19727, 19819, 19919, 19937, 19993, 20107, 20149, 20269, 20521, 20563, 20627, 20693, 20759, 20899, 20939, 21017]
set_option maxRecDepth 4096 in
theorem root0Candidates78Block1_shape : root0Tail207 = root0Candidates78Block1 ++ root0Tail268 := rfl
set_option maxRecDepth 4096 in
theorem root0Candidates78Block1_checked : root0Candidates78Block1.filter (SortedNatTree.contains FastRows.row4919) = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates78Block1_step : candidates edge 4 3 4919 root0Tail207 seed.bound =
    [] ++ candidates edge 4 3 4919 root0Tail268 seed.bound := by
  rw [root0Candidates78Block1_shape, candidates_append_of_within edge 4 3 4919 root0Candidates78Block1 root0Tail268 seed.bound (by decide)]
  have row : edge 4919 = (SortedNatTree.contains FastRows.row4919) := by
    change Data.row4919.contains = (SortedNatTree.contains FastRows.row4919)
    exact FastRows.row4919_checked.symm
  rw [row, root0Candidates78Block1_checked]
theorem root0Candidates78_end : candidates edge 4 3 4919 root0Tail268 seed.bound = [] := by decide
set_option maxRecDepth 4096 in
theorem root0Candidates78_checked : candidates edge 4 3 4919 root0Tail79 seed.bound = root0Candidates78 := by
  rw [root0Candidates78Block0_step, root0Candidates78Block1_step, root0Candidates78_end]
  rfl
theorem root0Step78 : scan edge (search edge 3) 4 [3] 3 root0Tail78 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail79 seed := by
  rw [root0Tail78, scan, if_neg (by decide), root0Candidates78_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail79
    (if 3 ≤ root0Candidates78.length then search edge 3 [3, 4919] root0Candidates78 4922 seed else seed) = _
  rw [if_neg (by decide)]
def root0Candidates79 : List Nat := [7253, 11393, 18233]
set_option maxRecDepth 8192 in
theorem root0Candidates79_checked : candidates edge 4 3 5051 root0Tail80 seed.bound = root0Candidates79 := by
  unfold candidates
  have row : edge 5051 = (SortedNatTree.contains FastRows.row5051) := by
    change Data.row5051.contains = (SortedNatTree.contains FastRows.row5051)
    exact FastRows.row5051_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root0Child79 : search edge 3 [3, 5051] root0Candidates79 5054 seed = seed := by decide
theorem root0Step79 : scan edge (search edge 3) 4 [3] 3 root0Tail79 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail80 seed := by
  rw [root0Tail79, scan, if_neg (by decide), root0Candidates79_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail80
    (if 3 ≤ root0Candidates79.length then search edge 3 [3, 5051] root0Candidates79 5054 seed else seed) = _
  rw [if_pos (by decide), root0Child79]
def root0Candidates80 : List Nat := [5323]
set_option maxRecDepth 8192 in
theorem root0Candidates80_checked : candidates edge 4 3 5059 root0Tail81 seed.bound = root0Candidates80 := by
  unfold candidates
  have row : edge 5059 = (SortedNatTree.contains FastRows.row5059) := by
    change Data.row5059.contains = (SortedNatTree.contains FastRows.row5059)
    exact FastRows.row5059_checked.symm
  rw [row]
  decide
theorem root0Step80 : scan edge (search edge 3) 4 [3] 3 root0Tail80 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail81 seed := by
  rw [root0Tail80, scan, if_neg (by decide), root0Candidates80_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail81
    (if 3 ≤ root0Candidates80.length then search edge 3 [3, 5059] root0Candidates80 5062 seed else seed) = _
  rw [if_neg (by decide)]
def root0Candidates81 : List Nat := [5507, 7649, 13619]
set_option maxRecDepth 8192 in
theorem root0Candidates81_checked : candidates edge 4 3 5099 root0Tail82 seed.bound = root0Candidates81 := by
  unfold candidates
  have row : edge 5099 = (SortedNatTree.contains FastRows.row5099) := by
    change Data.row5099.contains = (SortedNatTree.contains FastRows.row5099)
    exact FastRows.row5099_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root0Child81 : search edge 3 [3, 5099] root0Candidates81 5102 seed = seed := by decide
theorem root0Step81 : scan edge (search edge 3) 4 [3] 3 root0Tail81 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail82 seed := by
  rw [root0Tail81, scan, if_neg (by decide), root0Candidates81_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail82
    (if 3 ≤ root0Candidates81.length then search edge 3 [3, 5099] root0Candidates81 5102 seed else seed) = _
  rw [if_pos (by decide), root0Child81]
def root0Candidates82 : List Nat := [18713, 20693]
set_option maxRecDepth 8192 in
theorem root0Candidates82_checked : candidates edge 4 3 5171 root0Tail83 seed.bound = root0Candidates82 := by
  unfold candidates
  have row : edge 5171 = (SortedNatTree.contains FastRows.row5171) := by
    change Data.row5171.contains = (SortedNatTree.contains FastRows.row5171)
    exact FastRows.row5171_checked.symm
  rw [row]
  decide
theorem root0Step82 : scan edge (search edge 3) 4 [3] 3 root0Tail82 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail83 seed := by
  rw [root0Tail82, scan, if_neg (by decide), root0Candidates82_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail83
    (if 3 ≤ root0Candidates82.length then search edge 3 [3, 5171] root0Candidates82 5174 seed else seed) = _
  rw [if_neg (by decide)]
def root0Candidates83 : List Nat := [5869, 7159, 12703, 14107, 15493]
set_option maxRecDepth 8192 in
theorem root0Candidates83_checked : candidates edge 4 3 5281 root0Tail84 seed.bound = root0Candidates83 := by
  unfold candidates
  have row : edge 5281 = (SortedNatTree.contains FastRows.row5281) := by
    change Data.row5281.contains = (SortedNatTree.contains FastRows.row5281)
    exact FastRows.row5281_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root0Child83 : search edge 3 [3, 5281] root0Candidates83 5284 seed = seed := by decide
theorem root0Step83 : scan edge (search edge 3) 4 [3] 3 root0Tail83 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail84 seed := by
  rw [root0Tail83, scan, if_neg (by decide), root0Candidates83_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail84
    (if 3 ≤ root0Candidates83.length then search edge 3 [3, 5281] root0Candidates83 5284 seed else seed) = _
  rw [if_pos (by decide), root0Child83]
def root0Candidates84 : List Nat := [5521, 6037, 10357, 17257, 19387]
set_option maxRecDepth 8192 in
theorem root0Candidates84_checked : candidates edge 4 3 5323 root0Tail85 seed.bound = root0Candidates84 := by
  unfold candidates
  have row : edge 5323 = (SortedNatTree.contains FastRows.row5323) := by
    change Data.row5323.contains = (SortedNatTree.contains FastRows.row5323)
    exact FastRows.row5323_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root0Child84 : search edge 3 [3, 5323] root0Candidates84 5326 seed = seed := by decide
theorem root0Step84 : scan edge (search edge 3) 4 [3] 3 root0Tail84 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail85 seed := by
  rw [root0Tail84, scan, if_neg (by decide), root0Candidates84_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail85
    (if 3 ≤ root0Candidates84.length then search edge 3 [3, 5323] root0Candidates84 5326 seed else seed) = _
  rw [if_pos (by decide), root0Child84]
def root0Candidates85 : List Nat := [5507, 6599, 12269, 13613, 13931]
set_option maxRecDepth 8192 in
theorem root0Candidates85_checked : candidates edge 4 3 5381 root0Tail86 seed.bound = root0Candidates85 := by
  unfold candidates
  have row : edge 5381 = (SortedNatTree.contains FastRows.row5381) := by
    change Data.row5381.contains = (SortedNatTree.contains FastRows.row5381)
    exact FastRows.row5381_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root0Child85 : search edge 3 [3, 5381] root0Candidates85 5384 seed = seed := by decide
theorem root0Step85 : scan edge (search edge 3) 4 [3] 3 root0Tail85 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail86 seed := by
  rw [root0Tail85, scan, if_neg (by decide), root0Candidates85_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail86
    (if 3 ≤ root0Candidates85.length then search edge 3 [3, 5381] root0Candidates85 5384 seed else seed) = _
  rw [if_pos (by decide), root0Child85]
def root0Candidates86 : List Nat := [9241, 10627, 12601, 13381, 13399, 20107, 20521]
set_option maxRecDepth 8192 in
theorem root0Candidates86_checked : candidates edge 4 3 5419 root0Tail87 seed.bound = root0Candidates86 := by
  unfold candidates
  have row : edge 5419 = (SortedNatTree.contains FastRows.row5419) := by
    change Data.row5419.contains = (SortedNatTree.contains FastRows.row5419)
    exact FastRows.row5419_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root0Child86 : search edge 3 [3, 5419] root0Candidates86 5422 seed = seed := by decide
theorem root0Step86 : scan edge (search edge 3) 4 [3] 3 root0Tail86 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail87 seed := by
  rw [root0Tail86, scan, if_neg (by decide), root0Candidates86_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail87
    (if 3 ≤ root0Candidates86.length then search edge 3 [3, 5419] root0Candidates86 5422 seed else seed) = _
  rw [if_pos (by decide), root0Child86]
def root0Candidates87 : List Nat := [6073, 17029]
set_option maxRecDepth 8192 in
theorem root0Candidates87_checked : candidates edge 4 3 5449 root0Tail88 seed.bound = root0Candidates87 := by
  unfold candidates
  have row : edge 5449 = (SortedNatTree.contains FastRows.row5449) := by
    change Data.row5449.contains = (SortedNatTree.contains FastRows.row5449)
    exact FastRows.row5449_checked.symm
  rw [row]
  decide
theorem root0Step87 : scan edge (search edge 3) 4 [3] 3 root0Tail87 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail88 seed := by
  rw [root0Tail87, scan, if_neg (by decide), root0Candidates87_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail88
    (if 3 ≤ root0Candidates87.length then search edge 3 [3, 5449] root0Candidates87 5452 seed else seed) = _
  rw [if_neg (by decide)]
def root0Candidates88 : List Nat := [9623, 10169]
set_option maxRecDepth 8192 in
theorem root0Candidates88_checked : candidates edge 4 3 5507 root0Tail89 seed.bound = root0Candidates88 := by
  unfold candidates
  have row : edge 5507 = (SortedNatTree.contains FastRows.row5507) := by
    change Data.row5507.contains = (SortedNatTree.contains FastRows.row5507)
    exact FastRows.row5507_checked.symm
  rw [row]
  decide
theorem root0Step88 : scan edge (search edge 3) 4 [3] 3 root0Tail88 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail89 seed := by
  rw [root0Tail88, scan, if_neg (by decide), root0Candidates88_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail89
    (if 3 ≤ root0Candidates88.length then search edge 3 [3, 5507] root0Candidates88 5510 seed else seed) = _
  rw [if_neg (by decide)]
def root0Candidates89 : List Nat := [7159, 15373, 19819]
set_option maxRecDepth 8192 in
theorem root0Candidates89_checked : candidates edge 4 3 5521 root0Tail90 seed.bound = root0Candidates89 := by
  unfold candidates
  have row : edge 5521 = (SortedNatTree.contains FastRows.row5521) := by
    change Data.row5521.contains = (SortedNatTree.contains FastRows.row5521)
    exact FastRows.row5521_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root0Child89 : search edge 3 [3, 5521] root0Candidates89 5524 seed = seed := by decide
theorem root0Step89 : scan edge (search edge 3) 4 [3] 3 root0Tail89 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail90 seed := by
  rw [root0Tail89, scan, if_neg (by decide), root0Candidates89_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail90
    (if 3 ≤ root0Candidates89.length then search edge 3 [3, 5521] root0Candidates89 5524 seed else seed) = _
  rw [if_pos (by decide), root0Child89]
def root0Candidates90 : List Nat := []
set_option maxRecDepth 8192 in
theorem root0Candidates90_checked : candidates edge 4 3 5531 root0Tail91 seed.bound = root0Candidates90 := by
  unfold candidates
  have row : edge 5531 = (SortedNatTree.contains FastRows.row5531) := by
    change Data.row5531.contains = (SortedNatTree.contains FastRows.row5531)
    exact FastRows.row5531_checked.symm
  rw [row]
  decide
theorem root0Step90 : scan edge (search edge 3) 4 [3] 3 root0Tail90 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail91 seed := by
  rw [root0Tail90, scan, if_neg (by decide), root0Candidates90_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail91
    (if 3 ≤ root0Candidates90.length then search edge 3 [3, 5531] root0Candidates90 5534 seed else seed) = _
  rw [if_neg (by decide)]
def root0Candidates91 : List Nat := [8219, 14543, 17489]
set_option maxRecDepth 8192 in
theorem root0Candidates91_checked : candidates edge 4 3 5573 root0Tail92 seed.bound = root0Candidates91 := by
  unfold candidates
  have row : edge 5573 = (SortedNatTree.contains FastRows.row5573) := by
    change Data.row5573.contains = (SortedNatTree.contains FastRows.row5573)
    exact FastRows.row5573_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root0Child91 : search edge 3 [3, 5573] root0Candidates91 5576 seed = seed := by decide
theorem root0Step91 : scan edge (search edge 3) 4 [3] 3 root0Tail91 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail92 seed := by
  rw [root0Tail91, scan, if_neg (by decide), root0Candidates91_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail92
    (if 3 ≤ root0Candidates91.length then search edge 3 [3, 5573] root0Candidates91 5576 seed else seed) = _
  rw [if_pos (by decide), root0Child91]
def root0Candidates92 : List Nat := [6263, 8783, 11027, 15461, 17789, 18191]
set_option maxRecDepth 8192 in
theorem root0Candidates92_checked : candidates edge 4 3 5801 root0Tail93 seed.bound = root0Candidates92 := by
  unfold candidates
  have row : edge 5801 = (SortedNatTree.contains FastRows.row5801) := by
    change Data.row5801.contains = (SortedNatTree.contains FastRows.row5801)
    exact FastRows.row5801_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root0Child92 : search edge 3 [3, 5801] root0Candidates92 5804 seed = seed := by decide
theorem root0Step92 : scan edge (search edge 3) 4 [3] 3 root0Tail92 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail93 seed := by
  rw [root0Tail92, scan, if_neg (by decide), root0Candidates92_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail93
    (if 3 ≤ root0Candidates92.length then search edge 3 [3, 5801] root0Candidates92 5804 seed else seed) = _
  rw [if_pos (by decide), root0Child92]
def root0Candidates93 : List Nat := [10333, 20107]
set_option maxRecDepth 8192 in
theorem root0Candidates93_checked : candidates edge 4 3 5839 root0Tail94 seed.bound = root0Candidates93 := by
  unfold candidates
  have row : edge 5839 = (SortedNatTree.contains FastRows.row5839) := by
    change Data.row5839.contains = (SortedNatTree.contains FastRows.row5839)
    exact FastRows.row5839_checked.symm
  rw [row]
  decide
theorem root0Step93 : scan edge (search edge 3) 4 [3] 3 root0Tail93 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail94 seed := by
  rw [root0Tail93, scan, if_neg (by decide), root0Candidates93_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail94
    (if 3 ≤ root0Candidates93.length then search edge 3 [3, 5839] root0Candidates93 5842 seed else seed) = _
  rw [if_neg (by decide)]
def root0Candidates94 : List Nat := [15313, 19183, 19687]
set_option maxRecDepth 8192 in
theorem root0Candidates94_checked : candidates edge 4 3 5869 root0Tail95 seed.bound = root0Candidates94 := by
  unfold candidates
  have row : edge 5869 = (SortedNatTree.contains FastRows.row5869) := by
    change Data.row5869.contains = (SortedNatTree.contains FastRows.row5869)
    exact FastRows.row5869_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root0Child94 : search edge 3 [3, 5869] root0Candidates94 5872 seed = seed := by decide
theorem root0Step94 : scan edge (search edge 3) 4 [3] 3 root0Tail94 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail95 seed := by
  rw [root0Tail94, scan, if_neg (by decide), root0Candidates94_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail95
    (if 3 ≤ root0Candidates94.length then search edge 3 [3, 5869] root0Candidates94 5872 seed else seed) = _
  rw [if_pos (by decide), root0Child94]
def root0Candidates95 : List Nat := [6871, 7963, 14407, 14851, 19687]
set_option maxRecDepth 8192 in
theorem root0Candidates95_checked : candidates edge 4 3 5923 root0Tail96 seed.bound = root0Candidates95 := by
  unfold candidates
  have row : edge 5923 = (SortedNatTree.contains FastRows.row5923) := by
    change Data.row5923.contains = (SortedNatTree.contains FastRows.row5923)
    exact FastRows.row5923_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root0Child95 : search edge 3 [3, 5923] root0Candidates95 5926 seed = seed := by decide
theorem root0Step95 : scan edge (search edge 3) 4 [3] 3 root0Tail95 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail96 seed := by
  rw [root0Tail95, scan, if_neg (by decide), root0Candidates95_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail96
    (if 3 ≤ root0Candidates95.length then search edge 3 [3, 5923] root0Candidates95 5926 seed else seed) = _
  rw [if_pos (by decide), root0Child95]
end ProjectEulerSolutions.PrimePairLarge.Search
