import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot001Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root1Candidates64 : List Nat := [9601, 12301, 19597, 20947]
set_option maxRecDepth 8192 in
theorem root1Candidates64_checked : candidates edge 4 7 4729 root1Tail65 seed.bound = root1Candidates64 := by
  unfold candidates
  have row : edge 4729 = (SortedNatTree.contains FastRows.row4729) := by
    change Data.row4729.contains = (SortedNatTree.contains FastRows.row4729)
    exact FastRows.row4729_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child64 : search edge 3 [7, 4729] root1Candidates64 4736 seed = seed := by decide
theorem root1Step64 : scan edge (search edge 3) 4 [7] 7 root1Tail64 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail65 seed := by
  rw [root1Tail64, scan, if_neg (by decide), root1Candidates64_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail65
    (if 3 ≤ root1Candidates64.length then search edge 3 [7, 4729] root1Candidates64 4736 seed else seed) = _
  rw [if_pos (by decide), root1Child64]
def root1Candidates65 : List Nat := [5659, 8839, 8893, 9337, 9349, 11287, 13597, 13627]
set_option maxRecDepth 8192 in
theorem root1Candidates65_checked : candidates edge 4 7 4903 root1Tail66 seed.bound = root1Candidates65 := by
  unfold candidates
  have row : edge 4903 = (SortedNatTree.contains FastRows.row4903) := by
    change Data.row4903.contains = (SortedNatTree.contains FastRows.row4903)
    exact FastRows.row4903_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child65 : search edge 3 [7, 4903] root1Candidates65 4910 seed = seed := by decide
theorem root1Step65 : scan edge (search edge 3) 4 [7] 7 root1Tail65 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail66 seed := by
  rw [root1Tail65, scan, if_neg (by decide), root1Candidates65_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail66
    (if 3 ≤ root1Candidates65.length then search edge 3 [7, 4903] root1Candidates65 4910 seed else seed) = _
  rw [if_pos (by decide), root1Child65]
def root1Candidates66 : List Nat := [5689, 8017, 11317, 15643]
set_option maxRecDepth 8192 in
theorem root1Candidates66_checked : candidates edge 4 7 5437 root1Tail67 seed.bound = root1Candidates66 := by
  unfold candidates
  have row : edge 5437 = (SortedNatTree.contains FastRows.row5437) := by
    change Data.row5437.contains = (SortedNatTree.contains FastRows.row5437)
    exact FastRows.row5437_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child66 : search edge 3 [7, 5437] root1Candidates66 5444 seed = seed := by decide
theorem root1Step66 : scan edge (search edge 3) 4 [7] 7 root1Tail66 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail67 seed := by
  rw [root1Tail66, scan, if_neg (by decide), root1Candidates66_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail67
    (if 3 ≤ root1Candidates66.length then search edge 3 [7, 5437] root1Candidates66 5444 seed else seed) = _
  rw [if_pos (by decide), root1Child66]
def root1Candidates67 : List Nat := [17191, 17659]
set_option maxRecDepth 8192 in
theorem root1Candidates67_checked : candidates edge 4 7 5521 root1Tail68 seed.bound = root1Candidates67 := by
  unfold candidates
  have row : edge 5521 = (SortedNatTree.contains FastRows.row5521) := by
    change Data.row5521.contains = (SortedNatTree.contains FastRows.row5521)
    exact FastRows.row5521_checked.symm
  rw [row]
  decide
theorem root1Step67 : scan edge (search edge 3) 4 [7] 7 root1Tail67 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail68 seed := by
  rw [root1Tail67, scan, if_neg (by decide), root1Candidates67_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail68
    (if 3 ≤ root1Candidates67.length then search edge 3 [7, 5521] root1Candidates67 5528 seed else seed) = _
  rw [if_neg (by decide)]
def root1Candidates68 : List Nat := [20023]
set_option maxRecDepth 8192 in
theorem root1Candidates68_checked : candidates edge 4 7 5641 root1Tail69 seed.bound = root1Candidates68 := by
  unfold candidates
  have row : edge 5641 = (SortedNatTree.contains FastRows.row5641) := by
    change Data.row5641.contains = (SortedNatTree.contains FastRows.row5641)
    exact FastRows.row5641_checked.symm
  rw [row]
  decide
theorem root1Step68 : scan edge (search edge 3) 4 [7] 7 root1Tail68 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail69 seed := by
  rw [root1Tail68, scan, if_neg (by decide), root1Candidates68_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail69
    (if 3 ≤ root1Candidates68.length then search edge 3 [7, 5641] root1Candidates68 5648 seed else seed) = _
  rw [if_neg (by decide)]
def root1Candidates69 : List Nat := [14551, 14851, 18043]
set_option maxRecDepth 8192 in
theorem root1Candidates69_checked : candidates edge 4 7 5659 root1Tail70 seed.bound = root1Candidates69 := by
  unfold candidates
  have row : edge 5659 = (SortedNatTree.contains FastRows.row5659) := by
    change Data.row5659.contains = (SortedNatTree.contains FastRows.row5659)
    exact FastRows.row5659_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child69 : search edge 3 [7, 5659] root1Candidates69 5666 seed = seed := by decide
theorem root1Step69 : scan edge (search edge 3) 4 [7] 7 root1Tail69 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail70 seed := by
  rw [root1Tail69, scan, if_neg (by decide), root1Candidates69_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail70
    (if 3 ≤ root1Candidates69.length then search edge 3 [7, 5659] root1Candidates69 5666 seed else seed) = _
  rw [if_pos (by decide), root1Child69]
def root1Candidates70 : List Nat := [6163, 15643, 18457, 19333]
set_option maxRecDepth 8192 in
theorem root1Candidates70_checked : candidates edge 4 7 5689 root1Tail71 seed.bound = root1Candidates70 := by
  unfold candidates
  have row : edge 5689 = (SortedNatTree.contains FastRows.row5689) := by
    change Data.row5689.contains = (SortedNatTree.contains FastRows.row5689)
    exact FastRows.row5689_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child70 : search edge 3 [7, 5689] root1Candidates70 5696 seed = seed := by decide
theorem root1Step70 : scan edge (search edge 3) 4 [7] 7 root1Tail70 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail71 seed := by
  rw [root1Tail70, scan, if_neg (by decide), root1Candidates70_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail71
    (if 3 ≤ root1Candidates70.length then search edge 3 [7, 5689] root1Candidates70 5696 seed else seed) = _
  rw [if_pos (by decide), root1Child70]
def root1Candidates71 : List Nat := [10663, 11923, 19183]
set_option maxRecDepth 8192 in
theorem root1Candidates71_checked : candidates edge 4 7 5821 root1Tail72 seed.bound = root1Candidates71 := by
  unfold candidates
  have row : edge 5821 = (SortedNatTree.contains FastRows.row5821) := by
    change Data.row5821.contains = (SortedNatTree.contains FastRows.row5821)
    exact FastRows.row5821_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child71 : search edge 3 [7, 5821] root1Candidates71 5828 seed = seed := by decide
theorem root1Step71 : scan edge (search edge 3) 4 [7] 7 root1Tail71 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail72 seed := by
  rw [root1Tail71, scan, if_neg (by decide), root1Candidates71_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail72
    (if 3 ≤ root1Candidates71.length then search edge 3 [7, 5821] root1Candidates71 5828 seed else seed) = _
  rw [if_pos (by decide), root1Child71]
def root1Candidates72 : List Nat := [8311, 9043, 14557]
set_option maxRecDepth 8192 in
theorem root1Candidates72_checked : candidates edge 4 7 6091 root1Tail73 seed.bound = root1Candidates72 := by
  unfold candidates
  have row : edge 6091 = (SortedNatTree.contains FastRows.row6091) := by
    change Data.row6091.contains = (SortedNatTree.contains FastRows.row6091)
    exact FastRows.row6091_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child72 : search edge 3 [7, 6091] root1Candidates72 6098 seed = seed := by decide
theorem root1Step72 : scan edge (search edge 3) 4 [7] 7 root1Tail72 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail73 seed := by
  rw [root1Tail72, scan, if_neg (by decide), root1Candidates72_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail73
    (if 3 ≤ root1Candidates72.length then search edge 3 [7, 6091] root1Candidates72 6098 seed else seed) = _
  rw [if_pos (by decide), root1Child72]
def root1Candidates73 : List Nat := [8167, 9349, 10399, 10663, 11287, 13183, 16447]
set_option maxRecDepth 8192 in
theorem root1Candidates73_checked : candidates edge 4 7 6163 root1Tail74 seed.bound = root1Candidates73 := by
  unfold candidates
  have row : edge 6163 = (SortedNatTree.contains FastRows.row6163) := by
    change Data.row6163.contains = (SortedNatTree.contains FastRows.row6163)
    exact FastRows.row6163_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child73 : search edge 3 [7, 6163] root1Candidates73 6170 seed = seed := by decide
theorem root1Step73 : scan edge (search edge 3) 4 [7] 7 root1Tail73 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail74 seed := by
  rw [root1Tail73, scan, if_neg (by decide), root1Candidates73_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail74
    (if 3 ≤ root1Candidates73.length then search edge 3 [7, 6163] root1Candidates73 6170 seed else seed) = _
  rw [if_pos (by decide), root1Child73]
def root1Candidates74 : List Nat := [6949, 11299, 11923, 12511, 17491, 18457]
set_option maxRecDepth 8192 in
theorem root1Candidates74_checked : candidates edge 4 7 6421 root1Tail75 seed.bound = root1Candidates74 := by
  unfold candidates
  have row : edge 6421 = (SortedNatTree.contains FastRows.row6421) := by
    change Data.row6421.contains = (SortedNatTree.contains FastRows.row6421)
    exact FastRows.row6421_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child74 : search edge 3 [7, 6421] root1Candidates74 6428 seed = seed := by decide
theorem root1Step74 : scan edge (search edge 3) 4 [7] 7 root1Tail74 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail75 seed := by
  rw [root1Tail74, scan, if_neg (by decide), root1Candidates74_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail75
    (if 3 ≤ root1Candidates74.length then search edge 3 [7, 6421] root1Candidates74 6428 seed else seed) = _
  rw [if_pos (by decide), root1Child74]
def root1Candidates75 : List Nat := []
set_option maxRecDepth 8192 in
theorem root1Candidates75_checked : candidates edge 4 7 6481 root1Tail76 seed.bound = root1Candidates75 := by
  unfold candidates
  have row : edge 6481 = (SortedNatTree.contains FastRows.row6481) := by
    change Data.row6481.contains = (SortedNatTree.contains FastRows.row6481)
    exact FastRows.row6481_checked.symm
  rw [row]
  decide
theorem root1Step75 : scan edge (search edge 3) 4 [7] 7 root1Tail75 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail76 seed := by
  rw [root1Tail75, scan, if_neg (by decide), root1Candidates75_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail76
    (if 3 ≤ root1Candidates75.length then search edge 3 [7, 6481] root1Candidates75 6488 seed else seed) = _
  rw [if_neg (by decide)]
end ProjectEulerSolutions.PrimePairLarge.Search
