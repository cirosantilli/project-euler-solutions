import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot000Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root0Candidates96 : List Nat := [13381]
set_option maxRecDepth 8192 in
theorem root0Candidates96_checked : candidates edge 4 3 6037 root0Tail97 seed.bound = root0Candidates96 := by
  unfold candidates
  have row : edge 6037 = (SortedNatTree.contains FastRows.row6037) := by
    change Data.row6037.contains = (SortedNatTree.contains FastRows.row6037)
    exact FastRows.row6037_checked.symm
  rw [row]
  decide
theorem root0Step96 : scan edge (search edge 3) 4 [3] 3 root0Tail96 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail97 seed := by
  rw [root0Tail96, scan, if_neg (by decide), root0Candidates96_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail97
    (if 3 ≤ root0Candidates96.length then search edge 3 [3, 6037] root0Candidates96 6040 seed else seed) = _
  rw [if_neg (by decide)]
def root0Candidates97 : List Nat := [17959, 19387]
set_option maxRecDepth 8192 in
theorem root0Candidates97_checked : candidates edge 4 3 6073 root0Tail98 seed.bound = root0Candidates97 := by
  unfold candidates
  have row : edge 6073 = (SortedNatTree.contains FastRows.row6073) := by
    change Data.row6073.contains = (SortedNatTree.contains FastRows.row6073)
    exact FastRows.row6073_checked.symm
  rw [row]
  decide
theorem root0Step97 : scan edge (search edge 3) 4 [3] 3 root0Tail97 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail98 seed := by
  rw [root0Tail97, scan, if_neg (by decide), root0Candidates97_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail98
    (if 3 ≤ root0Candidates97.length then search edge 3 [3, 6073] root0Candidates97 6076 seed else seed) = _
  rw [if_neg (by decide)]
def root0Candidates98 : List Nat := [11177]
set_option maxRecDepth 8192 in
theorem root0Candidates98_checked : candidates edge 4 3 6263 root0Tail99 seed.bound = root0Candidates98 := by
  unfold candidates
  have row : edge 6263 = (SortedNatTree.contains FastRows.row6263) := by
    change Data.row6263.contains = (SortedNatTree.contains FastRows.row6263)
    exact FastRows.row6263_checked.symm
  rw [row]
  decide
theorem root0Step98 : scan edge (search edge 3) 4 [3] 3 root0Tail98 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail99 seed := by
  rw [root0Tail98, scan, if_neg (by decide), root0Candidates98_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail99
    (if 3 ≤ root0Candidates98.length then search edge 3 [3, 6263] root0Candidates98 6266 seed else seed) = _
  rw [if_neg (by decide)]
def root0Candidates99 : List Nat := [12451, 14407, 15493, 19699]
set_option maxRecDepth 8192 in
theorem root0Candidates99_checked : candidates edge 4 3 6277 root0Tail100 seed.bound = root0Candidates99 := by
  unfold candidates
  have row : edge 6277 = (SortedNatTree.contains FastRows.row6277) := by
    change Data.row6277.contains = (SortedNatTree.contains FastRows.row6277)
    exact FastRows.row6277_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root0Child99 : search edge 3 [3, 6277] root0Candidates99 6280 seed = seed := by decide
theorem root0Step99 : scan edge (search edge 3) 4 [3] 3 root0Tail99 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail100 seed := by
  rw [root0Tail99, scan, if_neg (by decide), root0Candidates99_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail100
    (if 3 ≤ root0Candidates99.length then search edge 3 [3, 6277] root0Candidates99 6280 seed else seed) = _
  rw [if_pos (by decide), root0Child99]
def root0Candidates100 : List Nat := [9887, 17327, 19031]
set_option maxRecDepth 8192 in
theorem root0Candidates100_checked : candidates edge 4 3 6353 root0Tail101 seed.bound = root0Candidates100 := by
  unfold candidates
  have row : edge 6353 = (SortedNatTree.contains FastRows.row6353) := by
    change Data.row6353.contains = (SortedNatTree.contains FastRows.row6353)
    exact FastRows.row6353_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root0Child100 : search edge 3 [3, 6353] root0Candidates100 6356 seed = seed := by decide
theorem root0Step100 : scan edge (search edge 3) 4 [3] 3 root0Tail100 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail101 seed := by
  rw [root0Tail100, scan, if_neg (by decide), root0Candidates100_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail101
    (if 3 ≤ root0Candidates100.length then search edge 3 [3, 6353] root0Candidates100 6356 seed else seed) = _
  rw [if_pos (by decide), root0Child100]
def root0Candidates101 : List Nat := [9241, 11197, 19183]
set_option maxRecDepth 8192 in
theorem root0Candidates101_checked : candidates edge 4 3 6451 root0Tail102 seed.bound = root0Candidates101 := by
  unfold candidates
  have row : edge 6451 = (SortedNatTree.contains FastRows.row6451) := by
    change Data.row6451.contains = (SortedNatTree.contains FastRows.row6451)
    exact FastRows.row6451_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root0Child101 : search edge 3 [3, 6451] root0Candidates101 6454 seed = seed := by decide
theorem root0Step101 : scan edge (search edge 3) 4 [3] 3 root0Tail101 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail102 seed := by
  rw [root0Tail101, scan, if_neg (by decide), root0Candidates101_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail102
    (if 3 ≤ root0Candidates101.length then search edge 3 [3, 6451] root0Candidates101 6454 seed else seed) = _
  rw [if_pos (by decide), root0Child101]
def root0Candidates102 : List Nat := [9241, 11827, 17077]
set_option maxRecDepth 8192 in
theorem root0Candidates102_checked : candidates edge 4 3 6469 root0Tail103 seed.bound = root0Candidates102 := by
  unfold candidates
  have row : edge 6469 = (SortedNatTree.contains FastRows.row6469) := by
    change Data.row6469.contains = (SortedNatTree.contains FastRows.row6469)
    exact FastRows.row6469_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root0Child102 : search edge 3 [3, 6469] root0Candidates102 6472 seed = seed := by decide
theorem root0Step102 : scan edge (search edge 3) 4 [3] 3 root0Tail102 seed =
    scan edge (search edge 3) 4 [3] 3 root0Tail103 seed := by
  rw [root0Tail102, scan, if_neg (by decide), root0Candidates102_checked]
  change scan edge (search edge 3) 4 [3] 3 root0Tail103
    (if 3 ≤ root0Candidates102.length then search edge 3 [3, 6469] root0Candidates102 6472 seed else seed) = _
  rw [if_pos (by decide), root0Child102]
end ProjectEulerSolutions.PrimePairLarge.Search
