import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot013Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root13Candidates32 : List Nat := [10343, 16901, 19421]
set_option maxRecDepth 8192 in
theorem root13Candidates32_checked : candidates edge 4 53 4283 root13Tail33 winner.bound = root13Candidates32 := by
  unfold candidates
  have row : edge 4283 = (SortedNatTree.contains FastRows.row4283) := by
    change Data.row4283.contains = (SortedNatTree.contains FastRows.row4283)
    exact FastRows.row4283_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root13Child32 : search edge 3 [53, 4283] root13Candidates32 4336 winner = winner := by decide
theorem root13Step32 : scan edge (search edge 3) 4 [53] 53 root13Tail32 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail33 winner := by
  rw [root13Tail32, scan, if_neg (by decide), root13Candidates32_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail33
    (if 3 ≤ root13Candidates32.length then search edge 3 [53, 4283] root13Candidates32 4336 winner else winner) = _
  rw [if_pos (by decide), root13Child32]
def root13Candidates33 : List Nat := []
set_option maxRecDepth 8192 in
theorem root13Candidates33_checked : candidates edge 4 53 4637 root13Tail34 winner.bound = root13Candidates33 := by
  unfold candidates
  have row : edge 4637 = (SortedNatTree.contains FastRows.row4637) := by
    change Data.row4637.contains = (SortedNatTree.contains FastRows.row4637)
    exact FastRows.row4637_checked.symm
  rw [row]
  decide
theorem root13Step33 : scan edge (search edge 3) 4 [53] 53 root13Tail33 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail34 winner := by
  rw [root13Tail33, scan, if_neg (by decide), root13Candidates33_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail34
    (if 3 ≤ root13Candidates33.length then search edge 3 [53, 4637] root13Candidates33 4690 winner else winner) = _
  rw [if_neg (by decide)]
def root13Candidates34 : List Nat := [6323, 7853, 8117, 14207, 19421]
set_option maxRecDepth 8192 in
theorem root13Candidates34_checked : candidates edge 4 53 4649 root13Tail35 winner.bound = root13Candidates34 := by
  unfold candidates
  have row : edge 4649 = (SortedNatTree.contains FastRows.row4649) := by
    change Data.row4649.contains = (SortedNatTree.contains FastRows.row4649)
    exact FastRows.row4649_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root13Child34 : search edge 3 [53, 4649] root13Candidates34 4702 winner = winner := by decide
theorem root13Step34 : scan edge (search edge 3) 4 [53] 53 root13Tail34 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail35 winner := by
  rw [root13Tail34, scan, if_neg (by decide), root13Candidates34_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail35
    (if 3 ≤ root13Candidates34.length then search edge 3 [53, 4649] root13Candidates34 4702 winner else winner) = _
  rw [if_pos (by decide), root13Child34]
def root13Candidates35 : List Nat := [19421]
set_option maxRecDepth 8192 in
theorem root13Candidates35_checked : candidates edge 4 53 4799 root13Tail36 winner.bound = root13Candidates35 := by
  unfold candidates
  have row : edge 4799 = (SortedNatTree.contains FastRows.row4799) := by
    change Data.row4799.contains = (SortedNatTree.contains FastRows.row4799)
    exact FastRows.row4799_checked.symm
  rw [row]
  decide
theorem root13Step35 : scan edge (search edge 3) 4 [53] 53 root13Tail35 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail36 winner := by
  rw [root13Tail35, scan, if_neg (by decide), root13Candidates35_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail36
    (if 3 ≤ root13Candidates35.length then search edge 3 [53, 4799] root13Candidates35 4852 winner else winner) = _
  rw [if_neg (by decide)]
def root13Candidates36 : List Nat := [18749]
set_option maxRecDepth 8192 in
theorem root13Candidates36_checked : candidates edge 4 53 4943 root13Tail37 winner.bound = root13Candidates36 := by
  unfold candidates
  have row : edge 4943 = (SortedNatTree.contains FastRows.row4943) := by
    change Data.row4943.contains = (SortedNatTree.contains FastRows.row4943)
    exact FastRows.row4943_checked.symm
  rw [row]
  decide
theorem root13Step36 : scan edge (search edge 3) 4 [53] 53 root13Tail36 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail37 winner := by
  rw [root13Tail36, scan, if_neg (by decide), root13Candidates36_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail37
    (if 3 ≤ root13Candidates36.length then search edge 3 [53, 4943] root13Candidates36 4996 winner else winner) = _
  rw [if_neg (by decide)]
def root13Candidates37 : List Nat := []
set_option maxRecDepth 8192 in
theorem root13Candidates37_checked : candidates edge 4 53 5273 root13Tail38 winner.bound = root13Candidates37 := by
  unfold candidates
  have row : edge 5273 = (SortedNatTree.contains FastRows.row5273) := by
    change Data.row5273.contains = (SortedNatTree.contains FastRows.row5273)
    exact FastRows.row5273_checked.symm
  rw [row]
  decide
theorem root13Step37 : scan edge (search edge 3) 4 [53] 53 root13Tail37 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail38 winner := by
  rw [root13Tail37, scan, if_neg (by decide), root13Candidates37_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail38
    (if 3 ≤ root13Candidates37.length then search edge 3 [53, 5273] root13Candidates37 5326 winner else winner) = _
  rw [if_neg (by decide)]
def root13Candidates38 : List Nat := [8123, 10061]
set_option maxRecDepth 8192 in
theorem root13Candidates38_checked : candidates edge 4 53 5303 root13Tail39 winner.bound = root13Candidates38 := by
  unfold candidates
  have row : edge 5303 = (SortedNatTree.contains FastRows.row5303) := by
    change Data.row5303.contains = (SortedNatTree.contains FastRows.row5303)
    exact FastRows.row5303_checked.symm
  rw [row]
  decide
theorem root13Step38 : scan edge (search edge 3) 4 [53] 53 root13Tail38 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail39 winner := by
  rw [root13Tail38, scan, if_neg (by decide), root13Candidates38_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail39
    (if 3 ≤ root13Candidates38.length then search edge 3 [53, 5303] root13Candidates38 5356 winner else winner) = _
  rw [if_neg (by decide)]
def root13Candidates39 : List Nat := [7703]
set_option maxRecDepth 8192 in
theorem root13Candidates39_checked : candidates edge 4 53 5333 root13Tail40 winner.bound = root13Candidates39 := by
  unfold candidates
  have row : edge 5333 = (SortedNatTree.contains FastRows.row5333) := by
    change Data.row5333.contains = (SortedNatTree.contains FastRows.row5333)
    exact FastRows.row5333_checked.symm
  rw [row]
  decide
theorem root13Step39 : scan edge (search edge 3) 4 [53] 53 root13Tail39 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail40 winner := by
  rw [root13Tail39, scan, if_neg (by decide), root13Candidates39_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail40
    (if 3 ≤ root13Candidates39.length then search edge 3 [53, 5333] root13Candidates39 5386 winner else winner) = _
  rw [if_neg (by decide)]
def root13Candidates40 : List Nat := [17903]
set_option maxRecDepth 8192 in
theorem root13Candidates40_checked : candidates edge 4 53 5783 root13Tail41 winner.bound = root13Candidates40 := by
  unfold candidates
  have row : edge 5783 = (SortedNatTree.contains FastRows.row5783) := by
    change Data.row5783.contains = (SortedNatTree.contains FastRows.row5783)
    exact FastRows.row5783_checked.symm
  rw [row]
  decide
theorem root13Step40 : scan edge (search edge 3) 4 [53] 53 root13Tail40 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail41 winner := by
  rw [root13Tail40, scan, if_neg (by decide), root13Candidates40_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail41
    (if 3 ≤ root13Candidates40.length then search edge 3 [53, 5783] root13Candidates40 5836 winner else winner) = _
  rw [if_neg (by decide)]
def root13Candidates41 : List Nat := []
set_option maxRecDepth 8192 in
theorem root13Candidates41_checked : candidates edge 4 53 5861 root13Tail42 winner.bound = root13Candidates41 := by
  unfold candidates
  have row : edge 5861 = (SortedNatTree.contains FastRows.row5861) := by
    change Data.row5861.contains = (SortedNatTree.contains FastRows.row5861)
    exact FastRows.row5861_checked.symm
  rw [row]
  decide
theorem root13Step41 : scan edge (search edge 3) 4 [53] 53 root13Tail41 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail42 winner := by
  rw [root13Tail41, scan, if_neg (by decide), root13Candidates41_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail42
    (if 3 ≤ root13Candidates41.length then search edge 3 [53, 5861] root13Candidates41 5914 winner else winner) = _
  rw [if_neg (by decide)]
def root13Candidates42 : List Nat := []
set_option maxRecDepth 8192 in
theorem root13Candidates42_checked : candidates edge 4 53 6287 root13Tail43 winner.bound = root13Candidates42 := by
  unfold candidates
  have row : edge 6287 = (SortedNatTree.contains FastRows.row6287) := by
    change Data.row6287.contains = (SortedNatTree.contains FastRows.row6287)
    exact FastRows.row6287_checked.symm
  rw [row]
  decide
theorem root13Step42 : scan edge (search edge 3) 4 [53] 53 root13Tail42 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail43 winner := by
  rw [root13Tail42, scan, if_neg (by decide), root13Candidates42_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail43
    (if 3 ≤ root13Candidates42.length then search edge 3 [53, 6287] root13Candidates42 6340 winner else winner) = _
  rw [if_neg (by decide)]
def root13Candidates43 : List Nat := [8423, 11909]
set_option maxRecDepth 8192 in
theorem root13Candidates43_checked : candidates edge 4 53 6311 root13Tail44 winner.bound = root13Candidates43 := by
  unfold candidates
  have row : edge 6311 = (SortedNatTree.contains FastRows.row6311) := by
    change Data.row6311.contains = (SortedNatTree.contains FastRows.row6311)
    exact FastRows.row6311_checked.symm
  rw [row]
  decide
theorem root13Step43 : scan edge (search edge 3) 4 [53] 53 root13Tail43 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail44 winner := by
  rw [root13Tail43, scan, if_neg (by decide), root13Candidates43_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail44
    (if 3 ≤ root13Candidates43.length then search edge 3 [53, 6311] root13Candidates43 6364 winner else winner) = _
  rw [if_neg (by decide)]
def root13Candidates44 : List Nat := [9311, 10271]
set_option maxRecDepth 8192 in
theorem root13Candidates44_checked : candidates edge 4 53 6323 root13Tail45 winner.bound = root13Candidates44 := by
  unfold candidates
  have row : edge 6323 = (SortedNatTree.contains FastRows.row6323) := by
    change Data.row6323.contains = (SortedNatTree.contains FastRows.row6323)
    exact FastRows.row6323_checked.symm
  rw [row]
  decide
theorem root13Step44 : scan edge (search edge 3) 4 [53] 53 root13Tail44 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail45 winner := by
  rw [root13Tail44, scan, if_neg (by decide), root13Candidates44_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail45
    (if 3 ≤ root13Candidates44.length then search edge 3 [53, 6323] root13Candidates44 6376 winner else winner) = _
  rw [if_neg (by decide)]
def root13Candidates45 : List Nat := [12323, 12911, 16349]
set_option maxRecDepth 8192 in
theorem root13Candidates45_checked : candidates edge 4 53 6353 root13Tail46 winner.bound = root13Candidates45 := by
  unfold candidates
  have row : edge 6353 = (SortedNatTree.contains FastRows.row6353) := by
    change Data.row6353.contains = (SortedNatTree.contains FastRows.row6353)
    exact FastRows.row6353_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root13Child45 : search edge 3 [53, 6353] root13Candidates45 6406 winner = winner := by decide
theorem root13Step45 : scan edge (search edge 3) 4 [53] 53 root13Tail45 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail46 winner := by
  rw [root13Tail45, scan, if_neg (by decide), root13Candidates45_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail46
    (if 3 ≤ root13Candidates45.length then search edge 3 [53, 6353] root13Candidates45 6406 winner else winner) = _
  rw [if_pos (by decide), root13Child45]
end ProjectEulerSolutions.PrimePairLarge.Search
