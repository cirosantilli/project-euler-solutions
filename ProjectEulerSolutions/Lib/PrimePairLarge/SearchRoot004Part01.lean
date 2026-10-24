import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot004Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root4Candidates32 : List Nat := [20231]
set_option maxRecDepth 8192 in
theorem root4Candidates32_checked : candidates edge 4 17 4241 root4Tail33 winner.bound = root4Candidates32 := by
  unfold candidates
  have row : edge 4241 = (SortedNatTree.contains FastRows.row4241) := by
    change Data.row4241.contains = (SortedNatTree.contains FastRows.row4241)
    exact FastRows.row4241_checked.symm
  rw [row]
  decide
theorem root4Step32 : scan edge (search edge 3) 4 [17] 17 root4Tail32 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail33 winner := by
  rw [root4Tail32, scan, if_neg (by decide), root4Candidates32_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail33
    (if 3 ≤ root4Candidates32.length then search edge 3 [17, 4241] root4Candidates32 4258 winner else winner) = _
  rw [if_neg (by decide)]
def root4Candidates33 : List Nat := [4637, 5003, 10613, 20231]
set_option maxRecDepth 8192 in
theorem root4Candidates33_checked : candidates edge 4 17 4583 root4Tail34 winner.bound = root4Candidates33 := by
  unfold candidates
  have row : edge 4583 = (SortedNatTree.contains FastRows.row4583) := by
    change Data.row4583.contains = (SortedNatTree.contains FastRows.row4583)
    exact FastRows.row4583_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root4Child33 : search edge 3 [17, 4583] root4Candidates33 4600 winner = winner := by decide
theorem root4Step33 : scan edge (search edge 3) 4 [17] 17 root4Tail33 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail34 winner := by
  rw [root4Tail33, scan, if_neg (by decide), root4Candidates33_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail34
    (if 3 ≤ root4Candidates33.length then search edge 3 [17, 4583] root4Candidates33 4600 winner else winner) = _
  rw [if_pos (by decide), root4Child33]
def root4Candidates34 : List Nat := [9041, 14759, 16217]
set_option maxRecDepth 8192 in
theorem root4Candidates34_checked : candidates edge 4 17 4637 root4Tail35 winner.bound = root4Candidates34 := by
  unfold candidates
  have row : edge 4637 = (SortedNatTree.contains FastRows.row4637) := by
    change Data.row4637.contains = (SortedNatTree.contains FastRows.row4637)
    exact FastRows.row4637_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root4Child34 : search edge 3 [17, 4637] root4Candidates34 4654 winner = winner := by decide
theorem root4Step34 : scan edge (search edge 3) 4 [17] 17 root4Tail34 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail35 winner := by
  rw [root4Tail34, scan, if_neg (by decide), root4Candidates34_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail35
    (if 3 ≤ root4Candidates34.length then search edge 3 [17, 4637] root4Candidates34 4654 winner else winner) = _
  rw [if_pos (by decide), root4Child34]
def root4Candidates35 : List Nat := [8537, 18719, 19583, 20297]
set_option maxRecDepth 8192 in
theorem root4Candidates35_checked : candidates edge 4 17 4649 root4Tail36 winner.bound = root4Candidates35 := by
  unfold candidates
  have row : edge 4649 = (SortedNatTree.contains FastRows.row4649) := by
    change Data.row4649.contains = (SortedNatTree.contains FastRows.row4649)
    exact FastRows.row4649_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root4Child35 : search edge 3 [17, 4649] root4Candidates35 4666 winner = winner := by decide
theorem root4Step35 : scan edge (search edge 3) 4 [17] 17 root4Tail35 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail36 winner := by
  rw [root4Tail35, scan, if_neg (by decide), root4Candidates35_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail36
    (if 3 ≤ root4Candidates35.length then search edge 3 [17, 4649] root4Candidates35 4666 winner else winner) = _
  rw [if_pos (by decide), root4Child35]
def root4Candidates36 : List Nat := [5519, 10937, 14957, 20639]
set_option maxRecDepth 8192 in
theorem root4Candidates36_checked : candidates edge 4 17 4673 root4Tail37 winner.bound = root4Candidates36 := by
  unfold candidates
  have row : edge 4673 = (SortedNatTree.contains FastRows.row4673) := by
    change Data.row4673.contains = (SortedNatTree.contains FastRows.row4673)
    exact FastRows.row4673_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root4Child36 : search edge 3 [17, 4673] root4Candidates36 4690 winner = winner := by decide
theorem root4Step36 : scan edge (search edge 3) 4 [17] 17 root4Tail36 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail37 winner := by
  rw [root4Tail36, scan, if_neg (by decide), root4Candidates36_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail37
    (if 3 ≤ root4Candidates36.length then search edge 3 [17, 4673] root4Candidates36 4690 winner else winner) = _
  rw [if_pos (by decide), root4Child36]
def root4Candidates37 : List Nat := [9743, 17609, 18869]
set_option maxRecDepth 8192 in
theorem root4Candidates37_checked : candidates edge 4 17 4703 root4Tail38 winner.bound = root4Candidates37 := by
  unfold candidates
  have row : edge 4703 = (SortedNatTree.contains FastRows.row4703) := by
    change Data.row4703.contains = (SortedNatTree.contains FastRows.row4703)
    exact FastRows.row4703_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root4Child37 : search edge 3 [17, 4703] root4Candidates37 4720 winner = winner := by decide
theorem root4Step37 : scan edge (search edge 3) 4 [17] 17 root4Tail37 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail38 winner := by
  rw [root4Tail37, scan, if_neg (by decide), root4Candidates37_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail38
    (if 3 ≤ root4Candidates37.length then search edge 3 [17, 4703] root4Candidates37 4720 winner else winner) = _
  rw [if_pos (by decide), root4Child37]
def root4Candidates38 : List Nat := [8231]
set_option maxRecDepth 8192 in
theorem root4Candidates38_checked : candidates edge 4 17 4877 root4Tail39 winner.bound = root4Candidates38 := by
  unfold candidates
  have row : edge 4877 = (SortedNatTree.contains FastRows.row4877) := by
    change Data.row4877.contains = (SortedNatTree.contains FastRows.row4877)
    exact FastRows.row4877_checked.symm
  rw [row]
  decide
theorem root4Step38 : scan edge (search edge 3) 4 [17] 17 root4Tail38 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail39 winner := by
  rw [root4Tail38, scan, if_neg (by decide), root4Candidates38_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail39
    (if 3 ≤ root4Candidates38.length then search edge 3 [17, 4877] root4Candidates38 4894 winner else winner) = _
  rw [if_neg (by decide)]
def root4Candidates39 : List Nat := [5261, 8819, 13763, 20297]
set_option maxRecDepth 8192 in
theorem root4Candidates39_checked : candidates edge 4 17 4943 root4Tail40 winner.bound = root4Candidates39 := by
  unfold candidates
  have row : edge 4943 = (SortedNatTree.contains FastRows.row4943) := by
    change Data.row4943.contains = (SortedNatTree.contains FastRows.row4943)
    exact FastRows.row4943_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root4Child39 : search edge 3 [17, 4943] root4Candidates39 4960 winner = winner := by decide
theorem root4Step39 : scan edge (search edge 3) 4 [17] 17 root4Tail39 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail40 winner := by
  rw [root4Tail39, scan, if_neg (by decide), root4Candidates39_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail40
    (if 3 ≤ root4Candidates39.length then search edge 3 [17, 4943] root4Candidates39 4960 winner else winner) = _
  rw [if_pos (by decide), root4Child39]
def root4Candidates40 : List Nat := []
set_option maxRecDepth 8192 in
theorem root4Candidates40_checked : candidates edge 4 17 5003 root4Tail41 winner.bound = root4Candidates40 := by
  unfold candidates
  have row : edge 5003 = (SortedNatTree.contains FastRows.row5003) := by
    change Data.row5003.contains = (SortedNatTree.contains FastRows.row5003)
    exact FastRows.row5003_checked.symm
  rw [row]
  decide
theorem root4Step40 : scan edge (search edge 3) 4 [17] 17 root4Tail40 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail41 winner := by
  rw [root4Tail40, scan, if_neg (by decide), root4Candidates40_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail41
    (if 3 ≤ root4Candidates40.length then search edge 3 [17, 5003] root4Candidates40 5020 winner else winner) = _
  rw [if_neg (by decide)]
def root4Candidates41 : List Nat := [8831, 9689, 16427, 18869]
set_option maxRecDepth 8192 in
theorem root4Candidates41_checked : candidates edge 4 17 5261 root4Tail42 winner.bound = root4Candidates41 := by
  unfold candidates
  have row : edge 5261 = (SortedNatTree.contains FastRows.row5261) := by
    change Data.row5261.contains = (SortedNatTree.contains FastRows.row5261)
    exact FastRows.row5261_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root4Child41 : search edge 3 [17, 5261] root4Candidates41 5278 winner = winner := by decide
theorem root4Step41 : scan edge (search edge 3) 4 [17] 17 root4Tail41 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail42 winner := by
  rw [root4Tail41, scan, if_neg (by decide), root4Candidates41_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail42
    (if 3 ≤ root4Candidates41.length then search edge 3 [17, 5261] root4Candidates41 5278 winner else winner) = _
  rw [if_pos (by decide), root4Child41]
def root4Candidates42 : List Nat := [7283, 15737, 20639]
set_option maxRecDepth 8192 in
theorem root4Candidates42_checked : candidates edge 4 17 5333 root4Tail43 winner.bound = root4Candidates42 := by
  unfold candidates
  have row : edge 5333 = (SortedNatTree.contains FastRows.row5333) := by
    change Data.row5333.contains = (SortedNatTree.contains FastRows.row5333)
    exact FastRows.row5333_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root4Child42 : search edge 3 [17, 5333] root4Candidates42 5350 winner = winner := by decide
theorem root4Step42 : scan edge (search edge 3) 4 [17] 17 root4Tail42 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail43 winner := by
  rw [root4Tail42, scan, if_neg (by decide), root4Candidates42_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail43
    (if 3 ≤ root4Candidates42.length then search edge 3 [17, 5333] root4Candidates42 5350 winner else winner) = _
  rw [if_pos (by decide), root4Child42]
def root4Candidates43 : List Nat := [8753, 17609, 20297]
set_option maxRecDepth 8192 in
theorem root4Candidates43_checked : candidates edge 4 17 5519 root4Tail44 winner.bound = root4Candidates43 := by
  unfold candidates
  have row : edge 5519 = (SortedNatTree.contains FastRows.row5519) := by
    change Data.row5519.contains = (SortedNatTree.contains FastRows.row5519)
    exact FastRows.row5519_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root4Child43 : search edge 3 [17, 5519] root4Candidates43 5536 winner = winner := by decide
theorem root4Step43 : scan edge (search edge 3) 4 [17] 17 root4Tail43 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail44 winner := by
  rw [root4Tail43, scan, if_neg (by decide), root4Candidates43_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail44
    (if 3 ≤ root4Candidates43.length then search edge 3 [17, 5519] root4Candidates43 5536 winner else winner) = _
  rw [if_pos (by decide), root4Child43]
def root4Candidates44 : List Nat := [9689, 13763, 16787]
set_option maxRecDepth 8192 in
theorem root4Candidates44_checked : candidates edge 4 17 5783 root4Tail45 winner.bound = root4Candidates44 := by
  unfold candidates
  have row : edge 5783 = (SortedNatTree.contains FastRows.row5783) := by
    change Data.row5783.contains = (SortedNatTree.contains FastRows.row5783)
    exact FastRows.row5783_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root4Child44 : search edge 3 [17, 5783] root4Candidates44 5800 winner = winner := by decide
theorem root4Step44 : scan edge (search edge 3) 4 [17] 17 root4Tail44 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail45 winner := by
  rw [root4Tail44, scan, if_neg (by decide), root4Candidates44_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail45
    (if 3 ≤ root4Candidates44.length then search edge 3 [17, 5783] root4Candidates44 5800 winner else winner) = _
  rw [if_pos (by decide), root4Child44]
def root4Candidates45 : List Nat := [8537, 9923, 10937, 12197, 19841]
set_option maxRecDepth 8192 in
theorem root4Candidates45_checked : candidates edge 4 17 5897 root4Tail46 winner.bound = root4Candidates45 := by
  unfold candidates
  have row : edge 5897 = (SortedNatTree.contains FastRows.row5897) := by
    change Data.row5897.contains = (SortedNatTree.contains FastRows.row5897)
    exact FastRows.row5897_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root4Child45 : search edge 3 [17, 5897] root4Candidates45 5914 winner = winner := by decide
theorem root4Step45 : scan edge (search edge 3) 4 [17] 17 root4Tail45 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail46 winner := by
  rw [root4Tail45, scan, if_neg (by decide), root4Candidates45_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail46
    (if 3 ≤ root4Candidates45.length then search edge 3 [17, 5897] root4Candidates45 5914 winner else winner) = _
  rw [if_pos (by decide), root4Child45]
def root4Candidates46 : List Nat := [6599, 14423]
set_option maxRecDepth 8192 in
theorem root4Candidates46_checked : candidates edge 4 17 6317 root4Tail47 winner.bound = root4Candidates46 := by
  unfold candidates
  have row : edge 6317 = (SortedNatTree.contains FastRows.row6317) := by
    change Data.row6317.contains = (SortedNatTree.contains FastRows.row6317)
    exact FastRows.row6317_checked.symm
  rw [row]
  decide
theorem root4Step46 : scan edge (search edge 3) 4 [17] 17 root4Tail46 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail47 winner := by
  rw [root4Tail46, scan, if_neg (by decide), root4Candidates46_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail47
    (if 3 ≤ root4Candidates46.length then search edge 3 [17, 6317] root4Candidates46 6334 winner else winner) = _
  rw [if_neg (by decide)]
def root4Candidates47 : List Nat := [9923, 18701]
set_option maxRecDepth 8192 in
theorem root4Candidates47_checked : candidates edge 4 17 6353 root4Tail48 winner.bound = root4Candidates47 := by
  unfold candidates
  have row : edge 6353 = (SortedNatTree.contains FastRows.row6353) := by
    change Data.row6353.contains = (SortedNatTree.contains FastRows.row6353)
    exact FastRows.row6353_checked.symm
  rw [row]
  decide
theorem root4Step47 : scan edge (search edge 3) 4 [17] 17 root4Tail47 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail48 winner := by
  rw [root4Tail47, scan, if_neg (by decide), root4Candidates47_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail48
    (if 3 ≤ root4Candidates47.length then search edge 3 [17, 6353] root4Candidates47 6370 winner else winner) = _
  rw [if_neg (by decide)]
end ProjectEulerSolutions.PrimePairLarge.Search
