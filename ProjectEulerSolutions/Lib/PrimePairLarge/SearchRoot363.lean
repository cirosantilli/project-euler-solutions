import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root363Candidates : List Nat := Data.row2473
set_option maxRecDepth 8192 in
theorem root363Candidates_checked : candidates edge 5 0 2473 tail364 winner.bound = root363Candidates := by
  apply candidates_eq_row edge 5 0 2473 tail364 root363Candidates winner.bound tail364_ordered
  · decide
  · decide
  · funext q
    rfl
def root363Tail42 : List Nat := []
def root363Tail41 : List Nat := 22003 :: root363Tail42
def root363Tail40 : List Nat := 21961 :: root363Tail41
def root363Tail39 : List Nat := 21433 :: root363Tail40
def root363Tail38 : List Nat := 21391 :: root363Tail39
def root363Tail37 : List Nat := 20809 :: root363Tail38
def root363Tail36 : List Nat := 20509 :: root363Tail37
def root363Tail35 : List Nat := 20113 :: root363Tail36
def root363Tail34 : List Nat := 19867 :: root363Tail35
def root363Tail33 : List Nat := 19777 :: root363Tail34
def root363Tail32 : List Nat := 19381 :: root363Tail33
def root363Tail31 : List Nat := 18541 :: root363Tail32
def root363Tail30 : List Nat := 18367 :: root363Tail31
def root363Tail29 : List Nat := 17599 :: root363Tail30
def root363Tail28 : List Nat := 17359 :: root363Tail29
def root363Tail27 : List Nat := 16903 :: root363Tail28
def root363Tail26 : List Nat := 16417 :: root363Tail27
def root363Tail25 : List Nat := 15493 :: root363Tail26
def root363Tail24 : List Nat := 15373 :: root363Tail25
def root363Tail23 : List Nat := 15307 :: root363Tail24
def root363Tail22 : List Nat := 15217 :: root363Tail23
def root363Tail21 : List Nat := 14149 :: root363Tail22
def root363Tail20 : List Nat := 13903 :: root363Tail21
def root363Tail19 : List Nat := 13249 :: root363Tail20
def root363Tail18 : List Nat := 12301 :: root363Tail19
def root363Tail17 : List Nat := 11689 :: root363Tail18
def root363Tail16 : List Nat := 11131 :: root363Tail17
def root363Tail15 : List Nat := 10957 :: root363Tail16
def root363Tail14 : List Nat := 8941 :: root363Tail15
def root363Tail13 : List Nat := 8269 :: root363Tail14
def root363Tail12 : List Nat := 7417 :: root363Tail13
def root363Tail11 : List Nat := 6427 :: root363Tail12
def root363Tail10 : List Nat := 6379 :: root363Tail11
def root363Tail9 : List Nat := 5881 :: root363Tail10
def root363Tail8 : List Nat := 5437 :: root363Tail9
def root363Tail7 : List Nat := 5023 :: root363Tail8
def root363Tail6 : List Nat := 4327 :: root363Tail7
def root363Tail5 : List Nat := 3889 :: root363Tail6
def root363Tail4 : List Nat := 3343 :: root363Tail5
def root363Tail3 : List Nat := 3001 :: root363Tail4
def root363Tail2 : List Nat := 2791 :: root363Tail3
def root363Tail1 : List Nat := 2713 :: root363Tail2
def root363Tail0 : List Nat := 2647 :: root363Tail1
def root363Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root363Candidates0_checked : candidates edge 4 2473 2647 root363Tail1 winner.bound = root363Candidates0 := by
  unfold candidates
  have row : edge 2647 = (SortedNatTree.contains FastRows.row2647) := by
    change Data.row2647.contains = (SortedNatTree.contains FastRows.row2647)
    exact FastRows.row2647_checked.symm
  rw [row]
  decide
theorem root363Step0 : scan edge (search edge 3) 4 [2473] 2473 root363Tail0 winner =
    scan edge (search edge 3) 4 [2473] 2473 root363Tail1 winner := by
  rw [root363Tail0, scan, if_neg (by decide), root363Candidates0_checked]
  change scan edge (search edge 3) 4 [2473] 2473 root363Tail1
    (if 3 ≤ root363Candidates0.length then search edge 3 [2473, 2647] root363Candidates0 5120 winner else winner) = _
  rw [if_neg (by decide)]
def root363Candidates1 : List Nat := [3889]
set_option maxRecDepth 8192 in
theorem root363Candidates1_checked : candidates edge 4 2473 2713 root363Tail2 winner.bound = root363Candidates1 := by
  unfold candidates
  have row : edge 2713 = (SortedNatTree.contains FastRows.row2713) := by
    change Data.row2713.contains = (SortedNatTree.contains FastRows.row2713)
    exact FastRows.row2713_checked.symm
  rw [row]
  decide
theorem root363Step1 : scan edge (search edge 3) 4 [2473] 2473 root363Tail1 winner =
    scan edge (search edge 3) 4 [2473] 2473 root363Tail2 winner := by
  rw [root363Tail1, scan, if_neg (by decide), root363Candidates1_checked]
  change scan edge (search edge 3) 4 [2473] 2473 root363Tail2
    (if 3 ≤ root363Candidates1.length then search edge 3 [2473, 2713] root363Candidates1 5186 winner else winner) = _
  rw [if_neg (by decide)]
def root363Candidates2 : List Nat := [3001, 6379]
set_option maxRecDepth 8192 in
theorem root363Candidates2_checked : candidates edge 4 2473 2791 root363Tail3 winner.bound = root363Candidates2 := by
  unfold candidates
  have row : edge 2791 = (SortedNatTree.contains FastRows.row2791) := by
    change Data.row2791.contains = (SortedNatTree.contains FastRows.row2791)
    exact FastRows.row2791_checked.symm
  rw [row]
  decide
theorem root363Step2 : scan edge (search edge 3) 4 [2473] 2473 root363Tail2 winner =
    scan edge (search edge 3) 4 [2473] 2473 root363Tail3 winner := by
  rw [root363Tail2, scan, if_neg (by decide), root363Candidates2_checked]
  change scan edge (search edge 3) 4 [2473] 2473 root363Tail3
    (if 3 ≤ root363Candidates2.length then search edge 3 [2473, 2791] root363Candidates2 5264 winner else winner) = _
  rw [if_neg (by decide)]
def root363Candidates3 : List Nat := [12301, 18367, 19381, 19777]
set_option maxRecDepth 8192 in
theorem root363Candidates3_checked : candidates edge 4 2473 3001 root363Tail4 winner.bound = root363Candidates3 := by
  unfold candidates
  have row : edge 3001 = (SortedNatTree.contains FastRows.row3001) := by
    change Data.row3001.contains = (SortedNatTree.contains FastRows.row3001)
    exact FastRows.row3001_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root363Child3 : search edge 3 [2473, 3001] root363Candidates3 5474 winner = winner := by decide
theorem root363Step3 : scan edge (search edge 3) 4 [2473] 2473 root363Tail3 winner =
    scan edge (search edge 3) 4 [2473] 2473 root363Tail4 winner := by
  rw [root363Tail3, scan, if_neg (by decide), root363Candidates3_checked]
  change scan edge (search edge 3) 4 [2473] 2473 root363Tail4
    (if 3 ≤ root363Candidates3.length then search edge 3 [2473, 3001] root363Candidates3 5474 winner else winner) = _
  rw [if_pos (by decide), root363Child3]
def root363Candidates4 : List Nat := [6379, 15307, 16903]
set_option maxRecDepth 8192 in
theorem root363Candidates4_checked : candidates edge 4 2473 3343 root363Tail5 winner.bound = root363Candidates4 := by
  unfold candidates
  have row : edge 3343 = (SortedNatTree.contains FastRows.row3343) := by
    change Data.row3343.contains = (SortedNatTree.contains FastRows.row3343)
    exact FastRows.row3343_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root363Child4 : search edge 3 [2473, 3343] root363Candidates4 5816 winner = winner := by decide
theorem root363Step4 : scan edge (search edge 3) 4 [2473] 2473 root363Tail4 winner =
    scan edge (search edge 3) 4 [2473] 2473 root363Tail5 winner := by
  rw [root363Tail4, scan, if_neg (by decide), root363Candidates4_checked]
  change scan edge (search edge 3) 4 [2473] 2473 root363Tail5
    (if 3 ≤ root363Candidates4.length then search edge 3 [2473, 3343] root363Candidates4 5816 winner else winner) = _
  rw [if_pos (by decide), root363Child4]
def root363Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root363Candidates5_checked : candidates edge 4 2473 3889 root363Tail6 winner.bound = root363Candidates5 := by
  unfold candidates
  have row : edge 3889 = (SortedNatTree.contains FastRows.row3889) := by
    change Data.row3889.contains = (SortedNatTree.contains FastRows.row3889)
    exact FastRows.row3889_checked.symm
  rw [row]
  decide
theorem root363Step5 : scan edge (search edge 3) 4 [2473] 2473 root363Tail5 winner =
    scan edge (search edge 3) 4 [2473] 2473 root363Tail6 winner := by
  rw [root363Tail5, scan, if_neg (by decide), root363Candidates5_checked]
  change scan edge (search edge 3) 4 [2473] 2473 root363Tail6
    (if 3 ≤ root363Candidates5.length then search edge 3 [2473, 3889] root363Candidates5 6362 winner else winner) = _
  rw [if_neg (by decide)]
def root363Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root363Candidates6_checked : candidates edge 4 2473 4327 root363Tail7 winner.bound = root363Candidates6 := by
  unfold candidates
  have row : edge 4327 = (SortedNatTree.contains FastRows.row4327) := by
    change Data.row4327.contains = (SortedNatTree.contains FastRows.row4327)
    exact FastRows.row4327_checked.symm
  rw [row]
  decide
theorem root363Step6 : scan edge (search edge 3) 4 [2473] 2473 root363Tail6 winner =
    scan edge (search edge 3) 4 [2473] 2473 root363Tail7 winner := by
  rw [root363Tail6, scan, if_neg (by decide), root363Candidates6_checked]
  change scan edge (search edge 3) 4 [2473] 2473 root363Tail7
    (if 3 ≤ root363Candidates6.length then search edge 3 [2473, 4327] root363Candidates6 6800 winner else winner) = _
  rw [if_neg (by decide)]
def root363Candidates7 : List Nat := [8269, 16417]
set_option maxRecDepth 8192 in
theorem root363Candidates7_checked : candidates edge 4 2473 5023 root363Tail8 winner.bound = root363Candidates7 := by
  unfold candidates
  have row : edge 5023 = (SortedNatTree.contains FastRows.row5023) := by
    change Data.row5023.contains = (SortedNatTree.contains FastRows.row5023)
    exact FastRows.row5023_checked.symm
  rw [row]
  decide
theorem root363Step7 : scan edge (search edge 3) 4 [2473] 2473 root363Tail7 winner =
    scan edge (search edge 3) 4 [2473] 2473 root363Tail8 winner := by
  rw [root363Tail7, scan, if_neg (by decide), root363Candidates7_checked]
  change scan edge (search edge 3) 4 [2473] 2473 root363Tail8
    (if 3 ≤ root363Candidates7.length then search edge 3 [2473, 5023] root363Candidates7 7496 winner else winner) = _
  rw [if_neg (by decide)]
def root363Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root363Candidates8_checked : candidates edge 4 2473 5437 root363Tail9 winner.bound = root363Candidates8 := by
  unfold candidates
  have row : edge 5437 = (SortedNatTree.contains FastRows.row5437) := by
    change Data.row5437.contains = (SortedNatTree.contains FastRows.row5437)
    exact FastRows.row5437_checked.symm
  rw [row]
  decide
theorem root363Step8 : scan edge (search edge 3) 4 [2473] 2473 root363Tail8 winner =
    scan edge (search edge 3) 4 [2473] 2473 root363Tail9 winner := by
  rw [root363Tail8, scan, if_neg (by decide), root363Candidates8_checked]
  change scan edge (search edge 3) 4 [2473] 2473 root363Tail9
    (if 3 ≤ root363Candidates8.length then search edge 3 [2473, 5437] root363Candidates8 7910 winner else winner) = _
  rw [if_neg (by decide)]
def root363Candidates9 : List Nat := [8269]
set_option maxRecDepth 8192 in
theorem root363Candidates9_checked : candidates edge 4 2473 5881 root363Tail10 winner.bound = root363Candidates9 := by
  unfold candidates
  have row : edge 5881 = (SortedNatTree.contains FastRows.row5881) := by
    change Data.row5881.contains = (SortedNatTree.contains FastRows.row5881)
    exact FastRows.row5881_checked.symm
  rw [row]
  decide
theorem root363Step9 : scan edge (search edge 3) 4 [2473] 2473 root363Tail9 winner =
    scan edge (search edge 3) 4 [2473] 2473 root363Tail10 winner := by
  rw [root363Tail9, scan, if_neg (by decide), root363Candidates9_checked]
  change scan edge (search edge 3) 4 [2473] 2473 root363Tail10
    (if 3 ≤ root363Candidates9.length then search edge 3 [2473, 5881] root363Candidates9 8354 winner else winner) = _
  rw [if_neg (by decide)]
theorem root363Scan10 : scan edge (search edge 3) 4 [2473] 2473 root363Tail10 winner = winner := by rw [root363Tail10, scan, if_pos (by decide)]
theorem root363Scan9 : scan edge (search edge 3) 4 [2473] 2473 root363Tail9 winner = winner :=
  (root363Step9).trans root363Scan10
theorem root363Scan8 : scan edge (search edge 3) 4 [2473] 2473 root363Tail8 winner = winner :=
  (root363Step8).trans root363Scan9
theorem root363Scan7 : scan edge (search edge 3) 4 [2473] 2473 root363Tail7 winner = winner :=
  (root363Step7).trans root363Scan8
theorem root363Scan6 : scan edge (search edge 3) 4 [2473] 2473 root363Tail6 winner = winner :=
  (root363Step6).trans root363Scan7
theorem root363Scan5 : scan edge (search edge 3) 4 [2473] 2473 root363Tail5 winner = winner :=
  (root363Step5).trans root363Scan6
theorem root363Scan4 : scan edge (search edge 3) 4 [2473] 2473 root363Tail4 winner = winner :=
  (root363Step4).trans root363Scan5
theorem root363Scan3 : scan edge (search edge 3) 4 [2473] 2473 root363Tail3 winner = winner :=
  (root363Step3).trans root363Scan4
theorem root363Scan2 : scan edge (search edge 3) 4 [2473] 2473 root363Tail2 winner = winner :=
  (root363Step2).trans root363Scan3
theorem root363Scan1 : scan edge (search edge 3) 4 [2473] 2473 root363Tail1 winner = winner :=
  (root363Step1).trans root363Scan2
theorem root363Scan0 : scan edge (search edge 3) 4 [2473] 2473 root363Tail0 winner = winner :=
  (root363Step0).trans root363Scan1
set_option maxRecDepth 4096 in
theorem root363Branch : search edge 4 [2473] root363Candidates 2473 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2473] 2473 root363Tail0 winner = winner
  exact root363Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
