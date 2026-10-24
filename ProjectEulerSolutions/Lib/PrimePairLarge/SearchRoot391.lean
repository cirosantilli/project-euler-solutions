import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root391Candidates : List Nat := Data.row2707
set_option maxRecDepth 8192 in
theorem root391Candidates_checked : candidates edge 5 0 2707 tail392 winner.bound = root391Candidates := by
  apply candidates_eq_row edge 5 0 2707 tail392 root391Candidates winner.bound tail392_ordered
  · decide
  · decide
  · funext q
    rfl
def root391Tail52 : List Nat := []
def root391Tail51 : List Nat := 22717 :: root391Tail52
def root391Tail50 : List Nat := 22381 :: root391Tail51
def root391Tail49 : List Nat := 21817 :: root391Tail50
def root391Tail48 : List Nat := 21601 :: root391Tail49
def root391Tail47 : List Nat := 21019 :: root391Tail48
def root391Tail46 : List Nat := 20743 :: root391Tail47
def root391Tail45 : List Nat := 20641 :: root391Tail46
def root391Tail44 : List Nat := 20011 :: root391Tail45
def root391Tail43 : List Nat := 19819 :: root391Tail44
def root391Tail42 : List Nat := 19609 :: root391Tail43
def root391Tail41 : List Nat := 19489 :: root391Tail42
def root391Tail40 : List Nat := 19231 :: root391Tail41
def root391Tail39 : List Nat := 18493 :: root391Tail40
def root391Tail38 : List Nat := 18313 :: root391Tail39
def root391Tail37 : List Nat := 16903 :: root391Tail38
def root391Tail36 : List Nat := 15583 :: root391Tail37
def root391Tail35 : List Nat := 15139 :: root391Tail36
def root391Tail34 : List Nat := 15073 :: root391Tail35
def root391Tail33 : List Nat := 14503 :: root391Tail34
def root391Tail32 : List Nat := 14281 :: root391Tail33
def root391Tail31 : List Nat := 14107 :: root391Tail32
def root391Tail30 : List Nat := 13249 :: root391Tail31
def root391Tail29 : List Nat := 12487 :: root391Tail30
def root391Tail28 : List Nat := 12157 :: root391Tail29
def root391Tail27 : List Nat := 12109 :: root391Tail28
def root391Tail26 : List Nat := 11443 :: root391Tail27
def root391Tail25 : List Nat := 10729 :: root391Tail26
def root391Tail24 : List Nat := 10723 :: root391Tail25
def root391Tail23 : List Nat := 10333 :: root391Tail24
def root391Tail22 : List Nat := 9973 :: root391Tail23
def root391Tail21 : List Nat := 8599 :: root391Tail22
def root391Tail20 : List Nat := 8053 :: root391Tail21
def root391Tail19 : List Nat := 7789 :: root391Tail20
def root391Tail18 : List Nat := 7759 :: root391Tail19
def root391Tail17 : List Nat := 6961 :: root391Tail18
def root391Tail16 : List Nat := 6607 :: root391Tail17
def root391Tail15 : List Nat := 6229 :: root391Tail16
def root391Tail14 : List Nat := 6079 :: root391Tail15
def root391Tail13 : List Nat := 5857 :: root391Tail14
def root391Tail12 : List Nat := 5701 :: root391Tail13
def root391Tail11 : List Nat := 5557 :: root391Tail12
def root391Tail10 : List Nat := 5413 :: root391Tail11
def root391Tail9 : List Nat := 4909 :: root391Tail10
def root391Tail8 : List Nat := 4507 :: root391Tail9
def root391Tail7 : List Nat := 4273 :: root391Tail8
def root391Tail6 : List Nat := 4261 :: root391Tail7
def root391Tail5 : List Nat := 3793 :: root391Tail6
def root391Tail4 : List Nat := 3517 :: root391Tail5
def root391Tail3 : List Nat := 3463 :: root391Tail4
def root391Tail2 : List Nat := 3253 :: root391Tail3
def root391Tail1 : List Nat := 2887 :: root391Tail2
def root391Tail0 : List Nat := 2791 :: root391Tail1
def root391Candidates0 : List Nat := [6607, 6961, 10333, 18313]
set_option maxRecDepth 8192 in
theorem root391Candidates0_checked : candidates edge 4 2707 2791 root391Tail1 winner.bound = root391Candidates0 := by
  unfold candidates
  have row : edge 2791 = (SortedNatTree.contains FastRows.row2791) := by
    change Data.row2791.contains = (SortedNatTree.contains FastRows.row2791)
    exact FastRows.row2791_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root391Child0 : search edge 3 [2707, 2791] root391Candidates0 5498 winner = winner := by decide
theorem root391Step0 : scan edge (search edge 3) 4 [2707] 2707 root391Tail0 winner =
    scan edge (search edge 3) 4 [2707] 2707 root391Tail1 winner := by
  rw [root391Tail0, scan, if_neg (by decide), root391Candidates0_checked]
  change scan edge (search edge 3) 4 [2707] 2707 root391Tail1
    (if 3 ≤ root391Candidates0.length then search edge 3 [2707, 2791] root391Candidates0 5498 winner else winner) = _
  rw [if_pos (by decide), root391Child0]
def root391Candidates1 : List Nat := [4273]
set_option maxRecDepth 8192 in
theorem root391Candidates1_checked : candidates edge 4 2707 2887 root391Tail2 winner.bound = root391Candidates1 := by
  unfold candidates
  have row : edge 2887 = (SortedNatTree.contains FastRows.row2887) := by
    change Data.row2887.contains = (SortedNatTree.contains FastRows.row2887)
    exact FastRows.row2887_checked.symm
  rw [row]
  decide
theorem root391Step1 : scan edge (search edge 3) 4 [2707] 2707 root391Tail1 winner =
    scan edge (search edge 3) 4 [2707] 2707 root391Tail2 winner := by
  rw [root391Tail1, scan, if_neg (by decide), root391Candidates1_checked]
  change scan edge (search edge 3) 4 [2707] 2707 root391Tail2
    (if 3 ≤ root391Candidates1.length then search edge 3 [2707, 2887] root391Candidates1 5594 winner else winner) = _
  rw [if_neg (by decide)]
def root391Candidates2 : List Nat := [3517, 5557, 9973, 13249, 14503, 18493]
set_option maxRecDepth 8192 in
theorem root391Candidates2_checked : candidates edge 4 2707 3253 root391Tail3 winner.bound = root391Candidates2 := by
  unfold candidates
  have row : edge 3253 = (SortedNatTree.contains FastRows.row3253) := by
    change Data.row3253.contains = (SortedNatTree.contains FastRows.row3253)
    exact FastRows.row3253_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root391Child2 : search edge 3 [2707, 3253] root391Candidates2 5960 winner = winner := by decide
theorem root391Step2 : scan edge (search edge 3) 4 [2707] 2707 root391Tail2 winner =
    scan edge (search edge 3) 4 [2707] 2707 root391Tail3 winner := by
  rw [root391Tail2, scan, if_neg (by decide), root391Candidates2_checked]
  change scan edge (search edge 3) 4 [2707] 2707 root391Tail3
    (if 3 ≤ root391Candidates2.length then search edge 3 [2707, 3253] root391Candidates2 5960 winner else winner) = _
  rw [if_pos (by decide), root391Child2]
def root391Candidates3 : List Nat := [6079, 8053, 11443, 14503, 15583]
set_option maxRecDepth 8192 in
theorem root391Candidates3_checked : candidates edge 4 2707 3463 root391Tail4 winner.bound = root391Candidates3 := by
  unfold candidates
  have row : edge 3463 = (SortedNatTree.contains FastRows.row3463) := by
    change Data.row3463.contains = (SortedNatTree.contains FastRows.row3463)
    exact FastRows.row3463_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root391Child3 : search edge 3 [2707, 3463] root391Candidates3 6170 winner = winner := by decide
theorem root391Step3 : scan edge (search edge 3) 4 [2707] 2707 root391Tail3 winner =
    scan edge (search edge 3) 4 [2707] 2707 root391Tail4 winner := by
  rw [root391Tail3, scan, if_neg (by decide), root391Candidates3_checked]
  change scan edge (search edge 3) 4 [2707] 2707 root391Tail4
    (if 3 ≤ root391Candidates3.length then search edge 3 [2707, 3463] root391Candidates3 6170 winner else winner) = _
  rw [if_pos (by decide), root391Child3]
def root391Candidates4 : List Nat := [8599]
set_option maxRecDepth 8192 in
theorem root391Candidates4_checked : candidates edge 4 2707 3517 root391Tail5 winner.bound = root391Candidates4 := by
  unfold candidates
  have row : edge 3517 = (SortedNatTree.contains FastRows.row3517) := by
    change Data.row3517.contains = (SortedNatTree.contains FastRows.row3517)
    exact FastRows.row3517_checked.symm
  rw [row]
  decide
theorem root391Step4 : scan edge (search edge 3) 4 [2707] 2707 root391Tail4 winner =
    scan edge (search edge 3) 4 [2707] 2707 root391Tail5 winner := by
  rw [root391Tail4, scan, if_neg (by decide), root391Candidates4_checked]
  change scan edge (search edge 3) 4 [2707] 2707 root391Tail5
    (if 3 ≤ root391Candidates4.length then search edge 3 [2707, 3517] root391Candidates4 6224 winner else winner) = _
  rw [if_neg (by decide)]
def root391Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root391Candidates5_checked : candidates edge 4 2707 3793 root391Tail6 winner.bound = root391Candidates5 := by
  unfold candidates
  have row : edge 3793 = (SortedNatTree.contains FastRows.row3793) := by
    change Data.row3793.contains = (SortedNatTree.contains FastRows.row3793)
    exact FastRows.row3793_checked.symm
  rw [row]
  decide
theorem root391Step5 : scan edge (search edge 3) 4 [2707] 2707 root391Tail5 winner =
    scan edge (search edge 3) 4 [2707] 2707 root391Tail6 winner := by
  rw [root391Tail5, scan, if_neg (by decide), root391Candidates5_checked]
  change scan edge (search edge 3) 4 [2707] 2707 root391Tail6
    (if 3 ≤ root391Candidates5.length then search edge 3 [2707, 3793] root391Candidates5 6500 winner else winner) = _
  rw [if_neg (by decide)]
def root391Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root391Candidates6_checked : candidates edge 4 2707 4261 root391Tail7 winner.bound = root391Candidates6 := by
  unfold candidates
  have row : edge 4261 = (SortedNatTree.contains FastRows.row4261) := by
    change Data.row4261.contains = (SortedNatTree.contains FastRows.row4261)
    exact FastRows.row4261_checked.symm
  rw [row]
  decide
theorem root391Step6 : scan edge (search edge 3) 4 [2707] 2707 root391Tail6 winner =
    scan edge (search edge 3) 4 [2707] 2707 root391Tail7 winner := by
  rw [root391Tail6, scan, if_neg (by decide), root391Candidates6_checked]
  change scan edge (search edge 3) 4 [2707] 2707 root391Tail7
    (if 3 ≤ root391Candidates6.length then search edge 3 [2707, 4261] root391Candidates6 6968 winner else winner) = _
  rw [if_neg (by decide)]
def root391Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root391Candidates7_checked : candidates edge 4 2707 4273 root391Tail8 winner.bound = root391Candidates7 := by
  unfold candidates
  have row : edge 4273 = (SortedNatTree.contains FastRows.row4273) := by
    change Data.row4273.contains = (SortedNatTree.contains FastRows.row4273)
    exact FastRows.row4273_checked.symm
  rw [row]
  decide
theorem root391Step7 : scan edge (search edge 3) 4 [2707] 2707 root391Tail7 winner =
    scan edge (search edge 3) 4 [2707] 2707 root391Tail8 winner := by
  rw [root391Tail7, scan, if_neg (by decide), root391Candidates7_checked]
  change scan edge (search edge 3) 4 [2707] 2707 root391Tail8
    (if 3 ≤ root391Candidates7.length then search edge 3 [2707, 4273] root391Candidates7 6980 winner else winner) = _
  rw [if_neg (by decide)]
def root391Candidates8 : List Nat := [7759]
set_option maxRecDepth 8192 in
theorem root391Candidates8_checked : candidates edge 4 2707 4507 root391Tail9 winner.bound = root391Candidates8 := by
  unfold candidates
  have row : edge 4507 = (SortedNatTree.contains FastRows.row4507) := by
    change Data.row4507.contains = (SortedNatTree.contains FastRows.row4507)
    exact FastRows.row4507_checked.symm
  rw [row]
  decide
theorem root391Step8 : scan edge (search edge 3) 4 [2707] 2707 root391Tail8 winner =
    scan edge (search edge 3) 4 [2707] 2707 root391Tail9 winner := by
  rw [root391Tail8, scan, if_neg (by decide), root391Candidates8_checked]
  change scan edge (search edge 3) 4 [2707] 2707 root391Tail9
    (if 3 ≤ root391Candidates8.length then search edge 3 [2707, 4507] root391Candidates8 7214 winner else winner) = _
  rw [if_neg (by decide)]
def root391Candidates9 : List Nat := [5857]
set_option maxRecDepth 8192 in
theorem root391Candidates9_checked : candidates edge 4 2707 4909 root391Tail10 winner.bound = root391Candidates9 := by
  unfold candidates
  have row : edge 4909 = (SortedNatTree.contains FastRows.row4909) := by
    change Data.row4909.contains = (SortedNatTree.contains FastRows.row4909)
    exact FastRows.row4909_checked.symm
  rw [row]
  decide
theorem root391Step9 : scan edge (search edge 3) 4 [2707] 2707 root391Tail9 winner =
    scan edge (search edge 3) 4 [2707] 2707 root391Tail10 winner := by
  rw [root391Tail9, scan, if_neg (by decide), root391Candidates9_checked]
  change scan edge (search edge 3) 4 [2707] 2707 root391Tail10
    (if 3 ≤ root391Candidates9.length then search edge 3 [2707, 4909] root391Candidates9 7616 winner else winner) = _
  rw [if_neg (by decide)]
def root391Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root391Candidates10_checked : candidates edge 4 2707 5413 root391Tail11 winner.bound = root391Candidates10 := by
  unfold candidates
  have row : edge 5413 = (SortedNatTree.contains FastRows.row5413) := by
    change Data.row5413.contains = (SortedNatTree.contains FastRows.row5413)
    exact FastRows.row5413_checked.symm
  rw [row]
  decide
theorem root391Step10 : scan edge (search edge 3) 4 [2707] 2707 root391Tail10 winner =
    scan edge (search edge 3) 4 [2707] 2707 root391Tail11 winner := by
  rw [root391Tail10, scan, if_neg (by decide), root391Candidates10_checked]
  change scan edge (search edge 3) 4 [2707] 2707 root391Tail11
    (if 3 ≤ root391Candidates10.length then search edge 3 [2707, 5413] root391Candidates10 8120 winner else winner) = _
  rw [if_neg (by decide)]
def root391Candidates11 : List Nat := [7789, 8053]
set_option maxRecDepth 8192 in
theorem root391Candidates11_checked : candidates edge 4 2707 5557 root391Tail12 winner.bound = root391Candidates11 := by
  unfold candidates
  have row : edge 5557 = (SortedNatTree.contains FastRows.row5557) := by
    change Data.row5557.contains = (SortedNatTree.contains FastRows.row5557)
    exact FastRows.row5557_checked.symm
  rw [row]
  decide
theorem root391Step11 : scan edge (search edge 3) 4 [2707] 2707 root391Tail11 winner =
    scan edge (search edge 3) 4 [2707] 2707 root391Tail12 winner := by
  rw [root391Tail11, scan, if_neg (by decide), root391Candidates11_checked]
  change scan edge (search edge 3) 4 [2707] 2707 root391Tail12
    (if 3 ≤ root391Candidates11.length then search edge 3 [2707, 5557] root391Candidates11 8264 winner else winner) = _
  rw [if_neg (by decide)]
def root391Candidates12 : List Nat := [16903]
set_option maxRecDepth 8192 in
theorem root391Candidates12_checked : candidates edge 4 2707 5701 root391Tail13 winner.bound = root391Candidates12 := by
  unfold candidates
  have row : edge 5701 = (SortedNatTree.contains FastRows.row5701) := by
    change Data.row5701.contains = (SortedNatTree.contains FastRows.row5701)
    exact FastRows.row5701_checked.symm
  rw [row]
  decide
theorem root391Step12 : scan edge (search edge 3) 4 [2707] 2707 root391Tail12 winner =
    scan edge (search edge 3) 4 [2707] 2707 root391Tail13 winner := by
  rw [root391Tail12, scan, if_neg (by decide), root391Candidates12_checked]
  change scan edge (search edge 3) 4 [2707] 2707 root391Tail13
    (if 3 ≤ root391Candidates12.length then search edge 3 [2707, 5701] root391Candidates12 8408 winner else winner) = _
  rw [if_neg (by decide)]
theorem root391Scan13 : scan edge (search edge 3) 4 [2707] 2707 root391Tail13 winner = winner := by rw [root391Tail13, scan, if_pos (by decide)]
theorem root391Scan12 : scan edge (search edge 3) 4 [2707] 2707 root391Tail12 winner = winner :=
  (root391Step12).trans root391Scan13
theorem root391Scan11 : scan edge (search edge 3) 4 [2707] 2707 root391Tail11 winner = winner :=
  (root391Step11).trans root391Scan12
theorem root391Scan10 : scan edge (search edge 3) 4 [2707] 2707 root391Tail10 winner = winner :=
  (root391Step10).trans root391Scan11
theorem root391Scan9 : scan edge (search edge 3) 4 [2707] 2707 root391Tail9 winner = winner :=
  (root391Step9).trans root391Scan10
theorem root391Scan8 : scan edge (search edge 3) 4 [2707] 2707 root391Tail8 winner = winner :=
  (root391Step8).trans root391Scan9
theorem root391Scan7 : scan edge (search edge 3) 4 [2707] 2707 root391Tail7 winner = winner :=
  (root391Step7).trans root391Scan8
theorem root391Scan6 : scan edge (search edge 3) 4 [2707] 2707 root391Tail6 winner = winner :=
  (root391Step6).trans root391Scan7
theorem root391Scan5 : scan edge (search edge 3) 4 [2707] 2707 root391Tail5 winner = winner :=
  (root391Step5).trans root391Scan6
theorem root391Scan4 : scan edge (search edge 3) 4 [2707] 2707 root391Tail4 winner = winner :=
  (root391Step4).trans root391Scan5
theorem root391Scan3 : scan edge (search edge 3) 4 [2707] 2707 root391Tail3 winner = winner :=
  (root391Step3).trans root391Scan4
theorem root391Scan2 : scan edge (search edge 3) 4 [2707] 2707 root391Tail2 winner = winner :=
  (root391Step2).trans root391Scan3
theorem root391Scan1 : scan edge (search edge 3) 4 [2707] 2707 root391Tail1 winner = winner :=
  (root391Step1).trans root391Scan2
theorem root391Scan0 : scan edge (search edge 3) 4 [2707] 2707 root391Tail0 winner = winner :=
  (root391Step0).trans root391Scan1
set_option maxRecDepth 4096 in
theorem root391Branch : search edge 4 [2707] root391Candidates 2707 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2707] 2707 root391Tail0 winner = winner
  exact root391Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
