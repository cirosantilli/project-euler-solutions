import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root318Candidates : List Nat := Data.row2131
set_option maxRecDepth 8192 in
theorem root318Candidates_checked : candidates edge 5 0 2131 tail319 winner.bound = root318Candidates := by
  apply candidates_eq_row edge 5 0 2131 tail319 root318Candidates winner.bound tail319_ordered
  · decide
  · decide
  · funext q
    rfl
def root318Tail45 : List Nat := []
def root318Tail44 : List Nat := 23623 :: root318Tail45
def root318Tail43 : List Nat := 23473 :: root318Tail44
def root318Tail42 : List Nat := 21943 :: root318Tail43
def root318Tail41 : List Nat := 21751 :: root318Tail42
def root318Tail40 : List Nat := 21661 :: root318Tail41
def root318Tail39 : List Nat := 20563 :: root318Tail40
def root318Tail38 : List Nat := 19597 :: root318Tail39
def root318Tail37 : List Nat := 19441 :: root318Tail38
def root318Tail36 : List Nat := 19381 :: root318Tail37
def root318Tail35 : List Nat := 19267 :: root318Tail36
def root318Tail34 : List Nat := 18121 :: root318Tail35
def root318Tail33 : List Nat := 18049 :: root318Tail34
def root318Tail32 : List Nat := 16699 :: root318Tail33
def root318Tail31 : List Nat := 16567 :: root318Tail32
def root318Tail30 : List Nat := 15619 :: root318Tail31
def root318Tail29 : List Nat := 14821 :: root318Tail30
def root318Tail28 : List Nat := 14173 :: root318Tail29
def root318Tail27 : List Nat := 13339 :: root318Tail28
def root318Tail26 : List Nat := 12919 :: root318Tail27
def root318Tail25 : List Nat := 12373 :: root318Tail26
def root318Tail24 : List Nat := 12049 :: root318Tail25
def root318Tail23 : List Nat := 10273 :: root318Tail24
def root318Tail22 : List Nat := 10159 :: root318Tail23
def root318Tail21 : List Nat := 9817 :: root318Tail22
def root318Tail20 : List Nat := 9769 :: root318Tail21
def root318Tail19 : List Nat := 8839 :: root318Tail20
def root318Tail18 : List Nat := 8377 :: root318Tail19
def root318Tail17 : List Nat := 7873 :: root318Tail18
def root318Tail16 : List Nat := 7669 :: root318Tail17
def root318Tail15 : List Nat := 7129 :: root318Tail16
def root318Tail14 : List Nat := 7027 :: root318Tail15
def root318Tail13 : List Nat := 6619 :: root318Tail14
def root318Tail12 : List Nat := 5689 :: root318Tail13
def root318Tail11 : List Nat := 5563 :: root318Tail12
def root318Tail10 : List Nat := 5023 :: root318Tail11
def root318Tail9 : List Nat := 4933 :: root318Tail10
def root318Tail8 : List Nat := 4567 :: root318Tail9
def root318Tail7 : List Nat := 4297 :: root318Tail8
def root318Tail6 : List Nat := 3547 :: root318Tail7
def root318Tail5 : List Nat := 3541 :: root318Tail6
def root318Tail4 : List Nat := 3343 :: root318Tail5
def root318Tail3 : List Nat := 3037 :: root318Tail4
def root318Tail2 : List Nat := 2833 :: root318Tail3
def root318Tail1 : List Nat := 2677 :: root318Tail2
def root318Tail0 : List Nat := 2293 :: root318Tail1
def root318Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root318Candidates0_checked : candidates edge 4 2131 2293 root318Tail1 winner.bound = root318Candidates0 := by
  unfold candidates
  have row : edge 2293 = (SortedNatTree.contains FastRows.row2293) := by
    change Data.row2293.contains = (SortedNatTree.contains FastRows.row2293)
    exact FastRows.row2293_checked.symm
  rw [row]
  decide
theorem root318Step0 : scan edge (search edge 3) 4 [2131] 2131 root318Tail0 winner =
    scan edge (search edge 3) 4 [2131] 2131 root318Tail1 winner := by
  rw [root318Tail0, scan, if_neg (by decide), root318Candidates0_checked]
  change scan edge (search edge 3) 4 [2131] 2131 root318Tail1
    (if 3 ≤ root318Candidates0.length then search edge 3 [2131, 2293] root318Candidates0 4424 winner else winner) = _
  rw [if_neg (by decide)]
def root318Candidates1 : List Nat := [2833, 5563]
set_option maxRecDepth 8192 in
theorem root318Candidates1_checked : candidates edge 4 2131 2677 root318Tail2 winner.bound = root318Candidates1 := by
  unfold candidates
  have row : edge 2677 = (SortedNatTree.contains FastRows.row2677) := by
    change Data.row2677.contains = (SortedNatTree.contains FastRows.row2677)
    exact FastRows.row2677_checked.symm
  rw [row]
  decide
theorem root318Step1 : scan edge (search edge 3) 4 [2131] 2131 root318Tail1 winner =
    scan edge (search edge 3) 4 [2131] 2131 root318Tail2 winner := by
  rw [root318Tail1, scan, if_neg (by decide), root318Candidates1_checked]
  change scan edge (search edge 3) 4 [2131] 2131 root318Tail2
    (if 3 ≤ root318Candidates1.length then search edge 3 [2131, 2677] root318Candidates1 4808 winner else winner) = _
  rw [if_neg (by decide)]
def root318Candidates2 : List Nat := [3037, 16567]
set_option maxRecDepth 8192 in
theorem root318Candidates2_checked : candidates edge 4 2131 2833 root318Tail3 winner.bound = root318Candidates2 := by
  unfold candidates
  have row : edge 2833 = (SortedNatTree.contains FastRows.row2833) := by
    change Data.row2833.contains = (SortedNatTree.contains FastRows.row2833)
    exact FastRows.row2833_checked.symm
  rw [row]
  decide
theorem root318Step2 : scan edge (search edge 3) 4 [2131] 2131 root318Tail2 winner =
    scan edge (search edge 3) 4 [2131] 2131 root318Tail3 winner := by
  rw [root318Tail2, scan, if_neg (by decide), root318Candidates2_checked]
  change scan edge (search edge 3) 4 [2131] 2131 root318Tail3
    (if 3 ≤ root318Candidates2.length then search edge 3 [2131, 2833] root318Candidates2 4964 winner else winner) = _
  rw [if_neg (by decide)]
def root318Candidates3 : List Nat := [5689, 7129]
set_option maxRecDepth 8192 in
theorem root318Candidates3_checked : candidates edge 4 2131 3037 root318Tail4 winner.bound = root318Candidates3 := by
  unfold candidates
  have row : edge 3037 = (SortedNatTree.contains FastRows.row3037) := by
    change Data.row3037.contains = (SortedNatTree.contains FastRows.row3037)
    exact FastRows.row3037_checked.symm
  rw [row]
  decide
theorem root318Step3 : scan edge (search edge 3) 4 [2131] 2131 root318Tail3 winner =
    scan edge (search edge 3) 4 [2131] 2131 root318Tail4 winner := by
  rw [root318Tail3, scan, if_neg (by decide), root318Candidates3_checked]
  change scan edge (search edge 3) 4 [2131] 2131 root318Tail4
    (if 3 ≤ root318Candidates3.length then search edge 3 [2131, 3037] root318Candidates3 5168 winner else winner) = _
  rw [if_neg (by decide)]
def root318Candidates4 : List Nat := [4567]
set_option maxRecDepth 8192 in
theorem root318Candidates4_checked : candidates edge 4 2131 3343 root318Tail5 winner.bound = root318Candidates4 := by
  unfold candidates
  have row : edge 3343 = (SortedNatTree.contains FastRows.row3343) := by
    change Data.row3343.contains = (SortedNatTree.contains FastRows.row3343)
    exact FastRows.row3343_checked.symm
  rw [row]
  decide
theorem root318Step4 : scan edge (search edge 3) 4 [2131] 2131 root318Tail4 winner =
    scan edge (search edge 3) 4 [2131] 2131 root318Tail5 winner := by
  rw [root318Tail4, scan, if_neg (by decide), root318Candidates4_checked]
  change scan edge (search edge 3) 4 [2131] 2131 root318Tail5
    (if 3 ≤ root318Candidates4.length then search edge 3 [2131, 3343] root318Candidates4 5474 winner else winner) = _
  rw [if_neg (by decide)]
def root318Candidates5 : List Nat := [4567, 12919]
set_option maxRecDepth 8192 in
theorem root318Candidates5_checked : candidates edge 4 2131 3541 root318Tail6 winner.bound = root318Candidates5 := by
  unfold candidates
  have row : edge 3541 = (SortedNatTree.contains FastRows.row3541) := by
    change Data.row3541.contains = (SortedNatTree.contains FastRows.row3541)
    exact FastRows.row3541_checked.symm
  rw [row]
  decide
theorem root318Step5 : scan edge (search edge 3) 4 [2131] 2131 root318Tail5 winner =
    scan edge (search edge 3) 4 [2131] 2131 root318Tail6 winner := by
  rw [root318Tail5, scan, if_neg (by decide), root318Candidates5_checked]
  change scan edge (search edge 3) 4 [2131] 2131 root318Tail6
    (if 3 ≤ root318Candidates5.length then search edge 3 [2131, 3541] root318Candidates5 5672 winner else winner) = _
  rw [if_neg (by decide)]
def root318Candidates6 : List Nat := [9817]
set_option maxRecDepth 8192 in
theorem root318Candidates6_checked : candidates edge 4 2131 3547 root318Tail7 winner.bound = root318Candidates6 := by
  unfold candidates
  have row : edge 3547 = (SortedNatTree.contains FastRows.row3547) := by
    change Data.row3547.contains = (SortedNatTree.contains FastRows.row3547)
    exact FastRows.row3547_checked.symm
  rw [row]
  decide
theorem root318Step6 : scan edge (search edge 3) 4 [2131] 2131 root318Tail6 winner =
    scan edge (search edge 3) 4 [2131] 2131 root318Tail7 winner := by
  rw [root318Tail6, scan, if_neg (by decide), root318Candidates6_checked]
  change scan edge (search edge 3) 4 [2131] 2131 root318Tail7
    (if 3 ≤ root318Candidates6.length then search edge 3 [2131, 3547] root318Candidates6 5678 winner else winner) = _
  rw [if_neg (by decide)]
def root318Candidates7 : List Nat := [5689]
set_option maxRecDepth 8192 in
theorem root318Candidates7_checked : candidates edge 4 2131 4297 root318Tail8 winner.bound = root318Candidates7 := by
  unfold candidates
  have row : edge 4297 = (SortedNatTree.contains FastRows.row4297) := by
    change Data.row4297.contains = (SortedNatTree.contains FastRows.row4297)
    exact FastRows.row4297_checked.symm
  rw [row]
  decide
theorem root318Step7 : scan edge (search edge 3) 4 [2131] 2131 root318Tail7 winner =
    scan edge (search edge 3) 4 [2131] 2131 root318Tail8 winner := by
  rw [root318Tail7, scan, if_neg (by decide), root318Candidates7_checked]
  change scan edge (search edge 3) 4 [2131] 2131 root318Tail8
    (if 3 ≤ root318Candidates7.length then search edge 3 [2131, 4297] root318Candidates7 6428 winner else winner) = _
  rw [if_neg (by decide)]
def root318Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root318Candidates8_checked : candidates edge 4 2131 4567 root318Tail9 winner.bound = root318Candidates8 := by
  unfold candidates
  have row : edge 4567 = (SortedNatTree.contains FastRows.row4567) := by
    change Data.row4567.contains = (SortedNatTree.contains FastRows.row4567)
    exact FastRows.row4567_checked.symm
  rw [row]
  decide
theorem root318Step8 : scan edge (search edge 3) 4 [2131] 2131 root318Tail8 winner =
    scan edge (search edge 3) 4 [2131] 2131 root318Tail9 winner := by
  rw [root318Tail8, scan, if_neg (by decide), root318Candidates8_checked]
  change scan edge (search edge 3) 4 [2131] 2131 root318Tail9
    (if 3 ≤ root318Candidates8.length then search edge 3 [2131, 4567] root318Candidates8 6698 winner else winner) = _
  rw [if_neg (by decide)]
def root318Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root318Candidates9_checked : candidates edge 4 2131 4933 root318Tail10 winner.bound = root318Candidates9 := by
  unfold candidates
  have row : edge 4933 = (SortedNatTree.contains FastRows.row4933) := by
    change Data.row4933.contains = (SortedNatTree.contains FastRows.row4933)
    exact FastRows.row4933_checked.symm
  rw [row]
  decide
theorem root318Step9 : scan edge (search edge 3) 4 [2131] 2131 root318Tail9 winner =
    scan edge (search edge 3) 4 [2131] 2131 root318Tail10 winner := by
  rw [root318Tail9, scan, if_neg (by decide), root318Candidates9_checked]
  change scan edge (search edge 3) 4 [2131] 2131 root318Tail10
    (if 3 ≤ root318Candidates9.length then search edge 3 [2131, 4933] root318Candidates9 7064 winner else winner) = _
  rw [if_neg (by decide)]
def root318Candidates10 : List Nat := [9769]
set_option maxRecDepth 8192 in
theorem root318Candidates10_checked : candidates edge 4 2131 5023 root318Tail11 winner.bound = root318Candidates10 := by
  unfold candidates
  have row : edge 5023 = (SortedNatTree.contains FastRows.row5023) := by
    change Data.row5023.contains = (SortedNatTree.contains FastRows.row5023)
    exact FastRows.row5023_checked.symm
  rw [row]
  decide
theorem root318Step10 : scan edge (search edge 3) 4 [2131] 2131 root318Tail10 winner =
    scan edge (search edge 3) 4 [2131] 2131 root318Tail11 winner := by
  rw [root318Tail10, scan, if_neg (by decide), root318Candidates10_checked]
  change scan edge (search edge 3) 4 [2131] 2131 root318Tail11
    (if 3 ≤ root318Candidates10.length then search edge 3 [2131, 5023] root318Candidates10 7154 winner else winner) = _
  rw [if_neg (by decide)]
def root318Candidates11 : List Nat := [9817, 14173]
set_option maxRecDepth 8192 in
theorem root318Candidates11_checked : candidates edge 4 2131 5563 root318Tail12 winner.bound = root318Candidates11 := by
  unfold candidates
  have row : edge 5563 = (SortedNatTree.contains FastRows.row5563) := by
    change Data.row5563.contains = (SortedNatTree.contains FastRows.row5563)
    exact FastRows.row5563_checked.symm
  rw [row]
  decide
theorem root318Step11 : scan edge (search edge 3) 4 [2131] 2131 root318Tail11 winner =
    scan edge (search edge 3) 4 [2131] 2131 root318Tail12 winner := by
  rw [root318Tail11, scan, if_neg (by decide), root318Candidates11_checked]
  change scan edge (search edge 3) 4 [2131] 2131 root318Tail12
    (if 3 ≤ root318Candidates11.length then search edge 3 [2131, 5563] root318Candidates11 7694 winner else winner) = _
  rw [if_neg (by decide)]
def root318Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root318Candidates12_checked : candidates edge 4 2131 5689 root318Tail13 winner.bound = root318Candidates12 := by
  unfold candidates
  have row : edge 5689 = (SortedNatTree.contains FastRows.row5689) := by
    change Data.row5689.contains = (SortedNatTree.contains FastRows.row5689)
    exact FastRows.row5689_checked.symm
  rw [row]
  decide
theorem root318Step12 : scan edge (search edge 3) 4 [2131] 2131 root318Tail12 winner =
    scan edge (search edge 3) 4 [2131] 2131 root318Tail13 winner := by
  rw [root318Tail12, scan, if_neg (by decide), root318Candidates12_checked]
  change scan edge (search edge 3) 4 [2131] 2131 root318Tail13
    (if 3 ≤ root318Candidates12.length then search edge 3 [2131, 5689] root318Candidates12 7820 winner else winner) = _
  rw [if_neg (by decide)]
theorem root318Scan13 : scan edge (search edge 3) 4 [2131] 2131 root318Tail13 winner = winner := by rw [root318Tail13, scan, if_pos (by decide)]
theorem root318Scan12 : scan edge (search edge 3) 4 [2131] 2131 root318Tail12 winner = winner :=
  (root318Step12).trans root318Scan13
theorem root318Scan11 : scan edge (search edge 3) 4 [2131] 2131 root318Tail11 winner = winner :=
  (root318Step11).trans root318Scan12
theorem root318Scan10 : scan edge (search edge 3) 4 [2131] 2131 root318Tail10 winner = winner :=
  (root318Step10).trans root318Scan11
theorem root318Scan9 : scan edge (search edge 3) 4 [2131] 2131 root318Tail9 winner = winner :=
  (root318Step9).trans root318Scan10
theorem root318Scan8 : scan edge (search edge 3) 4 [2131] 2131 root318Tail8 winner = winner :=
  (root318Step8).trans root318Scan9
theorem root318Scan7 : scan edge (search edge 3) 4 [2131] 2131 root318Tail7 winner = winner :=
  (root318Step7).trans root318Scan8
theorem root318Scan6 : scan edge (search edge 3) 4 [2131] 2131 root318Tail6 winner = winner :=
  (root318Step6).trans root318Scan7
theorem root318Scan5 : scan edge (search edge 3) 4 [2131] 2131 root318Tail5 winner = winner :=
  (root318Step5).trans root318Scan6
theorem root318Scan4 : scan edge (search edge 3) 4 [2131] 2131 root318Tail4 winner = winner :=
  (root318Step4).trans root318Scan5
theorem root318Scan3 : scan edge (search edge 3) 4 [2131] 2131 root318Tail3 winner = winner :=
  (root318Step3).trans root318Scan4
theorem root318Scan2 : scan edge (search edge 3) 4 [2131] 2131 root318Tail2 winner = winner :=
  (root318Step2).trans root318Scan3
theorem root318Scan1 : scan edge (search edge 3) 4 [2131] 2131 root318Tail1 winner = winner :=
  (root318Step1).trans root318Scan2
theorem root318Scan0 : scan edge (search edge 3) 4 [2131] 2131 root318Tail0 winner = winner :=
  (root318Step0).trans root318Scan1
set_option maxRecDepth 4096 in
theorem root318Branch : search edge 4 [2131] root318Candidates 2131 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2131] 2131 root318Tail0 winner = winner
  exact root318Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
