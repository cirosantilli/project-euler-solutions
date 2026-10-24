import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root250Candidates : List Nat := Data.row1607
set_option maxRecDepth 8192 in
theorem root250Candidates_checked : candidates edge 5 0 1607 tail251 winner.bound = root250Candidates := by
  apply candidates_eq_row edge 5 0 1607 tail251 root250Candidates winner.bound tail251_ordered
  · decide
  · decide
  · funext q
    rfl
def root250Tail45 : List Nat := []
def root250Tail44 : List Nat := 24281 :: root250Tail45
def root250Tail43 : List Nat := 23357 :: root250Tail44
def root250Tail42 : List Nat := 22937 :: root250Tail43
def root250Tail41 : List Nat := 22853 :: root250Tail42
def root250Tail40 : List Nat := 22013 :: root250Tail41
def root250Tail39 : List Nat := 21347 :: root250Tail40
def root250Tail38 : List Nat := 21191 :: root250Tail39
def root250Tail37 : List Nat := 21023 :: root250Tail38
def root250Tail36 : List Nat := 20939 :: root250Tail37
def root250Tail35 : List Nat := 19919 :: root250Tail36
def root250Tail34 : List Nat := 19913 :: root250Tail35
def root250Tail33 : List Nat := 19709 :: root250Tail34
def root250Tail32 : List Nat := 19661 :: root250Tail33
def root250Tail31 : List Nat := 18797 :: root250Tail32
def root250Tail30 : List Nat := 18671 :: root250Tail31
def root250Tail29 : List Nat := 18443 :: root250Tail30
def root250Tail28 : List Nat := 17939 :: root250Tail29
def root250Tail27 : List Nat := 16901 :: root250Tail28
def root250Tail26 : List Nat := 15017 :: root250Tail27
def root250Tail25 : List Nat := 13841 :: root250Tail26
def root250Tail24 : List Nat := 13697 :: root250Tail25
def root250Tail23 : List Nat := 13127 :: root250Tail24
def root250Tail22 : List Nat := 11777 :: root250Tail23
def root250Tail21 : List Nat := 10301 :: root250Tail22
def root250Tail20 : List Nat := 10181 :: root250Tail21
def root250Tail19 : List Nat := 9629 :: root250Tail20
def root250Tail18 : List Nat := 9221 :: root250Tail19
def root250Tail17 : List Nat := 8171 :: root250Tail18
def root250Tail16 : List Nat := 7877 :: root250Tail17
def root250Tail15 : List Nat := 7673 :: root250Tail16
def root250Tail14 : List Nat := 7541 :: root250Tail15
def root250Tail13 : List Nat := 7109 :: root250Tail14
def root250Tail12 : List Nat := 5903 :: root250Tail13
def root250Tail11 : List Nat := 5849 :: root250Tail12
def root250Tail10 : List Nat := 5471 :: root250Tail11
def root250Tail9 : List Nat := 5351 :: root250Tail10
def root250Tail8 : List Nat := 5099 :: root250Tail9
def root250Tail7 : List Nat := 5009 :: root250Tail8
def root250Tail6 : List Nat := 4637 :: root250Tail7
def root250Tail5 : List Nat := 4481 :: root250Tail6
def root250Tail4 : List Nat := 4271 :: root250Tail5
def root250Tail3 : List Nat := 3767 :: root250Tail4
def root250Tail2 : List Nat := 3203 :: root250Tail3
def root250Tail1 : List Nat := 2549 :: root250Tail2
def root250Tail0 : List Nat := 1901 :: root250Tail1
def root250Candidates0 : List Nat := [19913, 20939]
set_option maxRecDepth 8192 in
theorem root250Candidates0_checked : candidates edge 4 1607 1901 root250Tail1 winner.bound = root250Candidates0 := by
  unfold candidates
  have row : edge 1901 = (SortedNatTree.contains FastRows.row1901) := by
    change Data.row1901.contains = (SortedNatTree.contains FastRows.row1901)
    exact FastRows.row1901_checked.symm
  rw [row]
  decide
theorem root250Step0 : scan edge (search edge 3) 4 [1607] 1607 root250Tail0 winner =
    scan edge (search edge 3) 4 [1607] 1607 root250Tail1 winner := by
  rw [root250Tail0, scan, if_neg (by decide), root250Candidates0_checked]
  change scan edge (search edge 3) 4 [1607] 1607 root250Tail1
    (if 3 ≤ root250Candidates0.length then search edge 3 [1607, 1901] root250Candidates0 3508 winner else winner) = _
  rw [if_neg (by decide)]
def root250Candidates1 : List Nat := [5471, 7673]
set_option maxRecDepth 8192 in
theorem root250Candidates1_checked : candidates edge 4 1607 2549 root250Tail2 winner.bound = root250Candidates1 := by
  unfold candidates
  have row : edge 2549 = (SortedNatTree.contains FastRows.row2549) := by
    change Data.row2549.contains = (SortedNatTree.contains FastRows.row2549)
    exact FastRows.row2549_checked.symm
  rw [row]
  decide
theorem root250Step1 : scan edge (search edge 3) 4 [1607] 1607 root250Tail1 winner =
    scan edge (search edge 3) 4 [1607] 1607 root250Tail2 winner := by
  rw [root250Tail1, scan, if_neg (by decide), root250Candidates1_checked]
  change scan edge (search edge 3) 4 [1607] 1607 root250Tail2
    (if 3 ≤ root250Candidates1.length then search edge 3 [1607, 2549] root250Candidates1 4156 winner else winner) = _
  rw [if_neg (by decide)]
def root250Candidates2 : List Nat := [4481, 7541, 9221]
set_option maxRecDepth 8192 in
theorem root250Candidates2_checked : candidates edge 4 1607 3203 root250Tail3 winner.bound = root250Candidates2 := by
  unfold candidates
  have row : edge 3203 = (SortedNatTree.contains FastRows.row3203) := by
    change Data.row3203.contains = (SortedNatTree.contains FastRows.row3203)
    exact FastRows.row3203_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root250Child2 : search edge 3 [1607, 3203] root250Candidates2 4810 winner = winner := by decide
theorem root250Step2 : scan edge (search edge 3) 4 [1607] 1607 root250Tail2 winner =
    scan edge (search edge 3) 4 [1607] 1607 root250Tail3 winner := by
  rw [root250Tail2, scan, if_neg (by decide), root250Candidates2_checked]
  change scan edge (search edge 3) 4 [1607] 1607 root250Tail3
    (if 3 ≤ root250Candidates2.length then search edge 3 [1607, 3203] root250Candidates2 4810 winner else winner) = _
  rw [if_pos (by decide), root250Child2]
def root250Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root250Candidates3_checked : candidates edge 4 1607 3767 root250Tail4 winner.bound = root250Candidates3 := by
  unfold candidates
  have row : edge 3767 = (SortedNatTree.contains FastRows.row3767) := by
    change Data.row3767.contains = (SortedNatTree.contains FastRows.row3767)
    exact FastRows.row3767_checked.symm
  rw [row]
  decide
theorem root250Step3 : scan edge (search edge 3) 4 [1607] 1607 root250Tail3 winner =
    scan edge (search edge 3) 4 [1607] 1607 root250Tail4 winner := by
  rw [root250Tail3, scan, if_neg (by decide), root250Candidates3_checked]
  change scan edge (search edge 3) 4 [1607] 1607 root250Tail4
    (if 3 ≤ root250Candidates3.length then search edge 3 [1607, 3767] root250Candidates3 5374 winner else winner) = _
  rw [if_neg (by decide)]
def root250Candidates4 : List Nat := [5009, 11777]
set_option maxRecDepth 8192 in
theorem root250Candidates4_checked : candidates edge 4 1607 4271 root250Tail5 winner.bound = root250Candidates4 := by
  unfold candidates
  have row : edge 4271 = (SortedNatTree.contains FastRows.row4271) := by
    change Data.row4271.contains = (SortedNatTree.contains FastRows.row4271)
    exact FastRows.row4271_checked.symm
  rw [row]
  decide
theorem root250Step4 : scan edge (search edge 3) 4 [1607] 1607 root250Tail4 winner =
    scan edge (search edge 3) 4 [1607] 1607 root250Tail5 winner := by
  rw [root250Tail4, scan, if_neg (by decide), root250Candidates4_checked]
  change scan edge (search edge 3) 4 [1607] 1607 root250Tail5
    (if 3 ≤ root250Candidates4.length then search edge 3 [1607, 4271] root250Candidates4 5878 winner else winner) = _
  rw [if_neg (by decide)]
def root250Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root250Candidates5_checked : candidates edge 4 1607 4481 root250Tail6 winner.bound = root250Candidates5 := by
  unfold candidates
  have row : edge 4481 = (SortedNatTree.contains FastRows.row4481) := by
    change Data.row4481.contains = (SortedNatTree.contains FastRows.row4481)
    exact FastRows.row4481_checked.symm
  rw [row]
  decide
theorem root250Step5 : scan edge (search edge 3) 4 [1607] 1607 root250Tail5 winner =
    scan edge (search edge 3) 4 [1607] 1607 root250Tail6 winner := by
  rw [root250Tail5, scan, if_neg (by decide), root250Candidates5_checked]
  change scan edge (search edge 3) 4 [1607] 1607 root250Tail6
    (if 3 ≤ root250Candidates5.length then search edge 3 [1607, 4481] root250Candidates5 6088 winner else winner) = _
  rw [if_neg (by decide)]
def root250Candidates6 : List Nat := [5009, 5471, 18797, 19661]
set_option maxRecDepth 8192 in
theorem root250Candidates6_checked : candidates edge 4 1607 4637 root250Tail7 winner.bound = root250Candidates6 := by
  unfold candidates
  have row : edge 4637 = (SortedNatTree.contains FastRows.row4637) := by
    change Data.row4637.contains = (SortedNatTree.contains FastRows.row4637)
    exact FastRows.row4637_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root250Child6 : search edge 3 [1607, 4637] root250Candidates6 6244 winner = winner := by decide
theorem root250Step6 : scan edge (search edge 3) 4 [1607] 1607 root250Tail6 winner =
    scan edge (search edge 3) 4 [1607] 1607 root250Tail7 winner := by
  rw [root250Tail6, scan, if_neg (by decide), root250Candidates6_checked]
  change scan edge (search edge 3) 4 [1607] 1607 root250Tail7
    (if 3 ≤ root250Candidates6.length then search edge 3 [1607, 4637] root250Candidates6 6244 winner else winner) = _
  rw [if_pos (by decide), root250Child6]
def root250Candidates7 : List Nat := [5351, 5849, 7109, 18797]
set_option maxRecDepth 8192 in
theorem root250Candidates7_checked : candidates edge 4 1607 5009 root250Tail8 winner.bound = root250Candidates7 := by
  unfold candidates
  have row : edge 5009 = (SortedNatTree.contains FastRows.row5009) := by
    change Data.row5009.contains = (SortedNatTree.contains FastRows.row5009)
    exact FastRows.row5009_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root250Child7 : search edge 3 [1607, 5009] root250Candidates7 6616 winner = winner := by decide
theorem root250Step7 : scan edge (search edge 3) 4 [1607] 1607 root250Tail7 winner =
    scan edge (search edge 3) 4 [1607] 1607 root250Tail8 winner := by
  rw [root250Tail7, scan, if_neg (by decide), root250Candidates7_checked]
  change scan edge (search edge 3) 4 [1607] 1607 root250Tail8
    (if 3 ≤ root250Candidates7.length then search edge 3 [1607, 5009] root250Candidates7 6616 winner else winner) = _
  rw [if_pos (by decide), root250Child7]
def root250Candidates8 : List Nat := [5471]
set_option maxRecDepth 8192 in
theorem root250Candidates8_checked : candidates edge 4 1607 5099 root250Tail9 winner.bound = root250Candidates8 := by
  unfold candidates
  have row : edge 5099 = (SortedNatTree.contains FastRows.row5099) := by
    change Data.row5099.contains = (SortedNatTree.contains FastRows.row5099)
    exact FastRows.row5099_checked.symm
  rw [row]
  decide
theorem root250Step8 : scan edge (search edge 3) 4 [1607] 1607 root250Tail8 winner =
    scan edge (search edge 3) 4 [1607] 1607 root250Tail9 winner := by
  rw [root250Tail8, scan, if_neg (by decide), root250Candidates8_checked]
  change scan edge (search edge 3) 4 [1607] 1607 root250Tail9
    (if 3 ≤ root250Candidates8.length then search edge 3 [1607, 5099] root250Candidates8 6706 winner else winner) = _
  rw [if_neg (by decide)]
def root250Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root250Candidates9_checked : candidates edge 4 1607 5351 root250Tail10 winner.bound = root250Candidates9 := by
  unfold candidates
  have row : edge 5351 = (SortedNatTree.contains FastRows.row5351) := by
    change Data.row5351.contains = (SortedNatTree.contains FastRows.row5351)
    exact FastRows.row5351_checked.symm
  rw [row]
  decide
theorem root250Step9 : scan edge (search edge 3) 4 [1607] 1607 root250Tail9 winner =
    scan edge (search edge 3) 4 [1607] 1607 root250Tail10 winner := by
  rw [root250Tail9, scan, if_neg (by decide), root250Candidates9_checked]
  change scan edge (search edge 3) 4 [1607] 1607 root250Tail10
    (if 3 ≤ root250Candidates9.length then search edge 3 [1607, 5351] root250Candidates9 6958 winner else winner) = _
  rw [if_neg (by decide)]
def root250Candidates10 : List Nat := [8171]
set_option maxRecDepth 8192 in
theorem root250Candidates10_checked : candidates edge 4 1607 5471 root250Tail11 winner.bound = root250Candidates10 := by
  unfold candidates
  have row : edge 5471 = (SortedNatTree.contains FastRows.row5471) := by
    change Data.row5471.contains = (SortedNatTree.contains FastRows.row5471)
    exact FastRows.row5471_checked.symm
  rw [row]
  decide
theorem root250Step10 : scan edge (search edge 3) 4 [1607] 1607 root250Tail10 winner =
    scan edge (search edge 3) 4 [1607] 1607 root250Tail11 winner := by
  rw [root250Tail10, scan, if_neg (by decide), root250Candidates10_checked]
  change scan edge (search edge 3) 4 [1607] 1607 root250Tail11
    (if 3 ≤ root250Candidates10.length then search edge 3 [1607, 5471] root250Candidates10 7078 winner else winner) = _
  rw [if_neg (by decide)]
def root250Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root250Candidates11_checked : candidates edge 4 1607 5849 root250Tail12 winner.bound = root250Candidates11 := by
  unfold candidates
  have row : edge 5849 = (SortedNatTree.contains FastRows.row5849) := by
    change Data.row5849.contains = (SortedNatTree.contains FastRows.row5849)
    exact FastRows.row5849_checked.symm
  rw [row]
  decide
theorem root250Step11 : scan edge (search edge 3) 4 [1607] 1607 root250Tail11 winner =
    scan edge (search edge 3) 4 [1607] 1607 root250Tail12 winner := by
  rw [root250Tail11, scan, if_neg (by decide), root250Candidates11_checked]
  change scan edge (search edge 3) 4 [1607] 1607 root250Tail12
    (if 3 ≤ root250Candidates11.length then search edge 3 [1607, 5849] root250Candidates11 7456 winner else winner) = _
  rw [if_neg (by decide)]
def root250Candidates12 : List Nat := [7109, 7877]
set_option maxRecDepth 8192 in
theorem root250Candidates12_checked : candidates edge 4 1607 5903 root250Tail13 winner.bound = root250Candidates12 := by
  unfold candidates
  have row : edge 5903 = (SortedNatTree.contains FastRows.row5903) := by
    change Data.row5903.contains = (SortedNatTree.contains FastRows.row5903)
    exact FastRows.row5903_checked.symm
  rw [row]
  decide
theorem root250Step12 : scan edge (search edge 3) 4 [1607] 1607 root250Tail12 winner =
    scan edge (search edge 3) 4 [1607] 1607 root250Tail13 winner := by
  rw [root250Tail12, scan, if_neg (by decide), root250Candidates12_checked]
  change scan edge (search edge 3) 4 [1607] 1607 root250Tail13
    (if 3 ≤ root250Candidates12.length then search edge 3 [1607, 5903] root250Candidates12 7510 winner else winner) = _
  rw [if_neg (by decide)]
theorem root250Scan13 : scan edge (search edge 3) 4 [1607] 1607 root250Tail13 winner = winner := by rw [root250Tail13, scan, if_pos (by decide)]
theorem root250Scan12 : scan edge (search edge 3) 4 [1607] 1607 root250Tail12 winner = winner :=
  (root250Step12).trans root250Scan13
theorem root250Scan11 : scan edge (search edge 3) 4 [1607] 1607 root250Tail11 winner = winner :=
  (root250Step11).trans root250Scan12
theorem root250Scan10 : scan edge (search edge 3) 4 [1607] 1607 root250Tail10 winner = winner :=
  (root250Step10).trans root250Scan11
theorem root250Scan9 : scan edge (search edge 3) 4 [1607] 1607 root250Tail9 winner = winner :=
  (root250Step9).trans root250Scan10
theorem root250Scan8 : scan edge (search edge 3) 4 [1607] 1607 root250Tail8 winner = winner :=
  (root250Step8).trans root250Scan9
theorem root250Scan7 : scan edge (search edge 3) 4 [1607] 1607 root250Tail7 winner = winner :=
  (root250Step7).trans root250Scan8
theorem root250Scan6 : scan edge (search edge 3) 4 [1607] 1607 root250Tail6 winner = winner :=
  (root250Step6).trans root250Scan7
theorem root250Scan5 : scan edge (search edge 3) 4 [1607] 1607 root250Tail5 winner = winner :=
  (root250Step5).trans root250Scan6
theorem root250Scan4 : scan edge (search edge 3) 4 [1607] 1607 root250Tail4 winner = winner :=
  (root250Step4).trans root250Scan5
theorem root250Scan3 : scan edge (search edge 3) 4 [1607] 1607 root250Tail3 winner = winner :=
  (root250Step3).trans root250Scan4
theorem root250Scan2 : scan edge (search edge 3) 4 [1607] 1607 root250Tail2 winner = winner :=
  (root250Step2).trans root250Scan3
theorem root250Scan1 : scan edge (search edge 3) 4 [1607] 1607 root250Tail1 winner = winner :=
  (root250Step1).trans root250Scan2
theorem root250Scan0 : scan edge (search edge 3) 4 [1607] 1607 root250Tail0 winner = winner :=
  (root250Step0).trans root250Scan1
set_option maxRecDepth 4096 in
theorem root250Branch : search edge 4 [1607] root250Candidates 1607 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1607] 1607 root250Tail0 winner = winner
  exact root250Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
