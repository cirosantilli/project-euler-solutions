import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root469Candidates : List Nat := Data.row3347
set_option maxRecDepth 8192 in
theorem root469Candidates_checked : candidates edge 5 0 3347 tail470 winner.bound = root469Candidates := by
  apply candidates_eq_row edge 5 0 3347 tail470 root469Candidates winner.bound tail470_ordered
  · decide
  · decide
  · funext q
    rfl
def root469Tail42 : List Nat := []
def root469Tail41 : List Nat := 21377 :: root469Tail42
def root469Tail40 : List Nat := 21011 :: root469Tail41
def root469Tail39 : List Nat := 20873 :: root469Tail40
def root469Tail38 : List Nat := 20411 :: root469Tail39
def root469Tail37 : List Nat := 19961 :: root469Tail38
def root469Tail36 : List Nat := 19013 :: root469Tail37
def root469Tail35 : List Nat := 18869 :: root469Tail36
def root469Tail34 : List Nat := 17987 :: root469Tail35
def root469Tail33 : List Nat := 15971 :: root469Tail34
def root469Tail32 : List Nat := 14771 :: root469Tail33
def root469Tail31 : List Nat := 14591 :: root469Tail32
def root469Tail30 : List Nat := 13751 :: root469Tail31
def root469Tail29 : List Nat := 13619 :: root469Tail30
def root469Tail28 : List Nat := 13469 :: root469Tail29
def root469Tail27 : List Nat := 12821 :: root469Tail28
def root469Tail26 : List Nat := 12671 :: root469Tail27
def root469Tail25 : List Nat := 12377 :: root469Tail26
def root469Tail24 : List Nat := 12041 :: root469Tail25
def root469Tail23 : List Nat := 11777 :: root469Tail24
def root469Tail22 : List Nat := 11519 :: root469Tail23
def root469Tail21 : List Nat := 10427 :: root469Tail22
def root469Tail20 : List Nat := 10391 :: root469Tail21
def root469Tail19 : List Nat := 8999 :: root469Tail20
def root469Tail18 : List Nat := 8693 :: root469Tail19
def root469Tail17 : List Nat := 8273 :: root469Tail18
def root469Tail16 : List Nat := 8231 :: root469Tail17
def root469Tail15 : List Nat := 7793 :: root469Tail16
def root469Tail14 : List Nat := 7517 :: root469Tail15
def root469Tail13 : List Nat := 7331 :: root469Tail14
def root469Tail12 : List Nat := 7043 :: root469Tail13
def root469Tail11 : List Nat := 6959 :: root469Tail12
def root469Tail10 : List Nat := 6143 :: root469Tail11
def root469Tail9 : List Nat := 6101 :: root469Tail10
def root469Tail8 : List Nat := 6029 :: root469Tail9
def root469Tail7 : List Nat := 5399 :: root469Tail8
def root469Tail6 : List Nat := 5333 :: root469Tail7
def root469Tail5 : List Nat := 4967 :: root469Tail6
def root469Tail4 : List Nat := 4703 :: root469Tail5
def root469Tail3 : List Nat := 4673 :: root469Tail4
def root469Tail2 : List Nat := 4349 :: root469Tail3
def root469Tail1 : List Nat := 4211 :: root469Tail2
def root469Tail0 : List Nat := 3821 :: root469Tail1
def root469Candidates0 : List Nat := [4703, 12821, 14591]
set_option maxRecDepth 8192 in
theorem root469Candidates0_checked : candidates edge 4 3347 3821 root469Tail1 winner.bound = root469Candidates0 := by
  unfold candidates
  have row : edge 3821 = (SortedNatTree.contains FastRows.row3821) := by
    change Data.row3821.contains = (SortedNatTree.contains FastRows.row3821)
    exact FastRows.row3821_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root469Child0 : search edge 3 [3347, 3821] root469Candidates0 7168 winner = winner := by decide
theorem root469Step0 : scan edge (search edge 3) 4 [3347] 3347 root469Tail0 winner =
    scan edge (search edge 3) 4 [3347] 3347 root469Tail1 winner := by
  rw [root469Tail0, scan, if_neg (by decide), root469Candidates0_checked]
  change scan edge (search edge 3) 4 [3347] 3347 root469Tail1
    (if 3 ≤ root469Candidates0.length then search edge 3 [3347, 3821] root469Candidates0 7168 winner else winner) = _
  rw [if_pos (by decide), root469Child0]
def root469Candidates1 : List Nat := [13619]
set_option maxRecDepth 8192 in
theorem root469Candidates1_checked : candidates edge 4 3347 4211 root469Tail2 winner.bound = root469Candidates1 := by
  unfold candidates
  have row : edge 4211 = (SortedNatTree.contains FastRows.row4211) := by
    change Data.row4211.contains = (SortedNatTree.contains FastRows.row4211)
    exact FastRows.row4211_checked.symm
  rw [row]
  decide
theorem root469Step1 : scan edge (search edge 3) 4 [3347] 3347 root469Tail1 winner =
    scan edge (search edge 3) 4 [3347] 3347 root469Tail2 winner := by
  rw [root469Tail1, scan, if_neg (by decide), root469Candidates1_checked]
  change scan edge (search edge 3) 4 [3347] 3347 root469Tail2
    (if 3 ≤ root469Candidates1.length then search edge 3 [3347, 4211] root469Candidates1 7558 winner else winner) = _
  rw [if_neg (by decide)]
def root469Candidates2 : List Nat := [4967, 7793]
set_option maxRecDepth 8192 in
theorem root469Candidates2_checked : candidates edge 4 3347 4349 root469Tail3 winner.bound = root469Candidates2 := by
  unfold candidates
  have row : edge 4349 = (SortedNatTree.contains FastRows.row4349) := by
    change Data.row4349.contains = (SortedNatTree.contains FastRows.row4349)
    exact FastRows.row4349_checked.symm
  rw [row]
  decide
theorem root469Step2 : scan edge (search edge 3) 4 [3347] 3347 root469Tail2 winner =
    scan edge (search edge 3) 4 [3347] 3347 root469Tail3 winner := by
  rw [root469Tail2, scan, if_neg (by decide), root469Candidates2_checked]
  change scan edge (search edge 3) 4 [3347] 3347 root469Tail3
    (if 3 ≤ root469Candidates2.length then search edge 3 [3347, 4349] root469Candidates2 7696 winner else winner) = _
  rw [if_neg (by decide)]
def root469Candidates3 : List Nat := [10391]
set_option maxRecDepth 8192 in
theorem root469Candidates3_checked : candidates edge 4 3347 4673 root469Tail4 winner.bound = root469Candidates3 := by
  unfold candidates
  have row : edge 4673 = (SortedNatTree.contains FastRows.row4673) := by
    change Data.row4673.contains = (SortedNatTree.contains FastRows.row4673)
    exact FastRows.row4673_checked.symm
  rw [row]
  decide
theorem root469Step3 : scan edge (search edge 3) 4 [3347] 3347 root469Tail3 winner =
    scan edge (search edge 3) 4 [3347] 3347 root469Tail4 winner := by
  rw [root469Tail3, scan, if_neg (by decide), root469Candidates3_checked]
  change scan edge (search edge 3) 4 [3347] 3347 root469Tail4
    (if 3 ≤ root469Candidates3.length then search edge 3 [3347, 4673] root469Candidates3 8020 winner else winner) = _
  rw [if_neg (by decide)]
def root469Candidates4 : List Nat := [13751]
set_option maxRecDepth 8192 in
theorem root469Candidates4_checked : candidates edge 4 3347 4703 root469Tail5 winner.bound = root469Candidates4 := by
  unfold candidates
  have row : edge 4703 = (SortedNatTree.contains FastRows.row4703) := by
    change Data.row4703.contains = (SortedNatTree.contains FastRows.row4703)
    exact FastRows.row4703_checked.symm
  rw [row]
  decide
theorem root469Step4 : scan edge (search edge 3) 4 [3347] 3347 root469Tail4 winner =
    scan edge (search edge 3) 4 [3347] 3347 root469Tail5 winner := by
  rw [root469Tail4, scan, if_neg (by decide), root469Candidates4_checked]
  change scan edge (search edge 3) 4 [3347] 3347 root469Tail5
    (if 3 ≤ root469Candidates4.length then search edge 3 [3347, 4703] root469Candidates4 8050 winner else winner) = _
  rw [if_neg (by decide)]
def root469Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root469Candidates5_checked : candidates edge 4 3347 4967 root469Tail6 winner.bound = root469Candidates5 := by
  unfold candidates
  have row : edge 4967 = (SortedNatTree.contains FastRows.row4967) := by
    change Data.row4967.contains = (SortedNatTree.contains FastRows.row4967)
    exact FastRows.row4967_checked.symm
  rw [row]
  decide
theorem root469Step5 : scan edge (search edge 3) 4 [3347] 3347 root469Tail5 winner =
    scan edge (search edge 3) 4 [3347] 3347 root469Tail6 winner := by
  rw [root469Tail5, scan, if_neg (by decide), root469Candidates5_checked]
  change scan edge (search edge 3) 4 [3347] 3347 root469Tail6
    (if 3 ≤ root469Candidates5.length then search edge 3 [3347, 4967] root469Candidates5 8314 winner else winner) = _
  rw [if_neg (by decide)]
def root469Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root469Candidates6_checked : candidates edge 4 3347 5333 root469Tail7 winner.bound = root469Candidates6 := by
  unfold candidates
  have row : edge 5333 = (SortedNatTree.contains FastRows.row5333) := by
    change Data.row5333.contains = (SortedNatTree.contains FastRows.row5333)
    exact FastRows.row5333_checked.symm
  rw [row]
  decide
theorem root469Step6 : scan edge (search edge 3) 4 [3347] 3347 root469Tail6 winner =
    scan edge (search edge 3) 4 [3347] 3347 root469Tail7 winner := by
  rw [root469Tail6, scan, if_neg (by decide), root469Candidates6_checked]
  change scan edge (search edge 3) 4 [3347] 3347 root469Tail7
    (if 3 ≤ root469Candidates6.length then search edge 3 [3347, 5333] root469Candidates6 8680 winner else winner) = _
  rw [if_neg (by decide)]
def root469Candidates7 : List Nat := [13469]
set_option maxRecDepth 8192 in
theorem root469Candidates7_checked : candidates edge 4 3347 5399 root469Tail8 winner.bound = root469Candidates7 := by
  unfold candidates
  have row : edge 5399 = (SortedNatTree.contains FastRows.row5399) := by
    change Data.row5399.contains = (SortedNatTree.contains FastRows.row5399)
    exact FastRows.row5399_checked.symm
  rw [row]
  decide
theorem root469Step7 : scan edge (search edge 3) 4 [3347] 3347 root469Tail7 winner =
    scan edge (search edge 3) 4 [3347] 3347 root469Tail8 winner := by
  rw [root469Tail7, scan, if_neg (by decide), root469Candidates7_checked]
  change scan edge (search edge 3) 4 [3347] 3347 root469Tail8
    (if 3 ≤ root469Candidates7.length then search edge 3 [3347, 5399] root469Candidates7 8746 winner else winner) = _
  rw [if_neg (by decide)]
theorem root469Scan8 : scan edge (search edge 3) 4 [3347] 3347 root469Tail8 winner = winner := by rw [root469Tail8, scan, if_pos (by decide)]
theorem root469Scan7 : scan edge (search edge 3) 4 [3347] 3347 root469Tail7 winner = winner :=
  (root469Step7).trans root469Scan8
theorem root469Scan6 : scan edge (search edge 3) 4 [3347] 3347 root469Tail6 winner = winner :=
  (root469Step6).trans root469Scan7
theorem root469Scan5 : scan edge (search edge 3) 4 [3347] 3347 root469Tail5 winner = winner :=
  (root469Step5).trans root469Scan6
theorem root469Scan4 : scan edge (search edge 3) 4 [3347] 3347 root469Tail4 winner = winner :=
  (root469Step4).trans root469Scan5
theorem root469Scan3 : scan edge (search edge 3) 4 [3347] 3347 root469Tail3 winner = winner :=
  (root469Step3).trans root469Scan4
theorem root469Scan2 : scan edge (search edge 3) 4 [3347] 3347 root469Tail2 winner = winner :=
  (root469Step2).trans root469Scan3
theorem root469Scan1 : scan edge (search edge 3) 4 [3347] 3347 root469Tail1 winner = winner :=
  (root469Step1).trans root469Scan2
theorem root469Scan0 : scan edge (search edge 3) 4 [3347] 3347 root469Tail0 winner = winner :=
  (root469Step0).trans root469Scan1
set_option maxRecDepth 4096 in
theorem root469Branch : search edge 4 [3347] root469Candidates 3347 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3347] 3347 root469Tail0 winner = winner
  exact root469Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
