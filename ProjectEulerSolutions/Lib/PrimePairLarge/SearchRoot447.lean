import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root447Candidates : List Nat := Data.row3181
set_option maxRecDepth 8192 in
theorem root447Candidates_checked : candidates edge 5 0 3181 tail448 winner.bound = root447Candidates := by
  apply candidates_eq_row edge 5 0 3181 tail448 root447Candidates winner.bound tail448_ordered
  · decide
  · decide
  · funext q
    rfl
def root447Tail36 : List Nat := []
def root447Tail35 : List Nat := 22543 :: root447Tail36
def root447Tail34 : List Nat := 22027 :: root447Tail35
def root447Tail33 : List Nat := 21871 :: root447Tail34
def root447Tail32 : List Nat := 20611 :: root447Tail33
def root447Tail31 : List Nat := 19687 :: root447Tail32
def root447Tail30 : List Nat := 19267 :: root447Tail31
def root447Tail29 : List Nat := 18517 :: root447Tail30
def root447Tail28 : List Nat := 18217 :: root447Tail29
def root447Tail27 : List Nat := 17383 :: root447Tail28
def root447Tail26 : List Nat := 16063 :: root447Tail27
def root447Tail25 : List Nat := 15451 :: root447Tail26
def root447Tail24 : List Nat := 15361 :: root447Tail25
def root447Tail23 : List Nat := 13687 :: root447Tail24
def root447Tail22 : List Nat := 12253 :: root447Tail23
def root447Tail21 : List Nat := 11587 :: root447Tail22
def root447Tail20 : List Nat := 11197 :: root447Tail21
def root447Tail19 : List Nat := 11113 :: root447Tail20
def root447Tail18 : List Nat := 10729 :: root447Tail19
def root447Tail17 : List Nat := 9973 :: root447Tail18
def root447Tail16 : List Nat := 9421 :: root447Tail17
def root447Tail15 : List Nat := 9127 :: root447Tail16
def root447Tail14 : List Nat := 9043 :: root447Tail15
def root447Tail13 : List Nat := 8377 :: root447Tail14
def root447Tail12 : List Nat := 7507 :: root447Tail13
def root447Tail11 : List Nat := 6709 :: root447Tail12
def root447Tail10 : List Nat := 6481 :: root447Tail11
def root447Tail9 : List Nat := 6073 :: root447Tail10
def root447Tail8 : List Nat := 5527 :: root447Tail9
def root447Tail7 : List Nat := 5107 :: root447Tail8
def root447Tail6 : List Nat := 5011 :: root447Tail7
def root447Tail5 : List Nat := 4903 :: root447Tail6
def root447Tail4 : List Nat := 4507 :: root447Tail5
def root447Tail3 : List Nat := 4273 :: root447Tail4
def root447Tail2 : List Nat := 4219 :: root447Tail3
def root447Tail1 : List Nat := 4099 :: root447Tail2
def root447Tail0 : List Nat := 3343 :: root447Tail1
def root447Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root447Candidates0_checked : candidates edge 4 3181 3343 root447Tail1 winner.bound = root447Candidates0 := by
  unfold candidates
  have row : edge 3343 = (SortedNatTree.contains FastRows.row3343) := by
    change Data.row3343.contains = (SortedNatTree.contains FastRows.row3343)
    exact FastRows.row3343_checked.symm
  rw [row]
  decide
theorem root447Step0 : scan edge (search edge 3) 4 [3181] 3181 root447Tail0 winner =
    scan edge (search edge 3) 4 [3181] 3181 root447Tail1 winner := by
  rw [root447Tail0, scan, if_neg (by decide), root447Candidates0_checked]
  change scan edge (search edge 3) 4 [3181] 3181 root447Tail1
    (if 3 ≤ root447Candidates0.length then search edge 3 [3181, 3343] root447Candidates0 6524 winner else winner) = _
  rw [if_neg (by decide)]
def root447Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root447Candidates1_checked : candidates edge 4 3181 4099 root447Tail2 winner.bound = root447Candidates1 := by
  unfold candidates
  have row : edge 4099 = (SortedNatTree.contains FastRows.row4099) := by
    change Data.row4099.contains = (SortedNatTree.contains FastRows.row4099)
    exact FastRows.row4099_checked.symm
  rw [row]
  decide
theorem root447Step1 : scan edge (search edge 3) 4 [3181] 3181 root447Tail1 winner =
    scan edge (search edge 3) 4 [3181] 3181 root447Tail2 winner := by
  rw [root447Tail1, scan, if_neg (by decide), root447Candidates1_checked]
  change scan edge (search edge 3) 4 [3181] 3181 root447Tail2
    (if 3 ≤ root447Candidates1.length then search edge 3 [3181, 4099] root447Candidates1 7280 winner else winner) = _
  rw [if_neg (by decide)]
def root447Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root447Candidates2_checked : candidates edge 4 3181 4219 root447Tail3 winner.bound = root447Candidates2 := by
  unfold candidates
  have row : edge 4219 = (SortedNatTree.contains FastRows.row4219) := by
    change Data.row4219.contains = (SortedNatTree.contains FastRows.row4219)
    exact FastRows.row4219_checked.symm
  rw [row]
  decide
theorem root447Step2 : scan edge (search edge 3) 4 [3181] 3181 root447Tail2 winner =
    scan edge (search edge 3) 4 [3181] 3181 root447Tail3 winner := by
  rw [root447Tail2, scan, if_neg (by decide), root447Candidates2_checked]
  change scan edge (search edge 3) 4 [3181] 3181 root447Tail3
    (if 3 ≤ root447Candidates2.length then search edge 3 [3181, 4219] root447Candidates2 7400 winner else winner) = _
  rw [if_neg (by decide)]
def root447Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root447Candidates3_checked : candidates edge 4 3181 4273 root447Tail4 winner.bound = root447Candidates3 := by
  unfold candidates
  have row : edge 4273 = (SortedNatTree.contains FastRows.row4273) := by
    change Data.row4273.contains = (SortedNatTree.contains FastRows.row4273)
    exact FastRows.row4273_checked.symm
  rw [row]
  decide
theorem root447Step3 : scan edge (search edge 3) 4 [3181] 3181 root447Tail3 winner =
    scan edge (search edge 3) 4 [3181] 3181 root447Tail4 winner := by
  rw [root447Tail3, scan, if_neg (by decide), root447Candidates3_checked]
  change scan edge (search edge 3) 4 [3181] 3181 root447Tail4
    (if 3 ≤ root447Candidates3.length then search edge 3 [3181, 4273] root447Candidates3 7454 winner else winner) = _
  rw [if_neg (by decide)]
def root447Candidates4 : List Nat := [6709, 8377, 12253]
set_option maxRecDepth 8192 in
theorem root447Candidates4_checked : candidates edge 4 3181 4507 root447Tail5 winner.bound = root447Candidates4 := by
  unfold candidates
  have row : edge 4507 = (SortedNatTree.contains FastRows.row4507) := by
    change Data.row4507.contains = (SortedNatTree.contains FastRows.row4507)
    exact FastRows.row4507_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root447Child4 : search edge 3 [3181, 4507] root447Candidates4 7688 winner = winner := by decide
theorem root447Step4 : scan edge (search edge 3) 4 [3181] 3181 root447Tail4 winner =
    scan edge (search edge 3) 4 [3181] 3181 root447Tail5 winner := by
  rw [root447Tail4, scan, if_neg (by decide), root447Candidates4_checked]
  change scan edge (search edge 3) 4 [3181] 3181 root447Tail5
    (if 3 ≤ root447Candidates4.length then search edge 3 [3181, 4507] root447Candidates4 7688 winner else winner) = _
  rw [if_pos (by decide), root447Child4]
def root447Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root447Candidates5_checked : candidates edge 4 3181 4903 root447Tail6 winner.bound = root447Candidates5 := by
  unfold candidates
  have row : edge 4903 = (SortedNatTree.contains FastRows.row4903) := by
    change Data.row4903.contains = (SortedNatTree.contains FastRows.row4903)
    exact FastRows.row4903_checked.symm
  rw [row]
  decide
theorem root447Step5 : scan edge (search edge 3) 4 [3181] 3181 root447Tail5 winner =
    scan edge (search edge 3) 4 [3181] 3181 root447Tail6 winner := by
  rw [root447Tail5, scan, if_neg (by decide), root447Candidates5_checked]
  change scan edge (search edge 3) 4 [3181] 3181 root447Tail6
    (if 3 ≤ root447Candidates5.length then search edge 3 [3181, 4903] root447Candidates5 8084 winner else winner) = _
  rw [if_neg (by decide)]
def root447Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root447Candidates6_checked : candidates edge 4 3181 5011 root447Tail7 winner.bound = root447Candidates6 := by
  unfold candidates
  have row : edge 5011 = (SortedNatTree.contains FastRows.row5011) := by
    change Data.row5011.contains = (SortedNatTree.contains FastRows.row5011)
    exact FastRows.row5011_checked.symm
  rw [row]
  decide
theorem root447Step6 : scan edge (search edge 3) 4 [3181] 3181 root447Tail6 winner =
    scan edge (search edge 3) 4 [3181] 3181 root447Tail7 winner := by
  rw [root447Tail6, scan, if_neg (by decide), root447Candidates6_checked]
  change scan edge (search edge 3) 4 [3181] 3181 root447Tail7
    (if 3 ≤ root447Candidates6.length then search edge 3 [3181, 5011] root447Candidates6 8192 winner else winner) = _
  rw [if_neg (by decide)]
def root447Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root447Candidates7_checked : candidates edge 4 3181 5107 root447Tail8 winner.bound = root447Candidates7 := by
  unfold candidates
  have row : edge 5107 = (SortedNatTree.contains FastRows.row5107) := by
    change Data.row5107.contains = (SortedNatTree.contains FastRows.row5107)
    exact FastRows.row5107_checked.symm
  rw [row]
  decide
theorem root447Step7 : scan edge (search edge 3) 4 [3181] 3181 root447Tail7 winner =
    scan edge (search edge 3) 4 [3181] 3181 root447Tail8 winner := by
  rw [root447Tail7, scan, if_neg (by decide), root447Candidates7_checked]
  change scan edge (search edge 3) 4 [3181] 3181 root447Tail8
    (if 3 ≤ root447Candidates7.length then search edge 3 [3181, 5107] root447Candidates7 8288 winner else winner) = _
  rw [if_neg (by decide)]
def root447Candidates8 : List Nat := [13687]
set_option maxRecDepth 8192 in
theorem root447Candidates8_checked : candidates edge 4 3181 5527 root447Tail9 winner.bound = root447Candidates8 := by
  unfold candidates
  have row : edge 5527 = (SortedNatTree.contains FastRows.row5527) := by
    change Data.row5527.contains = (SortedNatTree.contains FastRows.row5527)
    exact FastRows.row5527_checked.symm
  rw [row]
  decide
theorem root447Step8 : scan edge (search edge 3) 4 [3181] 3181 root447Tail8 winner =
    scan edge (search edge 3) 4 [3181] 3181 root447Tail9 winner := by
  rw [root447Tail8, scan, if_neg (by decide), root447Candidates8_checked]
  change scan edge (search edge 3) 4 [3181] 3181 root447Tail9
    (if 3 ≤ root447Candidates8.length then search edge 3 [3181, 5527] root447Candidates8 8708 winner else winner) = _
  rw [if_neg (by decide)]
theorem root447Scan9 : scan edge (search edge 3) 4 [3181] 3181 root447Tail9 winner = winner := by rw [root447Tail9, scan, if_pos (by decide)]
theorem root447Scan8 : scan edge (search edge 3) 4 [3181] 3181 root447Tail8 winner = winner :=
  (root447Step8).trans root447Scan9
theorem root447Scan7 : scan edge (search edge 3) 4 [3181] 3181 root447Tail7 winner = winner :=
  (root447Step7).trans root447Scan8
theorem root447Scan6 : scan edge (search edge 3) 4 [3181] 3181 root447Tail6 winner = winner :=
  (root447Step6).trans root447Scan7
theorem root447Scan5 : scan edge (search edge 3) 4 [3181] 3181 root447Tail5 winner = winner :=
  (root447Step5).trans root447Scan6
theorem root447Scan4 : scan edge (search edge 3) 4 [3181] 3181 root447Tail4 winner = winner :=
  (root447Step4).trans root447Scan5
theorem root447Scan3 : scan edge (search edge 3) 4 [3181] 3181 root447Tail3 winner = winner :=
  (root447Step3).trans root447Scan4
theorem root447Scan2 : scan edge (search edge 3) 4 [3181] 3181 root447Tail2 winner = winner :=
  (root447Step2).trans root447Scan3
theorem root447Scan1 : scan edge (search edge 3) 4 [3181] 3181 root447Tail1 winner = winner :=
  (root447Step1).trans root447Scan2
theorem root447Scan0 : scan edge (search edge 3) 4 [3181] 3181 root447Tail0 winner = winner :=
  (root447Step0).trans root447Scan1
set_option maxRecDepth 4096 in
theorem root447Branch : search edge 4 [3181] root447Candidates 3181 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3181] 3181 root447Tail0 winner = winner
  exact root447Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
