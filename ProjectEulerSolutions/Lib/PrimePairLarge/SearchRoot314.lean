import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root314Candidates : List Nat := Data.row2099
set_option maxRecDepth 8192 in
theorem root314Candidates_checked : candidates edge 5 0 2099 tail315 winner.bound = root314Candidates := by
  apply candidates_eq_row edge 5 0 2099 tail315 root314Candidates winner.bound tail315_ordered
  · decide
  · decide
  · funext q
    rfl
def root314Tail34 : List Nat := []
def root314Tail33 : List Nat := 23159 :: root314Tail34
def root314Tail32 : List Nat := 22907 :: root314Tail33
def root314Tail31 : List Nat := 21143 :: root314Tail32
def root314Tail30 : List Nat := 21059 :: root314Tail31
def root314Tail29 : List Nat := 19793 :: root314Tail30
def root314Tail28 : List Nat := 19211 :: root314Tail29
def root314Tail27 : List Nat := 18803 :: root314Tail28
def root314Tail26 : List Nat := 17783 :: root314Tail27
def root314Tail25 : List Nat := 17597 :: root314Tail26
def root314Tail24 : List Nat := 14879 :: root314Tail25
def root314Tail23 : List Nat := 14867 :: root314Tail24
def root314Tail22 : List Nat := 14591 :: root314Tail23
def root314Tail21 : List Nat := 12953 :: root314Tail22
def root314Tail20 : List Nat := 12473 :: root314Tail21
def root314Tail19 : List Nat := 10559 :: root314Tail20
def root314Tail18 : List Nat := 10343 :: root314Tail19
def root314Tail17 : List Nat := 9887 :: root314Tail18
def root314Tail16 : List Nat := 9479 :: root314Tail17
def root314Tail15 : List Nat := 8093 :: root314Tail16
def root314Tail14 : List Nat := 8069 :: root314Tail15
def root314Tail13 : List Nat := 7841 :: root314Tail14
def root314Tail12 : List Nat := 7517 :: root314Tail13
def root314Tail11 : List Nat := 7211 :: root314Tail12
def root314Tail10 : List Nat := 6863 :: root314Tail11
def root314Tail9 : List Nat := 5717 :: root314Tail10
def root314Tail8 : List Nat := 5279 :: root314Tail9
def root314Tail7 : List Nat := 4241 :: root314Tail8
def root314Tail6 : List Nat := 3527 :: root314Tail7
def root314Tail5 : List Nat := 3083 :: root314Tail6
def root314Tail4 : List Nat := 2843 :: root314Tail5
def root314Tail3 : List Nat := 2693 :: root314Tail4
def root314Tail2 : List Nat := 2393 :: root314Tail3
def root314Tail1 : List Nat := 2381 :: root314Tail2
def root314Tail0 : List Nat := 2129 :: root314Tail1
def root314Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root314Candidates0_checked : candidates edge 4 2099 2129 root314Tail1 winner.bound = root314Candidates0 := by
  unfold candidates
  have row : edge 2129 = (SortedNatTree.contains FastRows.row2129) := by
    change Data.row2129.contains = (SortedNatTree.contains FastRows.row2129)
    exact FastRows.row2129_checked.symm
  rw [row]
  decide
theorem root314Step0 : scan edge (search edge 3) 4 [2099] 2099 root314Tail0 winner =
    scan edge (search edge 3) 4 [2099] 2099 root314Tail1 winner := by
  rw [root314Tail0, scan, if_neg (by decide), root314Candidates0_checked]
  change scan edge (search edge 3) 4 [2099] 2099 root314Tail1
    (if 3 ≤ root314Candidates0.length then search edge 3 [2099, 2129] root314Candidates0 4228 winner else winner) = _
  rw [if_neg (by decide)]
def root314Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root314Candidates1_checked : candidates edge 4 2099 2381 root314Tail2 winner.bound = root314Candidates1 := by
  unfold candidates
  have row : edge 2381 = (SortedNatTree.contains FastRows.row2381) := by
    change Data.row2381.contains = (SortedNatTree.contains FastRows.row2381)
    exact FastRows.row2381_checked.symm
  rw [row]
  decide
theorem root314Step1 : scan edge (search edge 3) 4 [2099] 2099 root314Tail1 winner =
    scan edge (search edge 3) 4 [2099] 2099 root314Tail2 winner := by
  rw [root314Tail1, scan, if_neg (by decide), root314Candidates1_checked]
  change scan edge (search edge 3) 4 [2099] 2099 root314Tail2
    (if 3 ≤ root314Candidates1.length then search edge 3 [2099, 2381] root314Candidates1 4480 winner else winner) = _
  rw [if_neg (by decide)]
def root314Candidates2 : List Nat := [3083, 4241, 8069]
set_option maxRecDepth 8192 in
theorem root314Candidates2_checked : candidates edge 4 2099 2393 root314Tail3 winner.bound = root314Candidates2 := by
  unfold candidates
  have row : edge 2393 = (SortedNatTree.contains FastRows.row2393) := by
    change Data.row2393.contains = (SortedNatTree.contains FastRows.row2393)
    exact FastRows.row2393_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root314Child2 : search edge 3 [2099, 2393] root314Candidates2 4492 winner = winner := by decide
theorem root314Step2 : scan edge (search edge 3) 4 [2099] 2099 root314Tail2 winner =
    scan edge (search edge 3) 4 [2099] 2099 root314Tail3 winner := by
  rw [root314Tail2, scan, if_neg (by decide), root314Candidates2_checked]
  change scan edge (search edge 3) 4 [2099] 2099 root314Tail3
    (if 3 ≤ root314Candidates2.length then search edge 3 [2099, 2393] root314Candidates2 4492 winner else winner) = _
  rw [if_pos (by decide), root314Child2]
def root314Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root314Candidates3_checked : candidates edge 4 2099 2693 root314Tail4 winner.bound = root314Candidates3 := by
  unfold candidates
  have row : edge 2693 = (SortedNatTree.contains FastRows.row2693) := by
    change Data.row2693.contains = (SortedNatTree.contains FastRows.row2693)
    exact FastRows.row2693_checked.symm
  rw [row]
  decide
theorem root314Step3 : scan edge (search edge 3) 4 [2099] 2099 root314Tail3 winner =
    scan edge (search edge 3) 4 [2099] 2099 root314Tail4 winner := by
  rw [root314Tail3, scan, if_neg (by decide), root314Candidates3_checked]
  change scan edge (search edge 3) 4 [2099] 2099 root314Tail4
    (if 3 ≤ root314Candidates3.length then search edge 3 [2099, 2693] root314Candidates3 4792 winner else winner) = _
  rw [if_neg (by decide)]
def root314Candidates4 : List Nat := [7841]
set_option maxRecDepth 8192 in
theorem root314Candidates4_checked : candidates edge 4 2099 2843 root314Tail5 winner.bound = root314Candidates4 := by
  unfold candidates
  have row : edge 2843 = (SortedNatTree.contains FastRows.row2843) := by
    change Data.row2843.contains = (SortedNatTree.contains FastRows.row2843)
    exact FastRows.row2843_checked.symm
  rw [row]
  decide
theorem root314Step4 : scan edge (search edge 3) 4 [2099] 2099 root314Tail4 winner =
    scan edge (search edge 3) 4 [2099] 2099 root314Tail5 winner := by
  rw [root314Tail4, scan, if_neg (by decide), root314Candidates4_checked]
  change scan edge (search edge 3) 4 [2099] 2099 root314Tail5
    (if 3 ≤ root314Candidates4.length then search edge 3 [2099, 2843] root314Candidates4 4942 winner else winner) = _
  rw [if_neg (by decide)]
def root314Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root314Candidates5_checked : candidates edge 4 2099 3083 root314Tail6 winner.bound = root314Candidates5 := by
  unfold candidates
  have row : edge 3083 = (SortedNatTree.contains FastRows.row3083) := by
    change Data.row3083.contains = (SortedNatTree.contains FastRows.row3083)
    exact FastRows.row3083_checked.symm
  rw [row]
  decide
theorem root314Step5 : scan edge (search edge 3) 4 [2099] 2099 root314Tail5 winner =
    scan edge (search edge 3) 4 [2099] 2099 root314Tail6 winner := by
  rw [root314Tail5, scan, if_neg (by decide), root314Candidates5_checked]
  change scan edge (search edge 3) 4 [2099] 2099 root314Tail6
    (if 3 ≤ root314Candidates5.length then search edge 3 [2099, 3083] root314Candidates5 5182 winner else winner) = _
  rw [if_neg (by decide)]
def root314Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root314Candidates6_checked : candidates edge 4 2099 3527 root314Tail7 winner.bound = root314Candidates6 := by
  unfold candidates
  have row : edge 3527 = (SortedNatTree.contains FastRows.row3527) := by
    change Data.row3527.contains = (SortedNatTree.contains FastRows.row3527)
    exact FastRows.row3527_checked.symm
  rw [row]
  decide
theorem root314Step6 : scan edge (search edge 3) 4 [2099] 2099 root314Tail6 winner =
    scan edge (search edge 3) 4 [2099] 2099 root314Tail7 winner := by
  rw [root314Tail6, scan, if_neg (by decide), root314Candidates6_checked]
  change scan edge (search edge 3) 4 [2099] 2099 root314Tail7
    (if 3 ≤ root314Candidates6.length then search edge 3 [2099, 3527] root314Candidates6 5626 winner else winner) = _
  rw [if_neg (by decide)]
def root314Candidates7 : List Nat := [8093]
set_option maxRecDepth 8192 in
theorem root314Candidates7_checked : candidates edge 4 2099 4241 root314Tail8 winner.bound = root314Candidates7 := by
  unfold candidates
  have row : edge 4241 = (SortedNatTree.contains FastRows.row4241) := by
    change Data.row4241.contains = (SortedNatTree.contains FastRows.row4241)
    exact FastRows.row4241_checked.symm
  rw [row]
  decide
theorem root314Step7 : scan edge (search edge 3) 4 [2099] 2099 root314Tail7 winner =
    scan edge (search edge 3) 4 [2099] 2099 root314Tail8 winner := by
  rw [root314Tail7, scan, if_neg (by decide), root314Candidates7_checked]
  change scan edge (search edge 3) 4 [2099] 2099 root314Tail8
    (if 3 ≤ root314Candidates7.length then search edge 3 [2099, 4241] root314Candidates7 6340 winner else winner) = _
  rw [if_neg (by decide)]
def root314Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root314Candidates8_checked : candidates edge 4 2099 5279 root314Tail9 winner.bound = root314Candidates8 := by
  unfold candidates
  have row : edge 5279 = (SortedNatTree.contains FastRows.row5279) := by
    change Data.row5279.contains = (SortedNatTree.contains FastRows.row5279)
    exact FastRows.row5279_checked.symm
  rw [row]
  decide
theorem root314Step8 : scan edge (search edge 3) 4 [2099] 2099 root314Tail8 winner =
    scan edge (search edge 3) 4 [2099] 2099 root314Tail9 winner := by
  rw [root314Tail8, scan, if_neg (by decide), root314Candidates8_checked]
  change scan edge (search edge 3) 4 [2099] 2099 root314Tail9
    (if 3 ≤ root314Candidates8.length then search edge 3 [2099, 5279] root314Candidates8 7378 winner else winner) = _
  rw [if_neg (by decide)]
def root314Candidates9 : List Nat := [9887]
set_option maxRecDepth 8192 in
theorem root314Candidates9_checked : candidates edge 4 2099 5717 root314Tail10 winner.bound = root314Candidates9 := by
  unfold candidates
  have row : edge 5717 = (SortedNatTree.contains FastRows.row5717) := by
    change Data.row5717.contains = (SortedNatTree.contains FastRows.row5717)
    exact FastRows.row5717_checked.symm
  rw [row]
  decide
theorem root314Step9 : scan edge (search edge 3) 4 [2099] 2099 root314Tail9 winner =
    scan edge (search edge 3) 4 [2099] 2099 root314Tail10 winner := by
  rw [root314Tail9, scan, if_neg (by decide), root314Candidates9_checked]
  change scan edge (search edge 3) 4 [2099] 2099 root314Tail10
    (if 3 ≤ root314Candidates9.length then search edge 3 [2099, 5717] root314Candidates9 7816 winner else winner) = _
  rw [if_neg (by decide)]
theorem root314Scan10 : scan edge (search edge 3) 4 [2099] 2099 root314Tail10 winner = winner := by rw [root314Tail10, scan, if_pos (by decide)]
theorem root314Scan9 : scan edge (search edge 3) 4 [2099] 2099 root314Tail9 winner = winner :=
  (root314Step9).trans root314Scan10
theorem root314Scan8 : scan edge (search edge 3) 4 [2099] 2099 root314Tail8 winner = winner :=
  (root314Step8).trans root314Scan9
theorem root314Scan7 : scan edge (search edge 3) 4 [2099] 2099 root314Tail7 winner = winner :=
  (root314Step7).trans root314Scan8
theorem root314Scan6 : scan edge (search edge 3) 4 [2099] 2099 root314Tail6 winner = winner :=
  (root314Step6).trans root314Scan7
theorem root314Scan5 : scan edge (search edge 3) 4 [2099] 2099 root314Tail5 winner = winner :=
  (root314Step5).trans root314Scan6
theorem root314Scan4 : scan edge (search edge 3) 4 [2099] 2099 root314Tail4 winner = winner :=
  (root314Step4).trans root314Scan5
theorem root314Scan3 : scan edge (search edge 3) 4 [2099] 2099 root314Tail3 winner = winner :=
  (root314Step3).trans root314Scan4
theorem root314Scan2 : scan edge (search edge 3) 4 [2099] 2099 root314Tail2 winner = winner :=
  (root314Step2).trans root314Scan3
theorem root314Scan1 : scan edge (search edge 3) 4 [2099] 2099 root314Tail1 winner = winner :=
  (root314Step1).trans root314Scan2
theorem root314Scan0 : scan edge (search edge 3) 4 [2099] 2099 root314Tail0 winner = winner :=
  (root314Step0).trans root314Scan1
set_option maxRecDepth 4096 in
theorem root314Branch : search edge 4 [2099] root314Candidates 2099 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2099] 2099 root314Tail0 winner = winner
  exact root314Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
