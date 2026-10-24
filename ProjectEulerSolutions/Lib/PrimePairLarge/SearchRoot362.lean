import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root362Candidates : List Nat := Data.row2467
set_option maxRecDepth 8192 in
theorem root362Candidates_checked : candidates edge 5 0 2467 tail363 winner.bound = root362Candidates := by
  apply candidates_eq_row edge 5 0 2467 tail363 root362Candidates winner.bound tail363_ordered
  · decide
  · decide
  · funext q
    rfl
def root362Tail44 : List Nat := []
def root362Tail43 : List Nat := 23371 :: root362Tail44
def root362Tail42 : List Nat := 22741 :: root362Tail43
def root362Tail41 : List Nat := 22501 :: root362Tail42
def root362Tail40 : List Nat := 22447 :: root362Tail41
def root362Tail39 : List Nat := 22027 :: root362Tail40
def root362Tail38 : List Nat := 19927 :: root362Tail39
def root362Tail37 : List Nat := 19801 :: root362Tail38
def root362Tail36 : List Nat := 17659 :: root362Tail37
def root362Tail35 : List Nat := 17539 :: root362Tail36
def root362Tail34 : List Nat := 16741 :: root362Tail35
def root362Tail33 : List Nat := 16417 :: root362Tail34
def root362Tail32 : List Nat := 15823 :: root362Tail33
def root362Tail31 : List Nat := 15349 :: root362Tail32
def root362Tail30 : List Nat := 14983 :: root362Tail31
def root362Tail29 : List Nat := 14653 :: root362Tail30
def root362Tail28 : List Nat := 14437 :: root362Tail29
def root362Tail27 : List Nat := 13921 :: root362Tail28
def root362Tail26 : List Nat := 13627 :: root362Tail27
def root362Tail25 : List Nat := 13477 :: root362Tail26
def root362Tail24 : List Nat := 12919 :: root362Tail25
def root362Tail23 : List Nat := 11821 :: root362Tail24
def root362Tail22 : List Nat := 11329 :: root362Tail23
def root362Tail21 : List Nat := 10957 :: root362Tail22
def root362Tail20 : List Nat := 10861 :: root362Tail21
def root362Tail19 : List Nat := 9931 :: root362Tail20
def root362Tail18 : List Nat := 9769 :: root362Tail19
def root362Tail17 : List Nat := 9601 :: root362Tail18
def root362Tail16 : List Nat := 8923 :: root362Tail17
def root362Tail15 : List Nat := 8377 :: root362Tail16
def root362Tail14 : List Nat := 8167 :: root362Tail15
def root362Tail13 : List Nat := 7963 :: root362Tail14
def root362Tail12 : List Nat := 7621 :: root362Tail13
def root362Tail11 : List Nat := 6529 :: root362Tail12
def root362Tail10 : List Nat := 6277 :: root362Tail11
def root362Tail9 : List Nat := 6037 :: root362Tail10
def root362Tail8 : List Nat := 5689 :: root362Tail9
def root362Tail7 : List Nat := 5683 :: root362Tail8
def root362Tail6 : List Nat := 5407 :: root362Tail7
def root362Tail5 : List Nat := 4861 :: root362Tail6
def root362Tail4 : List Nat := 4723 :: root362Tail5
def root362Tail3 : List Nat := 4051 :: root362Tail4
def root362Tail2 : List Nat := 3217 :: root362Tail3
def root362Tail1 : List Nat := 2551 :: root362Tail2
def root362Tail0 : List Nat := 2539 :: root362Tail1
def root362Candidates0 : List Nat := [4723]
set_option maxRecDepth 8192 in
theorem root362Candidates0_checked : candidates edge 4 2467 2539 root362Tail1 winner.bound = root362Candidates0 := by
  unfold candidates
  have row : edge 2539 = (SortedNatTree.contains FastRows.row2539) := by
    change Data.row2539.contains = (SortedNatTree.contains FastRows.row2539)
    exact FastRows.row2539_checked.symm
  rw [row]
  decide
theorem root362Step0 : scan edge (search edge 3) 4 [2467] 2467 root362Tail0 winner =
    scan edge (search edge 3) 4 [2467] 2467 root362Tail1 winner := by
  rw [root362Tail0, scan, if_neg (by decide), root362Candidates0_checked]
  change scan edge (search edge 3) 4 [2467] 2467 root362Tail1
    (if 3 ≤ root362Candidates0.length then search edge 3 [2467, 2539] root362Candidates0 5006 winner else winner) = _
  rw [if_neg (by decide)]
def root362Candidates1 : List Nat := [6037]
set_option maxRecDepth 8192 in
theorem root362Candidates1_checked : candidates edge 4 2467 2551 root362Tail2 winner.bound = root362Candidates1 := by
  unfold candidates
  have row : edge 2551 = (SortedNatTree.contains FastRows.row2551) := by
    change Data.row2551.contains = (SortedNatTree.contains FastRows.row2551)
    exact FastRows.row2551_checked.symm
  rw [row]
  decide
theorem root362Step1 : scan edge (search edge 3) 4 [2467] 2467 root362Tail1 winner =
    scan edge (search edge 3) 4 [2467] 2467 root362Tail2 winner := by
  rw [root362Tail1, scan, if_neg (by decide), root362Candidates1_checked]
  change scan edge (search edge 3) 4 [2467] 2467 root362Tail2
    (if 3 ≤ root362Candidates1.length then search edge 3 [2467, 2551] root362Candidates1 5018 winner else winner) = _
  rw [if_neg (by decide)]
def root362Candidates2 : List Nat := [10861]
set_option maxRecDepth 8192 in
theorem root362Candidates2_checked : candidates edge 4 2467 3217 root362Tail3 winner.bound = root362Candidates2 := by
  unfold candidates
  have row : edge 3217 = (SortedNatTree.contains FastRows.row3217) := by
    change Data.row3217.contains = (SortedNatTree.contains FastRows.row3217)
    exact FastRows.row3217_checked.symm
  rw [row]
  decide
theorem root362Step2 : scan edge (search edge 3) 4 [2467] 2467 root362Tail2 winner =
    scan edge (search edge 3) 4 [2467] 2467 root362Tail3 winner := by
  rw [root362Tail2, scan, if_neg (by decide), root362Candidates2_checked]
  change scan edge (search edge 3) 4 [2467] 2467 root362Tail3
    (if 3 ≤ root362Candidates2.length then search edge 3 [2467, 3217] root362Candidates2 5684 winner else winner) = _
  rw [if_neg (by decide)]
def root362Candidates3 : List Nat := [10957]
set_option maxRecDepth 8192 in
theorem root362Candidates3_checked : candidates edge 4 2467 4051 root362Tail4 winner.bound = root362Candidates3 := by
  unfold candidates
  have row : edge 4051 = (SortedNatTree.contains FastRows.row4051) := by
    change Data.row4051.contains = (SortedNatTree.contains FastRows.row4051)
    exact FastRows.row4051_checked.symm
  rw [row]
  decide
theorem root362Step3 : scan edge (search edge 3) 4 [2467] 2467 root362Tail3 winner =
    scan edge (search edge 3) 4 [2467] 2467 root362Tail4 winner := by
  rw [root362Tail3, scan, if_neg (by decide), root362Candidates3_checked]
  change scan edge (search edge 3) 4 [2467] 2467 root362Tail4
    (if 3 ≤ root362Candidates3.length then search edge 3 [2467, 4051] root362Candidates3 6518 winner else winner) = _
  rw [if_neg (by decide)]
def root362Candidates4 : List Nat := [8923]
set_option maxRecDepth 8192 in
theorem root362Candidates4_checked : candidates edge 4 2467 4723 root362Tail5 winner.bound = root362Candidates4 := by
  unfold candidates
  have row : edge 4723 = (SortedNatTree.contains FastRows.row4723) := by
    change Data.row4723.contains = (SortedNatTree.contains FastRows.row4723)
    exact FastRows.row4723_checked.symm
  rw [row]
  decide
theorem root362Step4 : scan edge (search edge 3) 4 [2467] 2467 root362Tail4 winner =
    scan edge (search edge 3) 4 [2467] 2467 root362Tail5 winner := by
  rw [root362Tail4, scan, if_neg (by decide), root362Candidates4_checked]
  change scan edge (search edge 3) 4 [2467] 2467 root362Tail5
    (if 3 ≤ root362Candidates4.length then search edge 3 [2467, 4723] root362Candidates4 7190 winner else winner) = _
  rw [if_neg (by decide)]
def root362Candidates5 : List Nat := [5689]
set_option maxRecDepth 8192 in
theorem root362Candidates5_checked : candidates edge 4 2467 4861 root362Tail6 winner.bound = root362Candidates5 := by
  unfold candidates
  have row : edge 4861 = (SortedNatTree.contains FastRows.row4861) := by
    change Data.row4861.contains = (SortedNatTree.contains FastRows.row4861)
    exact FastRows.row4861_checked.symm
  rw [row]
  decide
theorem root362Step5 : scan edge (search edge 3) 4 [2467] 2467 root362Tail5 winner =
    scan edge (search edge 3) 4 [2467] 2467 root362Tail6 winner := by
  rw [root362Tail5, scan, if_neg (by decide), root362Candidates5_checked]
  change scan edge (search edge 3) 4 [2467] 2467 root362Tail6
    (if 3 ≤ root362Candidates5.length then search edge 3 [2467, 4861] root362Candidates5 7328 winner else winner) = _
  rw [if_neg (by decide)]
def root362Candidates6 : List Nat := [10957, 17659]
set_option maxRecDepth 8192 in
theorem root362Candidates6_checked : candidates edge 4 2467 5407 root362Tail7 winner.bound = root362Candidates6 := by
  unfold candidates
  have row : edge 5407 = (SortedNatTree.contains FastRows.row5407) := by
    change Data.row5407.contains = (SortedNatTree.contains FastRows.row5407)
    exact FastRows.row5407_checked.symm
  rw [row]
  decide
theorem root362Step6 : scan edge (search edge 3) 4 [2467] 2467 root362Tail6 winner =
    scan edge (search edge 3) 4 [2467] 2467 root362Tail7 winner := by
  rw [root362Tail6, scan, if_neg (by decide), root362Candidates6_checked]
  change scan edge (search edge 3) 4 [2467] 2467 root362Tail7
    (if 3 ≤ root362Candidates6.length then search edge 3 [2467, 5407] root362Candidates6 7874 winner else winner) = _
  rw [if_neg (by decide)]
def root362Candidates7 : List Nat := [15823]
set_option maxRecDepth 8192 in
theorem root362Candidates7_checked : candidates edge 4 2467 5683 root362Tail8 winner.bound = root362Candidates7 := by
  unfold candidates
  have row : edge 5683 = (SortedNatTree.contains FastRows.row5683) := by
    change Data.row5683.contains = (SortedNatTree.contains FastRows.row5683)
    exact FastRows.row5683_checked.symm
  rw [row]
  decide
theorem root362Step7 : scan edge (search edge 3) 4 [2467] 2467 root362Tail7 winner =
    scan edge (search edge 3) 4 [2467] 2467 root362Tail8 winner := by
  rw [root362Tail7, scan, if_neg (by decide), root362Candidates7_checked]
  change scan edge (search edge 3) 4 [2467] 2467 root362Tail8
    (if 3 ≤ root362Candidates7.length then search edge 3 [2467, 5683] root362Candidates7 8150 winner else winner) = _
  rw [if_neg (by decide)]
def root362Candidates8 : List Nat := [6277, 15349]
set_option maxRecDepth 8192 in
theorem root362Candidates8_checked : candidates edge 4 2467 5689 root362Tail9 winner.bound = root362Candidates8 := by
  unfold candidates
  have row : edge 5689 = (SortedNatTree.contains FastRows.row5689) := by
    change Data.row5689.contains = (SortedNatTree.contains FastRows.row5689)
    exact FastRows.row5689_checked.symm
  rw [row]
  decide
theorem root362Step8 : scan edge (search edge 3) 4 [2467] 2467 root362Tail8 winner =
    scan edge (search edge 3) 4 [2467] 2467 root362Tail9 winner := by
  rw [root362Tail8, scan, if_neg (by decide), root362Candidates8_checked]
  change scan edge (search edge 3) 4 [2467] 2467 root362Tail9
    (if 3 ≤ root362Candidates8.length then search edge 3 [2467, 5689] root362Candidates8 8156 winner else winner) = _
  rw [if_neg (by decide)]
theorem root362Scan9 : scan edge (search edge 3) 4 [2467] 2467 root362Tail9 winner = winner := by rw [root362Tail9, scan, if_pos (by decide)]
theorem root362Scan8 : scan edge (search edge 3) 4 [2467] 2467 root362Tail8 winner = winner :=
  (root362Step8).trans root362Scan9
theorem root362Scan7 : scan edge (search edge 3) 4 [2467] 2467 root362Tail7 winner = winner :=
  (root362Step7).trans root362Scan8
theorem root362Scan6 : scan edge (search edge 3) 4 [2467] 2467 root362Tail6 winner = winner :=
  (root362Step6).trans root362Scan7
theorem root362Scan5 : scan edge (search edge 3) 4 [2467] 2467 root362Tail5 winner = winner :=
  (root362Step5).trans root362Scan6
theorem root362Scan4 : scan edge (search edge 3) 4 [2467] 2467 root362Tail4 winner = winner :=
  (root362Step4).trans root362Scan5
theorem root362Scan3 : scan edge (search edge 3) 4 [2467] 2467 root362Tail3 winner = winner :=
  (root362Step3).trans root362Scan4
theorem root362Scan2 : scan edge (search edge 3) 4 [2467] 2467 root362Tail2 winner = winner :=
  (root362Step2).trans root362Scan3
theorem root362Scan1 : scan edge (search edge 3) 4 [2467] 2467 root362Tail1 winner = winner :=
  (root362Step1).trans root362Scan2
theorem root362Scan0 : scan edge (search edge 3) 4 [2467] 2467 root362Tail0 winner = winner :=
  (root362Step0).trans root362Scan1
set_option maxRecDepth 4096 in
theorem root362Branch : search edge 4 [2467] root362Candidates 2467 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2467] 2467 root362Tail0 winner = winner
  exact root362Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
