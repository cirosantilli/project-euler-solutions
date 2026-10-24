import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root220Candidates : List Nat := Data.row1409
set_option maxRecDepth 8192 in
theorem root220Candidates_checked : candidates edge 5 0 1409 tail221 winner.bound = root220Candidates := by
  apply candidates_eq_row edge 5 0 1409 tail221 root220Candidates winner.bound tail221_ordered
  · decide
  · decide
  · funext q
    rfl
def root220Tail48 : List Nat := []
def root220Tail47 : List Nat := 23819 :: root220Tail48
def root220Tail46 : List Nat := 23609 :: root220Tail47
def root220Tail45 : List Nat := 23087 :: root220Tail46
def root220Tail44 : List Nat := 23039 :: root220Tail45
def root220Tail43 : List Nat := 22811 :: root220Tail44
def root220Tail42 : List Nat := 22349 :: root220Tail43
def root220Tail41 : List Nat := 22037 :: root220Tail42
def root220Tail40 : List Nat := 21491 :: root220Tail41
def root220Tail39 : List Nat := 20981 :: root220Tail40
def root220Tail38 : List Nat := 20903 :: root220Tail39
def root220Tail37 : List Nat := 20399 :: root220Tail38
def root220Tail36 : List Nat := 20357 :: root220Tail37
def root220Tail35 : List Nat := 20201 :: root220Tail36
def root220Tail34 : List Nat := 19319 :: root220Tail35
def root220Tail33 : List Nat := 18731 :: root220Tail34
def root220Tail32 : List Nat := 18257 :: root220Tail33
def root220Tail31 : List Nat := 18191 :: root220Tail32
def root220Tail30 : List Nat := 17939 :: root220Tail31
def root220Tail29 : List Nat := 17681 :: root220Tail30
def root220Tail28 : List Nat := 16427 :: root220Tail29
def root220Tail27 : List Nat := 15461 :: root220Tail28
def root220Tail26 : List Nat := 15083 :: root220Tail27
def root220Tail25 : List Nat := 14327 :: root220Tail26
def root220Tail24 : List Nat := 14303 :: root220Tail25
def root220Tail23 : List Nat := 13151 :: root220Tail24
def root220Tail22 : List Nat := 12809 :: root220Tail23
def root220Tail21 : List Nat := 12713 :: root220Tail22
def root220Tail20 : List Nat := 11633 :: root220Tail21
def root220Tail19 : List Nat := 11597 :: root220Tail20
def root220Tail18 : List Nat := 9857 :: root220Tail19
def root220Tail17 : List Nat := 9689 :: root220Tail18
def root220Tail16 : List Nat := 9473 :: root220Tail17
def root220Tail15 : List Nat := 9221 :: root220Tail16
def root220Tail14 : List Nat := 8681 :: root220Tail15
def root220Tail13 : List Nat := 7907 :: root220Tail14
def root220Tail12 : List Nat := 7817 :: root220Tail13
def root220Tail11 : List Nat := 7247 :: root220Tail12
def root220Tail10 : List Nat := 6983 :: root220Tail11
def root220Tail9 : List Nat := 6857 :: root220Tail10
def root220Tail8 : List Nat := 6521 :: root220Tail9
def root220Tail7 : List Nat := 6029 :: root220Tail8
def root220Tail6 : List Nat := 5387 :: root220Tail7
def root220Tail5 : List Nat := 5231 :: root220Tail6
def root220Tail4 : List Nat := 4643 :: root220Tail5
def root220Tail3 : List Nat := 4517 :: root220Tail4
def root220Tail2 : List Nat := 4013 :: root220Tail3
def root220Tail1 : List Nat := 3329 :: root220Tail2
def root220Tail0 : List Nat := 1499 :: root220Tail1
def root220Candidates0 : List Nat := [4013, 6521, 9221]
set_option maxRecDepth 8192 in
theorem root220Candidates0_checked : candidates edge 4 1409 1499 root220Tail1 winner.bound = root220Candidates0 := by
  unfold candidates
  have row : edge 1499 = (SortedNatTree.contains FastRows.row1499) := by
    change Data.row1499.contains = (SortedNatTree.contains FastRows.row1499)
    exact FastRows.row1499_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root220Child0 : search edge 3 [1409, 1499] root220Candidates0 2908 winner = winner := by decide
theorem root220Step0 : scan edge (search edge 3) 4 [1409] 1409 root220Tail0 winner =
    scan edge (search edge 3) 4 [1409] 1409 root220Tail1 winner := by
  rw [root220Tail0, scan, if_neg (by decide), root220Candidates0_checked]
  change scan edge (search edge 3) 4 [1409] 1409 root220Tail1
    (if 3 ≤ root220Candidates0.length then search edge 3 [1409, 1499] root220Candidates0 2908 winner else winner) = _
  rw [if_pos (by decide), root220Child0]
def root220Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root220Candidates1_checked : candidates edge 4 1409 3329 root220Tail2 winner.bound = root220Candidates1 := by
  unfold candidates
  have row : edge 3329 = (SortedNatTree.contains FastRows.row3329) := by
    change Data.row3329.contains = (SortedNatTree.contains FastRows.row3329)
    exact FastRows.row3329_checked.symm
  rw [row]
  decide
theorem root220Step1 : scan edge (search edge 3) 4 [1409] 1409 root220Tail1 winner =
    scan edge (search edge 3) 4 [1409] 1409 root220Tail2 winner := by
  rw [root220Tail1, scan, if_neg (by decide), root220Candidates1_checked]
  change scan edge (search edge 3) 4 [1409] 1409 root220Tail2
    (if 3 ≤ root220Candidates1.length then search edge 3 [1409, 3329] root220Candidates1 4738 winner else winner) = _
  rw [if_neg (by decide)]
def root220Candidates2 : List Nat := [4643, 16427]
set_option maxRecDepth 8192 in
theorem root220Candidates2_checked : candidates edge 4 1409 4013 root220Tail3 winner.bound = root220Candidates2 := by
  unfold candidates
  have row : edge 4013 = (SortedNatTree.contains FastRows.row4013) := by
    change Data.row4013.contains = (SortedNatTree.contains FastRows.row4013)
    exact FastRows.row4013_checked.symm
  rw [row]
  decide
theorem root220Step2 : scan edge (search edge 3) 4 [1409] 1409 root220Tail2 winner =
    scan edge (search edge 3) 4 [1409] 1409 root220Tail3 winner := by
  rw [root220Tail2, scan, if_neg (by decide), root220Candidates2_checked]
  change scan edge (search edge 3) 4 [1409] 1409 root220Tail3
    (if 3 ≤ root220Candidates2.length then search edge 3 [1409, 4013] root220Candidates2 5422 winner else winner) = _
  rw [if_neg (by decide)]
def root220Candidates3 : List Nat := [5387, 9689]
set_option maxRecDepth 8192 in
theorem root220Candidates3_checked : candidates edge 4 1409 4517 root220Tail4 winner.bound = root220Candidates3 := by
  unfold candidates
  have row : edge 4517 = (SortedNatTree.contains FastRows.row4517) := by
    change Data.row4517.contains = (SortedNatTree.contains FastRows.row4517)
    exact FastRows.row4517_checked.symm
  rw [row]
  decide
theorem root220Step3 : scan edge (search edge 3) 4 [1409] 1409 root220Tail3 winner =
    scan edge (search edge 3) 4 [1409] 1409 root220Tail4 winner := by
  rw [root220Tail3, scan, if_neg (by decide), root220Candidates3_checked]
  change scan edge (search edge 3) 4 [1409] 1409 root220Tail4
    (if 3 ≤ root220Candidates3.length then search edge 3 [1409, 4517] root220Candidates3 5926 winner else winner) = _
  rw [if_neg (by decide)]
def root220Candidates4 : List Nat := [5231, 14303, 15083, 17939]
set_option maxRecDepth 8192 in
theorem root220Candidates4_checked : candidates edge 4 1409 4643 root220Tail5 winner.bound = root220Candidates4 := by
  unfold candidates
  have row : edge 4643 = (SortedNatTree.contains FastRows.row4643) := by
    change Data.row4643.contains = (SortedNatTree.contains FastRows.row4643)
    exact FastRows.row4643_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root220Child4 : search edge 3 [1409, 4643] root220Candidates4 6052 winner = winner := by decide
theorem root220Step4 : scan edge (search edge 3) 4 [1409] 1409 root220Tail4 winner =
    scan edge (search edge 3) 4 [1409] 1409 root220Tail5 winner := by
  rw [root220Tail4, scan, if_neg (by decide), root220Candidates4_checked]
  change scan edge (search edge 3) 4 [1409] 1409 root220Tail5
    (if 3 ≤ root220Candidates4.length then search edge 3 [1409, 4643] root220Candidates4 6052 winner else winner) = _
  rw [if_pos (by decide), root220Child4]
def root220Candidates5 : List Nat := [5387]
set_option maxRecDepth 8192 in
theorem root220Candidates5_checked : candidates edge 4 1409 5231 root220Tail6 winner.bound = root220Candidates5 := by
  unfold candidates
  have row : edge 5231 = (SortedNatTree.contains FastRows.row5231) := by
    change Data.row5231.contains = (SortedNatTree.contains FastRows.row5231)
    exact FastRows.row5231_checked.symm
  rw [row]
  decide
theorem root220Step5 : scan edge (search edge 3) 4 [1409] 1409 root220Tail5 winner =
    scan edge (search edge 3) 4 [1409] 1409 root220Tail6 winner := by
  rw [root220Tail5, scan, if_neg (by decide), root220Candidates5_checked]
  change scan edge (search edge 3) 4 [1409] 1409 root220Tail6
    (if 3 ≤ root220Candidates5.length then search edge 3 [1409, 5231] root220Candidates5 6640 winner else winner) = _
  rw [if_neg (by decide)]
def root220Candidates6 : List Nat := [9689, 9857, 14327, 15083]
set_option maxRecDepth 8192 in
theorem root220Candidates6_checked : candidates edge 4 1409 5387 root220Tail7 winner.bound = root220Candidates6 := by
  unfold candidates
  have row : edge 5387 = (SortedNatTree.contains FastRows.row5387) := by
    change Data.row5387.contains = (SortedNatTree.contains FastRows.row5387)
    exact FastRows.row5387_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root220Child6 : search edge 3 [1409, 5387] root220Candidates6 6796 winner = winner := by decide
theorem root220Step6 : scan edge (search edge 3) 4 [1409] 1409 root220Tail6 winner =
    scan edge (search edge 3) 4 [1409] 1409 root220Tail7 winner := by
  rw [root220Tail6, scan, if_neg (by decide), root220Candidates6_checked]
  change scan edge (search edge 3) 4 [1409] 1409 root220Tail7
    (if 3 ≤ root220Candidates6.length then search edge 3 [1409, 5387] root220Candidates6 6796 winner else winner) = _
  rw [if_pos (by decide), root220Child6]
def root220Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root220Candidates7_checked : candidates edge 4 1409 6029 root220Tail8 winner.bound = root220Candidates7 := by
  unfold candidates
  have row : edge 6029 = (SortedNatTree.contains FastRows.row6029) := by
    change Data.row6029.contains = (SortedNatTree.contains FastRows.row6029)
    exact FastRows.row6029_checked.symm
  rw [row]
  decide
theorem root220Step7 : scan edge (search edge 3) 4 [1409] 1409 root220Tail7 winner =
    scan edge (search edge 3) 4 [1409] 1409 root220Tail8 winner := by
  rw [root220Tail7, scan, if_neg (by decide), root220Candidates7_checked]
  change scan edge (search edge 3) 4 [1409] 1409 root220Tail8
    (if 3 ≤ root220Candidates7.length then search edge 3 [1409, 6029] root220Candidates7 7438 winner else winner) = _
  rw [if_neg (by decide)]
theorem root220Scan8 : scan edge (search edge 3) 4 [1409] 1409 root220Tail8 winner = winner := by rw [root220Tail8, scan, if_pos (by decide)]
theorem root220Scan7 : scan edge (search edge 3) 4 [1409] 1409 root220Tail7 winner = winner :=
  (root220Step7).trans root220Scan8
theorem root220Scan6 : scan edge (search edge 3) 4 [1409] 1409 root220Tail6 winner = winner :=
  (root220Step6).trans root220Scan7
theorem root220Scan5 : scan edge (search edge 3) 4 [1409] 1409 root220Tail5 winner = winner :=
  (root220Step5).trans root220Scan6
theorem root220Scan4 : scan edge (search edge 3) 4 [1409] 1409 root220Tail4 winner = winner :=
  (root220Step4).trans root220Scan5
theorem root220Scan3 : scan edge (search edge 3) 4 [1409] 1409 root220Tail3 winner = winner :=
  (root220Step3).trans root220Scan4
theorem root220Scan2 : scan edge (search edge 3) 4 [1409] 1409 root220Tail2 winner = winner :=
  (root220Step2).trans root220Scan3
theorem root220Scan1 : scan edge (search edge 3) 4 [1409] 1409 root220Tail1 winner = winner :=
  (root220Step1).trans root220Scan2
theorem root220Scan0 : scan edge (search edge 3) 4 [1409] 1409 root220Tail0 winner = winner :=
  (root220Step0).trans root220Scan1
set_option maxRecDepth 4096 in
theorem root220Branch : search edge 4 [1409] root220Candidates 1409 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1409] 1409 root220Tail0 winner = winner
  exact root220Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
