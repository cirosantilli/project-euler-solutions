import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root265Candidates : List Nat := Data.row1721
set_option maxRecDepth 8192 in
theorem root265Candidates_checked : candidates edge 5 0 1721 tail266 winner.bound = root265Candidates := by
  apply candidates_eq_row edge 5 0 1721 tail266 root265Candidates winner.bound tail266_ordered
  · decide
  · decide
  · funext q
    rfl
def root265Tail48 : List Nat := []
def root265Tail47 : List Nat := 22697 :: root265Tail48
def root265Tail46 : List Nat := 21767 :: root265Tail47
def root265Tail45 : List Nat := 21557 :: root265Tail46
def root265Tail44 : List Nat := 20549 :: root265Tail45
def root265Tail43 : List Nat := 20177 :: root265Tail44
def root265Tail42 : List Nat := 19463 :: root265Tail43
def root265Tail41 : List Nat := 18341 :: root265Tail42
def root265Tail40 : List Nat := 18329 :: root265Tail41
def root265Tail39 : List Nat := 16871 :: root265Tail40
def root265Tail38 : List Nat := 16349 :: root265Tail39
def root265Tail37 : List Nat := 16223 :: root265Tail38
def root265Tail36 : List Nat := 15887 :: root265Tail37
def root265Tail35 : List Nat := 15287 :: root265Tail36
def root265Tail34 : List Nat := 15149 :: root265Tail35
def root265Tail33 : List Nat := 15137 :: root265Tail34
def root265Tail32 : List Nat := 15107 :: root265Tail33
def root265Tail31 : List Nat := 14951 :: root265Tail32
def root265Tail30 : List Nat := 14771 :: root265Tail31
def root265Tail29 : List Nat := 14753 :: root265Tail30
def root265Tail28 : List Nat := 14717 :: root265Tail29
def root265Tail27 : List Nat := 14561 :: root265Tail28
def root265Tail26 : List Nat := 12107 :: root265Tail27
def root265Tail25 : List Nat := 11579 :: root265Tail26
def root265Tail24 : List Nat := 10937 :: root265Tail25
def root265Tail23 : List Nat := 10079 :: root265Tail24
def root265Tail22 : List Nat := 9749 :: root265Tail23
def root265Tail21 : List Nat := 9677 :: root265Tail22
def root265Tail20 : List Nat := 8627 :: root265Tail21
def root265Tail19 : List Nat := 8573 :: root265Tail20
def root265Tail18 : List Nat := 8429 :: root265Tail19
def root265Tail17 : List Nat := 8387 :: root265Tail18
def root265Tail16 : List Nat := 8273 :: root265Tail17
def root265Tail15 : List Nat := 8231 :: root265Tail16
def root265Tail14 : List Nat := 8219 :: root265Tail15
def root265Tail13 : List Nat := 8039 :: root265Tail14
def root265Tail12 : List Nat := 7829 :: root265Tail13
def root265Tail11 : List Nat := 7589 :: root265Tail12
def root265Tail10 : List Nat := 7283 :: root265Tail11
def root265Tail9 : List Nat := 6779 :: root265Tail10
def root265Tail8 : List Nat := 6389 :: root265Tail9
def root265Tail7 : List Nat := 4679 :: root265Tail8
def root265Tail6 : List Nat := 3557 :: root265Tail7
def root265Tail5 : List Nat := 3359 :: root265Tail6
def root265Tail4 : List Nat := 3347 :: root265Tail5
def root265Tail3 : List Nat := 2621 :: root265Tail4
def root265Tail2 : List Nat := 2411 :: root265Tail3
def root265Tail1 : List Nat := 2297 :: root265Tail2
def root265Tail0 : List Nat := 2213 :: root265Tail1
def root265Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root265Candidates0_checked : candidates edge 4 1721 2213 root265Tail1 winner.bound = root265Candidates0 := by
  unfold candidates
  have row : edge 2213 = (SortedNatTree.contains FastRows.row2213) := by
    change Data.row2213.contains = (SortedNatTree.contains FastRows.row2213)
    exact FastRows.row2213_checked.symm
  rw [row]
  decide
theorem root265Step0 : scan edge (search edge 3) 4 [1721] 1721 root265Tail0 winner =
    scan edge (search edge 3) 4 [1721] 1721 root265Tail1 winner := by
  rw [root265Tail0, scan, if_neg (by decide), root265Candidates0_checked]
  change scan edge (search edge 3) 4 [1721] 1721 root265Tail1
    (if 3 ≤ root265Candidates0.length then search edge 3 [1721, 2213] root265Candidates0 3934 winner else winner) = _
  rw [if_neg (by decide)]
def root265Candidates1 : List Nat := [12107, 15287, 20549]
set_option maxRecDepth 8192 in
theorem root265Candidates1_checked : candidates edge 4 1721 2297 root265Tail2 winner.bound = root265Candidates1 := by
  unfold candidates
  have row : edge 2297 = (SortedNatTree.contains FastRows.row2297) := by
    change Data.row2297.contains = (SortedNatTree.contains FastRows.row2297)
    exact FastRows.row2297_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root265Child1 : search edge 3 [1721, 2297] root265Candidates1 4018 winner = winner := by decide
theorem root265Step1 : scan edge (search edge 3) 4 [1721] 1721 root265Tail1 winner =
    scan edge (search edge 3) 4 [1721] 1721 root265Tail2 winner := by
  rw [root265Tail1, scan, if_neg (by decide), root265Candidates1_checked]
  change scan edge (search edge 3) 4 [1721] 1721 root265Tail2
    (if 3 ≤ root265Candidates1.length then search edge 3 [1721, 2297] root265Candidates1 4018 winner else winner) = _
  rw [if_pos (by decide), root265Child1]
def root265Candidates2 : List Nat := [8273, 12107]
set_option maxRecDepth 8192 in
theorem root265Candidates2_checked : candidates edge 4 1721 2411 root265Tail3 winner.bound = root265Candidates2 := by
  unfold candidates
  have row : edge 2411 = (SortedNatTree.contains FastRows.row2411) := by
    change Data.row2411.contains = (SortedNatTree.contains FastRows.row2411)
    exact FastRows.row2411_checked.symm
  rw [row]
  decide
theorem root265Step2 : scan edge (search edge 3) 4 [1721] 1721 root265Tail2 winner =
    scan edge (search edge 3) 4 [1721] 1721 root265Tail3 winner := by
  rw [root265Tail2, scan, if_neg (by decide), root265Candidates2_checked]
  change scan edge (search edge 3) 4 [1721] 1721 root265Tail3
    (if 3 ≤ root265Candidates2.length then search edge 3 [1721, 2411] root265Candidates2 4132 winner else winner) = _
  rw [if_neg (by decide)]
def root265Candidates3 : List Nat := [19463]
set_option maxRecDepth 8192 in
theorem root265Candidates3_checked : candidates edge 4 1721 2621 root265Tail4 winner.bound = root265Candidates3 := by
  unfold candidates
  have row : edge 2621 = (SortedNatTree.contains FastRows.row2621) := by
    change Data.row2621.contains = (SortedNatTree.contains FastRows.row2621)
    exact FastRows.row2621_checked.symm
  rw [row]
  decide
theorem root265Step3 : scan edge (search edge 3) 4 [1721] 1721 root265Tail3 winner =
    scan edge (search edge 3) 4 [1721] 1721 root265Tail4 winner := by
  rw [root265Tail3, scan, if_neg (by decide), root265Candidates3_checked]
  change scan edge (search edge 3) 4 [1721] 1721 root265Tail4
    (if 3 ≤ root265Candidates3.length then search edge 3 [1721, 2621] root265Candidates3 4342 winner else winner) = _
  rw [if_neg (by decide)]
def root265Candidates4 : List Nat := [8231, 8273, 14771]
set_option maxRecDepth 8192 in
theorem root265Candidates4_checked : candidates edge 4 1721 3347 root265Tail5 winner.bound = root265Candidates4 := by
  unfold candidates
  have row : edge 3347 = (SortedNatTree.contains FastRows.row3347) := by
    change Data.row3347.contains = (SortedNatTree.contains FastRows.row3347)
    exact FastRows.row3347_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root265Child4 : search edge 3 [1721, 3347] root265Candidates4 5068 winner = winner := by decide
theorem root265Step4 : scan edge (search edge 3) 4 [1721] 1721 root265Tail4 winner =
    scan edge (search edge 3) 4 [1721] 1721 root265Tail5 winner := by
  rw [root265Tail4, scan, if_neg (by decide), root265Candidates4_checked]
  change scan edge (search edge 3) 4 [1721] 1721 root265Tail5
    (if 3 ≤ root265Candidates4.length then search edge 3 [1721, 3347] root265Candidates4 5068 winner else winner) = _
  rw [if_pos (by decide), root265Child4]
def root265Candidates5 : List Nat := [6779, 9677, 16871, 20177]
set_option maxRecDepth 8192 in
theorem root265Candidates5_checked : candidates edge 4 1721 3359 root265Tail6 winner.bound = root265Candidates5 := by
  unfold candidates
  have row : edge 3359 = (SortedNatTree.contains FastRows.row3359) := by
    change Data.row3359.contains = (SortedNatTree.contains FastRows.row3359)
    exact FastRows.row3359_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root265Child5 : search edge 3 [1721, 3359] root265Candidates5 5080 winner = winner := by decide
theorem root265Step5 : scan edge (search edge 3) 4 [1721] 1721 root265Tail5 winner =
    scan edge (search edge 3) 4 [1721] 1721 root265Tail6 winner := by
  rw [root265Tail5, scan, if_neg (by decide), root265Candidates5_checked]
  change scan edge (search edge 3) 4 [1721] 1721 root265Tail6
    (if 3 ≤ root265Candidates5.length then search edge 3 [1721, 3359] root265Candidates5 5080 winner else winner) = _
  rw [if_pos (by decide), root265Child5]
def root265Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root265Candidates6_checked : candidates edge 4 1721 3557 root265Tail7 winner.bound = root265Candidates6 := by
  unfold candidates
  have row : edge 3557 = (SortedNatTree.contains FastRows.row3557) := by
    change Data.row3557.contains = (SortedNatTree.contains FastRows.row3557)
    exact FastRows.row3557_checked.symm
  rw [row]
  decide
theorem root265Step6 : scan edge (search edge 3) 4 [1721] 1721 root265Tail6 winner =
    scan edge (search edge 3) 4 [1721] 1721 root265Tail7 winner := by
  rw [root265Tail6, scan, if_neg (by decide), root265Candidates6_checked]
  change scan edge (search edge 3) 4 [1721] 1721 root265Tail7
    (if 3 ≤ root265Candidates6.length then search edge 3 [1721, 3557] root265Candidates6 5278 winner else winner) = _
  rw [if_neg (by decide)]
def root265Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root265Candidates7_checked : candidates edge 4 1721 4679 root265Tail8 winner.bound = root265Candidates7 := by
  unfold candidates
  have row : edge 4679 = (SortedNatTree.contains FastRows.row4679) := by
    change Data.row4679.contains = (SortedNatTree.contains FastRows.row4679)
    exact FastRows.row4679_checked.symm
  rw [row]
  decide
theorem root265Step7 : scan edge (search edge 3) 4 [1721] 1721 root265Tail7 winner =
    scan edge (search edge 3) 4 [1721] 1721 root265Tail8 winner := by
  rw [root265Tail7, scan, if_neg (by decide), root265Candidates7_checked]
  change scan edge (search edge 3) 4 [1721] 1721 root265Tail8
    (if 3 ≤ root265Candidates7.length then search edge 3 [1721, 4679] root265Candidates7 6400 winner else winner) = _
  rw [if_neg (by decide)]
theorem root265Scan8 : scan edge (search edge 3) 4 [1721] 1721 root265Tail8 winner = winner := by rw [root265Tail8, scan, if_pos (by decide)]
theorem root265Scan7 : scan edge (search edge 3) 4 [1721] 1721 root265Tail7 winner = winner :=
  (root265Step7).trans root265Scan8
theorem root265Scan6 : scan edge (search edge 3) 4 [1721] 1721 root265Tail6 winner = winner :=
  (root265Step6).trans root265Scan7
theorem root265Scan5 : scan edge (search edge 3) 4 [1721] 1721 root265Tail5 winner = winner :=
  (root265Step5).trans root265Scan6
theorem root265Scan4 : scan edge (search edge 3) 4 [1721] 1721 root265Tail4 winner = winner :=
  (root265Step4).trans root265Scan5
theorem root265Scan3 : scan edge (search edge 3) 4 [1721] 1721 root265Tail3 winner = winner :=
  (root265Step3).trans root265Scan4
theorem root265Scan2 : scan edge (search edge 3) 4 [1721] 1721 root265Tail2 winner = winner :=
  (root265Step2).trans root265Scan3
theorem root265Scan1 : scan edge (search edge 3) 4 [1721] 1721 root265Tail1 winner = winner :=
  (root265Step1).trans root265Scan2
theorem root265Scan0 : scan edge (search edge 3) 4 [1721] 1721 root265Tail0 winner = winner :=
  (root265Step0).trans root265Scan1
set_option maxRecDepth 4096 in
theorem root265Branch : search edge 4 [1721] root265Candidates 1721 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1721] 1721 root265Tail0 winner = winner
  exact root265Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
