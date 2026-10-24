import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root383Candidates : List Nat := Data.row2663
set_option maxRecDepth 8192 in
theorem root383Candidates_checked : candidates edge 5 0 2663 tail384 winner.bound = root383Candidates := by
  apply candidates_eq_row edge 5 0 2663 tail384 root383Candidates winner.bound tail384_ordered
  · decide
  · decide
  · funext q
    rfl
def root383Tail35 : List Nat := []
def root383Tail34 : List Nat := 23081 :: root383Tail35
def root383Tail33 : List Nat := 22853 :: root383Tail34
def root383Tail32 : List Nat := 21863 :: root383Tail33
def root383Tail31 : List Nat := 21521 :: root383Tail32
def root383Tail30 : List Nat := 21467 :: root383Tail31
def root383Tail29 : List Nat := 19421 :: root383Tail30
def root383Tail28 : List Nat := 16943 :: root383Tail29
def root383Tail27 : List Nat := 16553 :: root383Tail28
def root383Tail26 : List Nat := 16097 :: root383Tail27
def root383Tail25 : List Nat := 15173 :: root383Tail26
def root383Tail24 : List Nat := 14957 :: root383Tail25
def root383Tail23 : List Nat := 14891 :: root383Tail24
def root383Tail22 : List Nat := 14369 :: root383Tail23
def root383Tail21 : List Nat := 14327 :: root383Tail22
def root383Tail20 : List Nat := 13997 :: root383Tail21
def root383Tail19 : List Nat := 13337 :: root383Tail20
def root383Tail18 : List Nat := 12479 :: root383Tail19
def root383Tail17 : List Nat := 11489 :: root383Tail18
def root383Tail16 : List Nat := 10259 :: root383Tail17
def root383Tail15 : List Nat := 10007 :: root383Tail16
def root383Tail14 : List Nat := 9293 :: root383Tail15
def root383Tail13 : List Nat := 8783 :: root383Tail14
def root383Tail12 : List Nat := 8291 :: root383Tail13
def root383Tail11 : List Nat := 7901 :: root383Tail12
def root383Tail10 : List Nat := 7607 :: root383Tail11
def root383Tail9 : List Nat := 6359 :: root383Tail10
def root383Tail8 : List Nat := 5849 :: root383Tail9
def root383Tail7 : List Nat := 5813 :: root383Tail8
def root383Tail6 : List Nat := 5669 :: root383Tail7
def root383Tail5 : List Nat := 5471 :: root383Tail6
def root383Tail4 : List Nat := 4457 :: root383Tail5
def root383Tail3 : List Nat := 3779 :: root383Tail4
def root383Tail2 : List Nat := 3533 :: root383Tail3
def root383Tail1 : List Nat := 3251 :: root383Tail2
def root383Tail0 : List Nat := 3119 :: root383Tail1
def root383Candidates0 : List Nat := [6359]
set_option maxRecDepth 8192 in
theorem root383Candidates0_checked : candidates edge 4 2663 3119 root383Tail1 winner.bound = root383Candidates0 := by
  unfold candidates
  have row : edge 3119 = (SortedNatTree.contains FastRows.row3119) := by
    change Data.row3119.contains = (SortedNatTree.contains FastRows.row3119)
    exact FastRows.row3119_checked.symm
  rw [row]
  decide
theorem root383Step0 : scan edge (search edge 3) 4 [2663] 2663 root383Tail0 winner =
    scan edge (search edge 3) 4 [2663] 2663 root383Tail1 winner := by
  rw [root383Tail0, scan, if_neg (by decide), root383Candidates0_checked]
  change scan edge (search edge 3) 4 [2663] 2663 root383Tail1
    (if 3 ≤ root383Candidates0.length then search edge 3 [2663, 3119] root383Candidates0 5782 winner else winner) = _
  rw [if_neg (by decide)]
def root383Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root383Candidates1_checked : candidates edge 4 2663 3251 root383Tail2 winner.bound = root383Candidates1 := by
  unfold candidates
  have row : edge 3251 = (SortedNatTree.contains FastRows.row3251) := by
    change Data.row3251.contains = (SortedNatTree.contains FastRows.row3251)
    exact FastRows.row3251_checked.symm
  rw [row]
  decide
theorem root383Step1 : scan edge (search edge 3) 4 [2663] 2663 root383Tail1 winner =
    scan edge (search edge 3) 4 [2663] 2663 root383Tail2 winner := by
  rw [root383Tail1, scan, if_neg (by decide), root383Candidates1_checked]
  change scan edge (search edge 3) 4 [2663] 2663 root383Tail2
    (if 3 ≤ root383Candidates1.length then search edge 3 [2663, 3251] root383Candidates1 5914 winner else winner) = _
  rw [if_neg (by decide)]
def root383Candidates2 : List Nat := [7607]
set_option maxRecDepth 8192 in
theorem root383Candidates2_checked : candidates edge 4 2663 3533 root383Tail3 winner.bound = root383Candidates2 := by
  unfold candidates
  have row : edge 3533 = (SortedNatTree.contains FastRows.row3533) := by
    change Data.row3533.contains = (SortedNatTree.contains FastRows.row3533)
    exact FastRows.row3533_checked.symm
  rw [row]
  decide
theorem root383Step2 : scan edge (search edge 3) 4 [2663] 2663 root383Tail2 winner =
    scan edge (search edge 3) 4 [2663] 2663 root383Tail3 winner := by
  rw [root383Tail2, scan, if_neg (by decide), root383Candidates2_checked]
  change scan edge (search edge 3) 4 [2663] 2663 root383Tail3
    (if 3 ≤ root383Candidates2.length then search edge 3 [2663, 3533] root383Candidates2 6196 winner else winner) = _
  rw [if_neg (by decide)]
def root383Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root383Candidates3_checked : candidates edge 4 2663 3779 root383Tail4 winner.bound = root383Candidates3 := by
  unfold candidates
  have row : edge 3779 = (SortedNatTree.contains FastRows.row3779) := by
    change Data.row3779.contains = (SortedNatTree.contains FastRows.row3779)
    exact FastRows.row3779_checked.symm
  rw [row]
  decide
theorem root383Step3 : scan edge (search edge 3) 4 [2663] 2663 root383Tail3 winner =
    scan edge (search edge 3) 4 [2663] 2663 root383Tail4 winner := by
  rw [root383Tail3, scan, if_neg (by decide), root383Candidates3_checked]
  change scan edge (search edge 3) 4 [2663] 2663 root383Tail4
    (if 3 ≤ root383Candidates3.length then search edge 3 [2663, 3779] root383Candidates3 6442 winner else winner) = _
  rw [if_neg (by decide)]
def root383Candidates4 : List Nat := [7607]
set_option maxRecDepth 8192 in
theorem root383Candidates4_checked : candidates edge 4 2663 4457 root383Tail5 winner.bound = root383Candidates4 := by
  unfold candidates
  have row : edge 4457 = (SortedNatTree.contains FastRows.row4457) := by
    change Data.row4457.contains = (SortedNatTree.contains FastRows.row4457)
    exact FastRows.row4457_checked.symm
  rw [row]
  decide
theorem root383Step4 : scan edge (search edge 3) 4 [2663] 2663 root383Tail4 winner =
    scan edge (search edge 3) 4 [2663] 2663 root383Tail5 winner := by
  rw [root383Tail4, scan, if_neg (by decide), root383Candidates4_checked]
  change scan edge (search edge 3) 4 [2663] 2663 root383Tail5
    (if 3 ≤ root383Candidates4.length then search edge 3 [2663, 4457] root383Candidates4 7120 winner else winner) = _
  rw [if_neg (by decide)]
def root383Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root383Candidates5_checked : candidates edge 4 2663 5471 root383Tail6 winner.bound = root383Candidates5 := by
  unfold candidates
  have row : edge 5471 = (SortedNatTree.contains FastRows.row5471) := by
    change Data.row5471.contains = (SortedNatTree.contains FastRows.row5471)
    exact FastRows.row5471_checked.symm
  rw [row]
  decide
theorem root383Step5 : scan edge (search edge 3) 4 [2663] 2663 root383Tail5 winner =
    scan edge (search edge 3) 4 [2663] 2663 root383Tail6 winner := by
  rw [root383Tail5, scan, if_neg (by decide), root383Candidates5_checked]
  change scan edge (search edge 3) 4 [2663] 2663 root383Tail6
    (if 3 ≤ root383Candidates5.length then search edge 3 [2663, 5471] root383Candidates5 8134 winner else winner) = _
  rw [if_neg (by decide)]
def root383Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root383Candidates6_checked : candidates edge 4 2663 5669 root383Tail7 winner.bound = root383Candidates6 := by
  unfold candidates
  have row : edge 5669 = (SortedNatTree.contains FastRows.row5669) := by
    change Data.row5669.contains = (SortedNatTree.contains FastRows.row5669)
    exact FastRows.row5669_checked.symm
  rw [row]
  decide
theorem root383Step6 : scan edge (search edge 3) 4 [2663] 2663 root383Tail6 winner =
    scan edge (search edge 3) 4 [2663] 2663 root383Tail7 winner := by
  rw [root383Tail6, scan, if_neg (by decide), root383Candidates6_checked]
  change scan edge (search edge 3) 4 [2663] 2663 root383Tail7
    (if 3 ≤ root383Candidates6.length then search edge 3 [2663, 5669] root383Candidates6 8332 winner else winner) = _
  rw [if_neg (by decide)]
def root383Candidates7 : List Nat := [10007, 14957]
set_option maxRecDepth 8192 in
theorem root383Candidates7_checked : candidates edge 4 2663 5813 root383Tail8 winner.bound = root383Candidates7 := by
  unfold candidates
  have row : edge 5813 = (SortedNatTree.contains FastRows.row5813) := by
    change Data.row5813.contains = (SortedNatTree.contains FastRows.row5813)
    exact FastRows.row5813_checked.symm
  rw [row]
  decide
theorem root383Step7 : scan edge (search edge 3) 4 [2663] 2663 root383Tail7 winner =
    scan edge (search edge 3) 4 [2663] 2663 root383Tail8 winner := by
  rw [root383Tail7, scan, if_neg (by decide), root383Candidates7_checked]
  change scan edge (search edge 3) 4 [2663] 2663 root383Tail8
    (if 3 ≤ root383Candidates7.length then search edge 3 [2663, 5813] root383Candidates7 8476 winner else winner) = _
  rw [if_neg (by decide)]
theorem root383Scan8 : scan edge (search edge 3) 4 [2663] 2663 root383Tail8 winner = winner := by rw [root383Tail8, scan, if_pos (by decide)]
theorem root383Scan7 : scan edge (search edge 3) 4 [2663] 2663 root383Tail7 winner = winner :=
  (root383Step7).trans root383Scan8
theorem root383Scan6 : scan edge (search edge 3) 4 [2663] 2663 root383Tail6 winner = winner :=
  (root383Step6).trans root383Scan7
theorem root383Scan5 : scan edge (search edge 3) 4 [2663] 2663 root383Tail5 winner = winner :=
  (root383Step5).trans root383Scan6
theorem root383Scan4 : scan edge (search edge 3) 4 [2663] 2663 root383Tail4 winner = winner :=
  (root383Step4).trans root383Scan5
theorem root383Scan3 : scan edge (search edge 3) 4 [2663] 2663 root383Tail3 winner = winner :=
  (root383Step3).trans root383Scan4
theorem root383Scan2 : scan edge (search edge 3) 4 [2663] 2663 root383Tail2 winner = winner :=
  (root383Step2).trans root383Scan3
theorem root383Scan1 : scan edge (search edge 3) 4 [2663] 2663 root383Tail1 winner = winner :=
  (root383Step1).trans root383Scan2
theorem root383Scan0 : scan edge (search edge 3) 4 [2663] 2663 root383Tail0 winner = winner :=
  (root383Step0).trans root383Scan1
set_option maxRecDepth 4096 in
theorem root383Branch : search edge 4 [2663] root383Candidates 2663 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2663] 2663 root383Tail0 winner = winner
  exact root383Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
