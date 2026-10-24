import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root263Candidates : List Nat := Data.row1699
set_option maxRecDepth 8192 in
theorem root263Candidates_checked : candidates edge 5 0 1699 tail264 winner.bound = root263Candidates := by
  apply candidates_eq_row edge 5 0 1699 tail264 root263Candidates winner.bound tail264_ordered
  · decide
  · decide
  · funext q
    rfl
def root263Tail46 : List Nat := []
def root263Tail45 : List Nat := 23497 :: root263Tail46
def root263Tail44 : List Nat := 22783 :: root263Tail45
def root263Tail43 : List Nat := 21799 :: root263Tail44
def root263Tail42 : List Nat := 21727 :: root263Tail43
def root263Tail41 : List Nat := 21481 :: root263Tail42
def root263Tail40 : List Nat := 19609 :: root263Tail41
def root263Tail39 : List Nat := 18481 :: root263Tail40
def root263Tail38 : List Nat := 17569 :: root263Tail39
def root263Tail37 : List Nat := 17383 :: root263Tail38
def root263Tail36 : List Nat := 16339 :: root263Tail37
def root263Tail35 : List Nat := 15511 :: root263Tail36
def root263Tail34 : List Nat := 15121 :: root263Tail35
def root263Tail33 : List Nat := 14923 :: root263Tail34
def root263Tail32 : List Nat := 14737 :: root263Tail33
def root263Tail31 : List Nat := 13399 :: root263Tail32
def root263Tail30 : List Nat := 13099 :: root263Tail31
def root263Tail29 : List Nat := 13033 :: root263Tail30
def root263Tail28 : List Nat := 12577 :: root263Tail29
def root263Tail27 : List Nat := 12553 :: root263Tail28
def root263Tail26 : List Nat := 12517 :: root263Tail27
def root263Tail25 : List Nat := 12343 :: root263Tail26
def root263Tail24 : List Nat := 11353 :: root263Tail25
def root263Tail23 : List Nat := 11317 :: root263Tail24
def root263Tail22 : List Nat := 10957 :: root263Tail23
def root263Tail21 : List Nat := 10789 :: root263Tail22
def root263Tail20 : List Nat := 8713 :: root263Tail21
def root263Tail19 : List Nat := 7639 :: root263Tail20
def root263Tail18 : List Nat := 7549 :: root263Tail19
def root263Tail17 : List Nat := 7507 :: root263Tail18
def root263Tail16 : List Nat := 6691 :: root263Tail17
def root263Tail15 : List Nat := 6571 :: root263Tail16
def root263Tail14 : List Nat := 6121 :: root263Tail15
def root263Tail13 : List Nat := 5791 :: root263Tail14
def root263Tail12 : List Nat := 5659 :: root263Tail13
def root263Tail11 : List Nat := 4567 :: root263Tail12
def root263Tail10 : List Nat := 4363 :: root263Tail11
def root263Tail9 : List Nat := 4177 :: root263Tail10
def root263Tail8 : List Nat := 3853 :: root263Tail9
def root263Tail7 : List Nat := 3607 :: root263Tail8
def root263Tail6 : List Nat := 3373 :: root263Tail7
def root263Tail5 : List Nat := 3313 :: root263Tail6
def root263Tail4 : List Nat := 2851 :: root263Tail5
def root263Tail3 : List Nat := 2203 :: root263Tail4
def root263Tail2 : List Nat := 2179 :: root263Tail3
def root263Tail1 : List Nat := 1861 :: root263Tail2
def root263Tail0 : List Nat := 1759 :: root263Tail1
def root263Candidates0 : List Nat := [4177, 15121]
set_option maxRecDepth 8192 in
theorem root263Candidates0_checked : candidates edge 4 1699 1759 root263Tail1 winner.bound = root263Candidates0 := by
  unfold candidates
  have row : edge 1759 = (SortedNatTree.contains FastRows.row1759) := by
    change Data.row1759.contains = (SortedNatTree.contains FastRows.row1759)
    exact FastRows.row1759_checked.symm
  rw [row]
  decide
theorem root263Step0 : scan edge (search edge 3) 4 [1699] 1699 root263Tail0 winner =
    scan edge (search edge 3) 4 [1699] 1699 root263Tail1 winner := by
  rw [root263Tail0, scan, if_neg (by decide), root263Candidates0_checked]
  change scan edge (search edge 3) 4 [1699] 1699 root263Tail1
    (if 3 ≤ root263Candidates0.length then search edge 3 [1699, 1759] root263Candidates0 3458 winner else winner) = _
  rw [if_neg (by decide)]
def root263Candidates1 : List Nat := [6121]
set_option maxRecDepth 8192 in
theorem root263Candidates1_checked : candidates edge 4 1699 1861 root263Tail2 winner.bound = root263Candidates1 := by
  unfold candidates
  have row : edge 1861 = (SortedNatTree.contains FastRows.row1861) := by
    change Data.row1861.contains = (SortedNatTree.contains FastRows.row1861)
    exact FastRows.row1861_checked.symm
  rw [row]
  decide
theorem root263Step1 : scan edge (search edge 3) 4 [1699] 1699 root263Tail1 winner =
    scan edge (search edge 3) 4 [1699] 1699 root263Tail2 winner := by
  rw [root263Tail1, scan, if_neg (by decide), root263Candidates1_checked]
  change scan edge (search edge 3) 4 [1699] 1699 root263Tail2
    (if 3 ≤ root263Candidates1.length then search edge 3 [1699, 1861] root263Candidates1 3560 winner else winner) = _
  rw [if_neg (by decide)]
def root263Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root263Candidates2_checked : candidates edge 4 1699 2179 root263Tail3 winner.bound = root263Candidates2 := by
  unfold candidates
  have row : edge 2179 = (SortedNatTree.contains FastRows.row2179) := by
    change Data.row2179.contains = (SortedNatTree.contains FastRows.row2179)
    exact FastRows.row2179_checked.symm
  rw [row]
  decide
theorem root263Step2 : scan edge (search edge 3) 4 [1699] 1699 root263Tail2 winner =
    scan edge (search edge 3) 4 [1699] 1699 root263Tail3 winner := by
  rw [root263Tail2, scan, if_neg (by decide), root263Candidates2_checked]
  change scan edge (search edge 3) 4 [1699] 1699 root263Tail3
    (if 3 ≤ root263Candidates2.length then search edge 3 [1699, 2179] root263Candidates2 3878 winner else winner) = _
  rw [if_neg (by decide)]
def root263Candidates3 : List Nat := [11353]
set_option maxRecDepth 8192 in
theorem root263Candidates3_checked : candidates edge 4 1699 2203 root263Tail4 winner.bound = root263Candidates3 := by
  unfold candidates
  have row : edge 2203 = (SortedNatTree.contains FastRows.row2203) := by
    change Data.row2203.contains = (SortedNatTree.contains FastRows.row2203)
    exact FastRows.row2203_checked.symm
  rw [row]
  decide
theorem root263Step3 : scan edge (search edge 3) 4 [1699] 1699 root263Tail3 winner =
    scan edge (search edge 3) 4 [1699] 1699 root263Tail4 winner := by
  rw [root263Tail3, scan, if_neg (by decide), root263Candidates3_checked]
  change scan edge (search edge 3) 4 [1699] 1699 root263Tail4
    (if 3 ≤ root263Candidates3.length then search edge 3 [1699, 2203] root263Candidates3 3902 winner else winner) = _
  rw [if_neg (by decide)]
def root263Candidates4 : List Nat := [3313, 19609]
set_option maxRecDepth 8192 in
theorem root263Candidates4_checked : candidates edge 4 1699 2851 root263Tail5 winner.bound = root263Candidates4 := by
  unfold candidates
  have row : edge 2851 = (SortedNatTree.contains FastRows.row2851) := by
    change Data.row2851.contains = (SortedNatTree.contains FastRows.row2851)
    exact FastRows.row2851_checked.symm
  rw [row]
  decide
theorem root263Step4 : scan edge (search edge 3) 4 [1699] 1699 root263Tail4 winner =
    scan edge (search edge 3) 4 [1699] 1699 root263Tail5 winner := by
  rw [root263Tail4, scan, if_neg (by decide), root263Candidates4_checked]
  change scan edge (search edge 3) 4 [1699] 1699 root263Tail5
    (if 3 ≤ root263Candidates4.length then search edge 3 [1699, 2851] root263Candidates4 4550 winner else winner) = _
  rw [if_neg (by decide)]
def root263Candidates5 : List Nat := [14923]
set_option maxRecDepth 8192 in
theorem root263Candidates5_checked : candidates edge 4 1699 3313 root263Tail6 winner.bound = root263Candidates5 := by
  unfold candidates
  have row : edge 3313 = (SortedNatTree.contains FastRows.row3313) := by
    change Data.row3313.contains = (SortedNatTree.contains FastRows.row3313)
    exact FastRows.row3313_checked.symm
  rw [row]
  decide
theorem root263Step5 : scan edge (search edge 3) 4 [1699] 1699 root263Tail5 winner =
    scan edge (search edge 3) 4 [1699] 1699 root263Tail6 winner := by
  rw [root263Tail5, scan, if_neg (by decide), root263Candidates5_checked]
  change scan edge (search edge 3) 4 [1699] 1699 root263Tail6
    (if 3 ≤ root263Candidates5.length then search edge 3 [1699, 3313] root263Candidates5 5012 winner else winner) = _
  rw [if_neg (by decide)]
def root263Candidates6 : List Nat := [18481]
set_option maxRecDepth 8192 in
theorem root263Candidates6_checked : candidates edge 4 1699 3373 root263Tail7 winner.bound = root263Candidates6 := by
  unfold candidates
  have row : edge 3373 = (SortedNatTree.contains FastRows.row3373) := by
    change Data.row3373.contains = (SortedNatTree.contains FastRows.row3373)
    exact FastRows.row3373_checked.symm
  rw [row]
  decide
theorem root263Step6 : scan edge (search edge 3) 4 [1699] 1699 root263Tail6 winner =
    scan edge (search edge 3) 4 [1699] 1699 root263Tail7 winner := by
  rw [root263Tail6, scan, if_neg (by decide), root263Candidates6_checked]
  change scan edge (search edge 3) 4 [1699] 1699 root263Tail7
    (if 3 ≤ root263Candidates6.length then search edge 3 [1699, 3373] root263Candidates6 5072 winner else winner) = _
  rw [if_neg (by decide)]
def root263Candidates7 : List Nat := [6571, 13033, 14737]
set_option maxRecDepth 8192 in
theorem root263Candidates7_checked : candidates edge 4 1699 3607 root263Tail8 winner.bound = root263Candidates7 := by
  unfold candidates
  have row : edge 3607 = (SortedNatTree.contains FastRows.row3607) := by
    change Data.row3607.contains = (SortedNatTree.contains FastRows.row3607)
    exact FastRows.row3607_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root263Child7 : search edge 3 [1699, 3607] root263Candidates7 5306 winner = winner := by decide
theorem root263Step7 : scan edge (search edge 3) 4 [1699] 1699 root263Tail7 winner =
    scan edge (search edge 3) 4 [1699] 1699 root263Tail8 winner := by
  rw [root263Tail7, scan, if_neg (by decide), root263Candidates7_checked]
  change scan edge (search edge 3) 4 [1699] 1699 root263Tail8
    (if 3 ≤ root263Candidates7.length then search edge 3 [1699, 3607] root263Candidates7 5306 winner else winner) = _
  rw [if_pos (by decide), root263Child7]
def root263Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root263Candidates8_checked : candidates edge 4 1699 3853 root263Tail9 winner.bound = root263Candidates8 := by
  unfold candidates
  have row : edge 3853 = (SortedNatTree.contains FastRows.row3853) := by
    change Data.row3853.contains = (SortedNatTree.contains FastRows.row3853)
    exact FastRows.row3853_checked.symm
  rw [row]
  decide
theorem root263Step8 : scan edge (search edge 3) 4 [1699] 1699 root263Tail8 winner =
    scan edge (search edge 3) 4 [1699] 1699 root263Tail9 winner := by
  rw [root263Tail8, scan, if_neg (by decide), root263Candidates8_checked]
  change scan edge (search edge 3) 4 [1699] 1699 root263Tail9
    (if 3 ≤ root263Candidates8.length then search edge 3 [1699, 3853] root263Candidates8 5552 winner else winner) = _
  rw [if_neg (by decide)]
def root263Candidates9 : List Nat := [4363, 5791]
set_option maxRecDepth 8192 in
theorem root263Candidates9_checked : candidates edge 4 1699 4177 root263Tail10 winner.bound = root263Candidates9 := by
  unfold candidates
  have row : edge 4177 = (SortedNatTree.contains FastRows.row4177) := by
    change Data.row4177.contains = (SortedNatTree.contains FastRows.row4177)
    exact FastRows.row4177_checked.symm
  rw [row]
  decide
theorem root263Step9 : scan edge (search edge 3) 4 [1699] 1699 root263Tail9 winner =
    scan edge (search edge 3) 4 [1699] 1699 root263Tail10 winner := by
  rw [root263Tail9, scan, if_neg (by decide), root263Candidates9_checked]
  change scan edge (search edge 3) 4 [1699] 1699 root263Tail10
    (if 3 ≤ root263Candidates9.length then search edge 3 [1699, 4177] root263Candidates9 5876 winner else winner) = _
  rw [if_neg (by decide)]
def root263Candidates10 : List Nat := [8713]
set_option maxRecDepth 8192 in
theorem root263Candidates10_checked : candidates edge 4 1699 4363 root263Tail11 winner.bound = root263Candidates10 := by
  unfold candidates
  have row : edge 4363 = (SortedNatTree.contains FastRows.row4363) := by
    change Data.row4363.contains = (SortedNatTree.contains FastRows.row4363)
    exact FastRows.row4363_checked.symm
  rw [row]
  decide
theorem root263Step10 : scan edge (search edge 3) 4 [1699] 1699 root263Tail10 winner =
    scan edge (search edge 3) 4 [1699] 1699 root263Tail11 winner := by
  rw [root263Tail10, scan, if_neg (by decide), root263Candidates10_checked]
  change scan edge (search edge 3) 4 [1699] 1699 root263Tail11
    (if 3 ≤ root263Candidates10.length then search edge 3 [1699, 4363] root263Candidates10 6062 winner else winner) = _
  rw [if_neg (by decide)]
def root263Candidates11 : List Nat := [14737]
set_option maxRecDepth 8192 in
theorem root263Candidates11_checked : candidates edge 4 1699 4567 root263Tail12 winner.bound = root263Candidates11 := by
  unfold candidates
  have row : edge 4567 = (SortedNatTree.contains FastRows.row4567) := by
    change Data.row4567.contains = (SortedNatTree.contains FastRows.row4567)
    exact FastRows.row4567_checked.symm
  rw [row]
  decide
theorem root263Step11 : scan edge (search edge 3) 4 [1699] 1699 root263Tail11 winner =
    scan edge (search edge 3) 4 [1699] 1699 root263Tail12 winner := by
  rw [root263Tail11, scan, if_neg (by decide), root263Candidates11_checked]
  change scan edge (search edge 3) 4 [1699] 1699 root263Tail12
    (if 3 ≤ root263Candidates11.length then search edge 3 [1699, 4567] root263Candidates11 6266 winner else winner) = _
  rw [if_neg (by decide)]
def root263Candidates12 : List Nat := [6121]
set_option maxRecDepth 8192 in
theorem root263Candidates12_checked : candidates edge 4 1699 5659 root263Tail13 winner.bound = root263Candidates12 := by
  unfold candidates
  have row : edge 5659 = (SortedNatTree.contains FastRows.row5659) := by
    change Data.row5659.contains = (SortedNatTree.contains FastRows.row5659)
    exact FastRows.row5659_checked.symm
  rw [row]
  decide
theorem root263Step12 : scan edge (search edge 3) 4 [1699] 1699 root263Tail12 winner =
    scan edge (search edge 3) 4 [1699] 1699 root263Tail13 winner := by
  rw [root263Tail12, scan, if_neg (by decide), root263Candidates12_checked]
  change scan edge (search edge 3) 4 [1699] 1699 root263Tail13
    (if 3 ≤ root263Candidates12.length then search edge 3 [1699, 5659] root263Candidates12 7358 winner else winner) = _
  rw [if_neg (by decide)]
def root263Candidates13 : List Nat := [17569]
set_option maxRecDepth 8192 in
theorem root263Candidates13_checked : candidates edge 4 1699 5791 root263Tail14 winner.bound = root263Candidates13 := by
  unfold candidates
  have row : edge 5791 = (SortedNatTree.contains FastRows.row5791) := by
    change Data.row5791.contains = (SortedNatTree.contains FastRows.row5791)
    exact FastRows.row5791_checked.symm
  rw [row]
  decide
theorem root263Step13 : scan edge (search edge 3) 4 [1699] 1699 root263Tail13 winner =
    scan edge (search edge 3) 4 [1699] 1699 root263Tail14 winner := by
  rw [root263Tail13, scan, if_neg (by decide), root263Candidates13_checked]
  change scan edge (search edge 3) 4 [1699] 1699 root263Tail14
    (if 3 ≤ root263Candidates13.length then search edge 3 [1699, 5791] root263Candidates13 7490 winner else winner) = _
  rw [if_neg (by decide)]
theorem root263Scan14 : scan edge (search edge 3) 4 [1699] 1699 root263Tail14 winner = winner := by rw [root263Tail14, scan, if_pos (by decide)]
theorem root263Scan13 : scan edge (search edge 3) 4 [1699] 1699 root263Tail13 winner = winner :=
  (root263Step13).trans root263Scan14
theorem root263Scan12 : scan edge (search edge 3) 4 [1699] 1699 root263Tail12 winner = winner :=
  (root263Step12).trans root263Scan13
theorem root263Scan11 : scan edge (search edge 3) 4 [1699] 1699 root263Tail11 winner = winner :=
  (root263Step11).trans root263Scan12
theorem root263Scan10 : scan edge (search edge 3) 4 [1699] 1699 root263Tail10 winner = winner :=
  (root263Step10).trans root263Scan11
theorem root263Scan9 : scan edge (search edge 3) 4 [1699] 1699 root263Tail9 winner = winner :=
  (root263Step9).trans root263Scan10
theorem root263Scan8 : scan edge (search edge 3) 4 [1699] 1699 root263Tail8 winner = winner :=
  (root263Step8).trans root263Scan9
theorem root263Scan7 : scan edge (search edge 3) 4 [1699] 1699 root263Tail7 winner = winner :=
  (root263Step7).trans root263Scan8
theorem root263Scan6 : scan edge (search edge 3) 4 [1699] 1699 root263Tail6 winner = winner :=
  (root263Step6).trans root263Scan7
theorem root263Scan5 : scan edge (search edge 3) 4 [1699] 1699 root263Tail5 winner = winner :=
  (root263Step5).trans root263Scan6
theorem root263Scan4 : scan edge (search edge 3) 4 [1699] 1699 root263Tail4 winner = winner :=
  (root263Step4).trans root263Scan5
theorem root263Scan3 : scan edge (search edge 3) 4 [1699] 1699 root263Tail3 winner = winner :=
  (root263Step3).trans root263Scan4
theorem root263Scan2 : scan edge (search edge 3) 4 [1699] 1699 root263Tail2 winner = winner :=
  (root263Step2).trans root263Scan3
theorem root263Scan1 : scan edge (search edge 3) 4 [1699] 1699 root263Tail1 winner = winner :=
  (root263Step1).trans root263Scan2
theorem root263Scan0 : scan edge (search edge 3) 4 [1699] 1699 root263Tail0 winner = winner :=
  (root263Step0).trans root263Scan1
set_option maxRecDepth 4096 in
theorem root263Branch : search edge 4 [1699] root263Candidates 1699 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1699] 1699 root263Tail0 winner = winner
  exact root263Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
