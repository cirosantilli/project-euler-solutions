import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root419Candidates : List Nat := Data.row2917
set_option maxRecDepth 8192 in
theorem root419Candidates_checked : candidates edge 5 0 2917 tail420 winner.bound = root419Candidates := by
  apply candidates_eq_row edge 5 0 2917 tail420 root419Candidates winner.bound tail420_ordered
  · decide
  · decide
  · funext q
    rfl
def root419Tail40 : List Nat := []
def root419Tail39 : List Nat := 22129 :: root419Tail40
def root419Tail38 : List Nat := 22111 :: root419Tail39
def root419Tail37 : List Nat := 22063 :: root419Tail38
def root419Tail36 : List Nat := 21841 :: root419Tail37
def root419Tail35 : List Nat := 21523 :: root419Tail36
def root419Tail34 : List Nat := 21517 :: root419Tail35
def root419Tail33 : List Nat := 21499 :: root419Tail34
def root419Tail32 : List Nat := 21163 :: root419Tail33
def root419Tail31 : List Nat := 20959 :: root419Tail32
def root419Tail30 : List Nat := 20749 :: root419Tail31
def root419Tail29 : List Nat := 20641 :: root419Tail30
def root419Tail28 : List Nat := 19801 :: root419Tail29
def root419Tail27 : List Nat := 19249 :: root419Tail28
def root419Tail26 : List Nat := 19081 :: root419Tail27
def root419Tail25 : List Nat := 18121 :: root419Tail26
def root419Tail24 : List Nat := 16759 :: root419Tail25
def root419Tail23 : List Nat := 15661 :: root419Tail24
def root419Tail22 : List Nat := 14653 :: root419Tail23
def root419Tail21 : List Nat := 13567 :: root419Tail22
def root419Tail20 : List Nat := 12829 :: root419Tail21
def root419Tail19 : List Nat := 11443 :: root419Tail20
def root419Tail18 : List Nat := 11149 :: root419Tail19
def root419Tail17 : List Nat := 10267 :: root419Tail18
def root419Tail16 : List Nat := 10099 :: root419Tail17
def root419Tail15 : List Nat := 9859 :: root419Tail16
def root419Tail14 : List Nat := 9721 :: root419Tail15
def root419Tail13 : List Nat := 8731 :: root419Tail14
def root419Tail12 : List Nat := 8581 :: root419Tail13
def root419Tail11 : List Nat := 8179 :: root419Tail12
def root419Tail10 : List Nat := 7549 :: root419Tail11
def root419Tail9 : List Nat := 5851 :: root419Tail10
def root419Tail8 : List Nat := 5437 :: root419Tail9
def root419Tail7 : List Nat := 5227 :: root419Tail8
def root419Tail6 : List Nat := 5011 :: root419Tail7
def root419Tail5 : List Nat := 4909 :: root419Tail6
def root419Tail4 : List Nat := 4003 :: root419Tail5
def root419Tail3 : List Nat := 3733 :: root419Tail4
def root419Tail2 : List Nat := 3643 :: root419Tail3
def root419Tail1 : List Nat := 3631 :: root419Tail2
def root419Tail0 : List Nat := 3169 :: root419Tail1
def root419Candidates0 : List Nat := [3733, 8179, 11149]
set_option maxRecDepth 8192 in
theorem root419Candidates0_checked : candidates edge 4 2917 3169 root419Tail1 winner.bound = root419Candidates0 := by
  unfold candidates
  have row : edge 3169 = (SortedNatTree.contains FastRows.row3169) := by
    change Data.row3169.contains = (SortedNatTree.contains FastRows.row3169)
    exact FastRows.row3169_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root419Child0 : search edge 3 [2917, 3169] root419Candidates0 6086 winner = winner := by decide
theorem root419Step0 : scan edge (search edge 3) 4 [2917] 2917 root419Tail0 winner =
    scan edge (search edge 3) 4 [2917] 2917 root419Tail1 winner := by
  rw [root419Tail0, scan, if_neg (by decide), root419Candidates0_checked]
  change scan edge (search edge 3) 4 [2917] 2917 root419Tail1
    (if 3 ≤ root419Candidates0.length then search edge 3 [2917, 3169] root419Candidates0 6086 winner else winner) = _
  rw [if_pos (by decide), root419Child0]
def root419Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root419Candidates1_checked : candidates edge 4 2917 3631 root419Tail2 winner.bound = root419Candidates1 := by
  unfold candidates
  have row : edge 3631 = (SortedNatTree.contains FastRows.row3631) := by
    change Data.row3631.contains = (SortedNatTree.contains FastRows.row3631)
    exact FastRows.row3631_checked.symm
  rw [row]
  decide
theorem root419Step1 : scan edge (search edge 3) 4 [2917] 2917 root419Tail1 winner =
    scan edge (search edge 3) 4 [2917] 2917 root419Tail2 winner := by
  rw [root419Tail1, scan, if_neg (by decide), root419Candidates1_checked]
  change scan edge (search edge 3) 4 [2917] 2917 root419Tail2
    (if 3 ≤ root419Candidates1.length then search edge 3 [2917, 3631] root419Candidates1 6548 winner else winner) = _
  rw [if_neg (by decide)]
def root419Candidates2 : List Nat := [8581]
set_option maxRecDepth 8192 in
theorem root419Candidates2_checked : candidates edge 4 2917 3643 root419Tail3 winner.bound = root419Candidates2 := by
  unfold candidates
  have row : edge 3643 = (SortedNatTree.contains FastRows.row3643) := by
    change Data.row3643.contains = (SortedNatTree.contains FastRows.row3643)
    exact FastRows.row3643_checked.symm
  rw [row]
  decide
theorem root419Step2 : scan edge (search edge 3) 4 [2917] 2917 root419Tail2 winner =
    scan edge (search edge 3) 4 [2917] 2917 root419Tail3 winner := by
  rw [root419Tail2, scan, if_neg (by decide), root419Candidates2_checked]
  change scan edge (search edge 3) 4 [2917] 2917 root419Tail3
    (if 3 ≤ root419Candidates2.length then search edge 3 [2917, 3643] root419Candidates2 6560 winner else winner) = _
  rw [if_neg (by decide)]
def root419Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root419Candidates3_checked : candidates edge 4 2917 3733 root419Tail4 winner.bound = root419Candidates3 := by
  unfold candidates
  have row : edge 3733 = (SortedNatTree.contains FastRows.row3733) := by
    change Data.row3733.contains = (SortedNatTree.contains FastRows.row3733)
    exact FastRows.row3733_checked.symm
  rw [row]
  decide
theorem root419Step3 : scan edge (search edge 3) 4 [2917] 2917 root419Tail3 winner =
    scan edge (search edge 3) 4 [2917] 2917 root419Tail4 winner := by
  rw [root419Tail3, scan, if_neg (by decide), root419Candidates3_checked]
  change scan edge (search edge 3) 4 [2917] 2917 root419Tail4
    (if 3 ≤ root419Candidates3.length then search edge 3 [2917, 3733] root419Candidates3 6650 winner else winner) = _
  rw [if_neg (by decide)]
def root419Candidates4 : List Nat := [5851]
set_option maxRecDepth 8192 in
theorem root419Candidates4_checked : candidates edge 4 2917 4003 root419Tail5 winner.bound = root419Candidates4 := by
  unfold candidates
  have row : edge 4003 = (SortedNatTree.contains FastRows.row4003) := by
    change Data.row4003.contains = (SortedNatTree.contains FastRows.row4003)
    exact FastRows.row4003_checked.symm
  rw [row]
  decide
theorem root419Step4 : scan edge (search edge 3) 4 [2917] 2917 root419Tail4 winner =
    scan edge (search edge 3) 4 [2917] 2917 root419Tail5 winner := by
  rw [root419Tail4, scan, if_neg (by decide), root419Candidates4_checked]
  change scan edge (search edge 3) 4 [2917] 2917 root419Tail5
    (if 3 ≤ root419Candidates4.length then search edge 3 [2917, 4003] root419Candidates4 6920 winner else winner) = _
  rw [if_neg (by decide)]
def root419Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root419Candidates5_checked : candidates edge 4 2917 4909 root419Tail6 winner.bound = root419Candidates5 := by
  unfold candidates
  have row : edge 4909 = (SortedNatTree.contains FastRows.row4909) := by
    change Data.row4909.contains = (SortedNatTree.contains FastRows.row4909)
    exact FastRows.row4909_checked.symm
  rw [row]
  decide
theorem root419Step5 : scan edge (search edge 3) 4 [2917] 2917 root419Tail5 winner =
    scan edge (search edge 3) 4 [2917] 2917 root419Tail6 winner := by
  rw [root419Tail5, scan, if_neg (by decide), root419Candidates5_checked]
  change scan edge (search edge 3) 4 [2917] 2917 root419Tail6
    (if 3 ≤ root419Candidates5.length then search edge 3 [2917, 4909] root419Candidates5 7826 winner else winner) = _
  rw [if_neg (by decide)]
def root419Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root419Candidates6_checked : candidates edge 4 2917 5011 root419Tail7 winner.bound = root419Candidates6 := by
  unfold candidates
  have row : edge 5011 = (SortedNatTree.contains FastRows.row5011) := by
    change Data.row5011.contains = (SortedNatTree.contains FastRows.row5011)
    exact FastRows.row5011_checked.symm
  rw [row]
  decide
theorem root419Step6 : scan edge (search edge 3) 4 [2917] 2917 root419Tail6 winner =
    scan edge (search edge 3) 4 [2917] 2917 root419Tail7 winner := by
  rw [root419Tail6, scan, if_neg (by decide), root419Candidates6_checked]
  change scan edge (search edge 3) 4 [2917] 2917 root419Tail7
    (if 3 ≤ root419Candidates6.length then search edge 3 [2917, 5011] root419Candidates6 7928 winner else winner) = _
  rw [if_neg (by decide)]
def root419Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root419Candidates7_checked : candidates edge 4 2917 5227 root419Tail8 winner.bound = root419Candidates7 := by
  unfold candidates
  have row : edge 5227 = (SortedNatTree.contains FastRows.row5227) := by
    change Data.row5227.contains = (SortedNatTree.contains FastRows.row5227)
    exact FastRows.row5227_checked.symm
  rw [row]
  decide
theorem root419Step7 : scan edge (search edge 3) 4 [2917] 2917 root419Tail7 winner =
    scan edge (search edge 3) 4 [2917] 2917 root419Tail8 winner := by
  rw [root419Tail7, scan, if_neg (by decide), root419Candidates7_checked]
  change scan edge (search edge 3) 4 [2917] 2917 root419Tail8
    (if 3 ≤ root419Candidates7.length then search edge 3 [2917, 5227] root419Candidates7 8144 winner else winner) = _
  rw [if_neg (by decide)]
def root419Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root419Candidates8_checked : candidates edge 4 2917 5437 root419Tail9 winner.bound = root419Candidates8 := by
  unfold candidates
  have row : edge 5437 = (SortedNatTree.contains FastRows.row5437) := by
    change Data.row5437.contains = (SortedNatTree.contains FastRows.row5437)
    exact FastRows.row5437_checked.symm
  rw [row]
  decide
theorem root419Step8 : scan edge (search edge 3) 4 [2917] 2917 root419Tail8 winner =
    scan edge (search edge 3) 4 [2917] 2917 root419Tail9 winner := by
  rw [root419Tail8, scan, if_neg (by decide), root419Candidates8_checked]
  change scan edge (search edge 3) 4 [2917] 2917 root419Tail9
    (if 3 ≤ root419Candidates8.length then search edge 3 [2917, 5437] root419Candidates8 8354 winner else winner) = _
  rw [if_neg (by decide)]
theorem root419Scan9 : scan edge (search edge 3) 4 [2917] 2917 root419Tail9 winner = winner := by rw [root419Tail9, scan, if_pos (by decide)]
theorem root419Scan8 : scan edge (search edge 3) 4 [2917] 2917 root419Tail8 winner = winner :=
  (root419Step8).trans root419Scan9
theorem root419Scan7 : scan edge (search edge 3) 4 [2917] 2917 root419Tail7 winner = winner :=
  (root419Step7).trans root419Scan8
theorem root419Scan6 : scan edge (search edge 3) 4 [2917] 2917 root419Tail6 winner = winner :=
  (root419Step6).trans root419Scan7
theorem root419Scan5 : scan edge (search edge 3) 4 [2917] 2917 root419Tail5 winner = winner :=
  (root419Step5).trans root419Scan6
theorem root419Scan4 : scan edge (search edge 3) 4 [2917] 2917 root419Tail4 winner = winner :=
  (root419Step4).trans root419Scan5
theorem root419Scan3 : scan edge (search edge 3) 4 [2917] 2917 root419Tail3 winner = winner :=
  (root419Step3).trans root419Scan4
theorem root419Scan2 : scan edge (search edge 3) 4 [2917] 2917 root419Tail2 winner = winner :=
  (root419Step2).trans root419Scan3
theorem root419Scan1 : scan edge (search edge 3) 4 [2917] 2917 root419Tail1 winner = winner :=
  (root419Step1).trans root419Scan2
theorem root419Scan0 : scan edge (search edge 3) 4 [2917] 2917 root419Tail0 winner = winner :=
  (root419Step0).trans root419Scan1
set_option maxRecDepth 4096 in
theorem root419Branch : search edge 4 [2917] root419Candidates 2917 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2917] 2917 root419Tail0 winner = winner
  exact root419Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
