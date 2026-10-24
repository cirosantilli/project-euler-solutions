import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root260Candidates : List Nat := Data.row1669
set_option maxRecDepth 8192 in
theorem root260Candidates_checked : candidates edge 5 0 1669 tail261 winner.bound = root260Candidates := by
  apply candidates_eq_row edge 5 0 1669 tail261 root260Candidates winner.bound tail261_ordered
  · decide
  · decide
  · funext q
    rfl
def root260Tail44 : List Nat := []
def root260Tail43 : List Nat := 24061 :: root260Tail44
def root260Tail42 : List Nat := 22153 :: root260Tail43
def root260Tail41 : List Nat := 21163 :: root260Tail42
def root260Tail40 : List Nat := 20029 :: root260Tail41
def root260Tail39 : List Nat := 19603 :: root260Tail40
def root260Tail38 : List Nat := 19483 :: root260Tail39
def root260Tail37 : List Nat := 19213 :: root260Tail38
def root260Tail36 : List Nat := 18979 :: root260Tail37
def root260Tail35 : List Nat := 18919 :: root260Tail36
def root260Tail34 : List Nat := 18679 :: root260Tail35
def root260Tail33 : List Nat := 18379 :: root260Tail34
def root260Tail32 : List Nat := 17749 :: root260Tail33
def root260Tail31 : List Nat := 17341 :: root260Tail32
def root260Tail30 : List Nat := 16831 :: root260Tail31
def root260Tail29 : List Nat := 16447 :: root260Tail30
def root260Tail28 : List Nat := 13381 :: root260Tail29
def root260Tail27 : List Nat := 13099 :: root260Tail28
def root260Tail26 : List Nat := 12457 :: root260Tail27
def root260Tail25 : List Nat := 11959 :: root260Tail26
def root260Tail24 : List Nat := 11497 :: root260Tail25
def root260Tail23 : List Nat := 10729 :: root260Tail24
def root260Tail22 : List Nat := 9733 :: root260Tail23
def root260Tail21 : List Nat := 9091 :: root260Tail22
def root260Tail20 : List Nat := 8839 :: root260Tail21
def root260Tail19 : List Nat := 8599 :: root260Tail20
def root260Tail18 : List Nat := 7873 :: root260Tail19
def root260Tail17 : List Nat := 7867 :: root260Tail18
def root260Tail16 : List Nat := 7669 :: root260Tail17
def root260Tail15 : List Nat := 6733 :: root260Tail16
def root260Tail14 : List Nat := 6709 :: root260Tail15
def root260Tail13 : List Nat := 6661 :: root260Tail14
def root260Tail12 : List Nat := 6577 :: root260Tail13
def root260Tail11 : List Nat := 5743 :: root260Tail12
def root260Tail10 : List Nat := 5659 :: root260Tail11
def root260Tail9 : List Nat := 4651 :: root260Tail10
def root260Tail8 : List Nat := 4441 :: root260Tail9
def root260Tail7 : List Nat := 4273 :: root260Tail8
def root260Tail6 : List Nat := 3511 :: root260Tail7
def root260Tail5 : List Nat := 3499 :: root260Tail6
def root260Tail4 : List Nat := 3301 :: root260Tail5
def root260Tail3 : List Nat := 3181 :: root260Tail4
def root260Tail2 : List Nat := 2791 :: root260Tail3
def root260Tail1 : List Nat := 2707 :: root260Tail2
def root260Tail0 : List Nat := 2593 :: root260Tail1
def root260Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root260Candidates0_checked : candidates edge 4 1669 2593 root260Tail1 winner.bound = root260Candidates0 := by
  unfold candidates
  have row : edge 2593 = (SortedNatTree.contains FastRows.row2593) := by
    change Data.row2593.contains = (SortedNatTree.contains FastRows.row2593)
    exact FastRows.row2593_checked.symm
  rw [row]
  decide
theorem root260Step0 : scan edge (search edge 3) 4 [1669] 1669 root260Tail0 winner =
    scan edge (search edge 3) 4 [1669] 1669 root260Tail1 winner := by
  rw [root260Tail0, scan, if_neg (by decide), root260Candidates0_checked]
  change scan edge (search edge 3) 4 [1669] 1669 root260Tail1
    (if 3 ≤ root260Candidates0.length then search edge 3 [1669, 2593] root260Candidates0 4262 winner else winner) = _
  rw [if_neg (by decide)]
def root260Candidates1 : List Nat := [2791, 4273, 8599, 10729]
set_option maxRecDepth 8192 in
theorem root260Candidates1_checked : candidates edge 4 1669 2707 root260Tail2 winner.bound = root260Candidates1 := by
  unfold candidates
  have row : edge 2707 = (SortedNatTree.contains FastRows.row2707) := by
    change Data.row2707.contains = (SortedNatTree.contains FastRows.row2707)
    exact FastRows.row2707_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root260Child1 : search edge 3 [1669, 2707] root260Candidates1 4376 winner = winner := by decide
theorem root260Step1 : scan edge (search edge 3) 4 [1669] 1669 root260Tail1 winner =
    scan edge (search edge 3) 4 [1669] 1669 root260Tail2 winner := by
  rw [root260Tail1, scan, if_neg (by decide), root260Candidates1_checked]
  change scan edge (search edge 3) 4 [1669] 1669 root260Tail2
    (if 3 ≤ root260Candidates1.length then search edge 3 [1669, 2707] root260Candidates1 4376 winner else winner) = _
  rw [if_pos (by decide), root260Child1]
def root260Candidates2 : List Nat := [3499]
set_option maxRecDepth 8192 in
theorem root260Candidates2_checked : candidates edge 4 1669 2791 root260Tail3 winner.bound = root260Candidates2 := by
  unfold candidates
  have row : edge 2791 = (SortedNatTree.contains FastRows.row2791) := by
    change Data.row2791.contains = (SortedNatTree.contains FastRows.row2791)
    exact FastRows.row2791_checked.symm
  rw [row]
  decide
theorem root260Step2 : scan edge (search edge 3) 4 [1669] 1669 root260Tail2 winner =
    scan edge (search edge 3) 4 [1669] 1669 root260Tail3 winner := by
  rw [root260Tail2, scan, if_neg (by decide), root260Candidates2_checked]
  change scan edge (search edge 3) 4 [1669] 1669 root260Tail3
    (if 3 ≤ root260Candidates2.length then search edge 3 [1669, 2791] root260Candidates2 4460 winner else winner) = _
  rw [if_neg (by decide)]
def root260Candidates3 : List Nat := [4273, 6709, 10729]
set_option maxRecDepth 8192 in
theorem root260Candidates3_checked : candidates edge 4 1669 3181 root260Tail4 winner.bound = root260Candidates3 := by
  unfold candidates
  have row : edge 3181 = (SortedNatTree.contains FastRows.row3181) := by
    change Data.row3181.contains = (SortedNatTree.contains FastRows.row3181)
    exact FastRows.row3181_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root260Child3 : search edge 3 [1669, 3181] root260Candidates3 4850 winner = winner := by decide
theorem root260Step3 : scan edge (search edge 3) 4 [1669] 1669 root260Tail3 winner =
    scan edge (search edge 3) 4 [1669] 1669 root260Tail4 winner := by
  rw [root260Tail3, scan, if_neg (by decide), root260Candidates3_checked]
  change scan edge (search edge 3) 4 [1669] 1669 root260Tail4
    (if 3 ≤ root260Candidates3.length then search edge 3 [1669, 3181] root260Candidates3 4850 winner else winner) = _
  rw [if_pos (by decide), root260Child3]
def root260Candidates4 : List Nat := [4651]
set_option maxRecDepth 8192 in
theorem root260Candidates4_checked : candidates edge 4 1669 3301 root260Tail5 winner.bound = root260Candidates4 := by
  unfold candidates
  have row : edge 3301 = (SortedNatTree.contains FastRows.row3301) := by
    change Data.row3301.contains = (SortedNatTree.contains FastRows.row3301)
    exact FastRows.row3301_checked.symm
  rw [row]
  decide
theorem root260Step4 : scan edge (search edge 3) 4 [1669] 1669 root260Tail4 winner =
    scan edge (search edge 3) 4 [1669] 1669 root260Tail5 winner := by
  rw [root260Tail4, scan, if_neg (by decide), root260Candidates4_checked]
  change scan edge (search edge 3) 4 [1669] 1669 root260Tail5
    (if 3 ≤ root260Candidates4.length then search edge 3 [1669, 3301] root260Candidates4 4970 winner else winner) = _
  rw [if_neg (by decide)]
def root260Candidates5 : List Nat := [4651, 8599]
set_option maxRecDepth 8192 in
theorem root260Candidates5_checked : candidates edge 4 1669 3499 root260Tail6 winner.bound = root260Candidates5 := by
  unfold candidates
  have row : edge 3499 = (SortedNatTree.contains FastRows.row3499) := by
    change Data.row3499.contains = (SortedNatTree.contains FastRows.row3499)
    exact FastRows.row3499_checked.symm
  rw [row]
  decide
theorem root260Step5 : scan edge (search edge 3) 4 [1669] 1669 root260Tail5 winner =
    scan edge (search edge 3) 4 [1669] 1669 root260Tail6 winner := by
  rw [root260Tail5, scan, if_neg (by decide), root260Candidates5_checked]
  change scan edge (search edge 3) 4 [1669] 1669 root260Tail6
    (if 3 ≤ root260Candidates5.length then search edge 3 [1669, 3499] root260Candidates5 5168 winner else winner) = _
  rw [if_neg (by decide)]
def root260Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root260Candidates6_checked : candidates edge 4 1669 3511 root260Tail7 winner.bound = root260Candidates6 := by
  unfold candidates
  have row : edge 3511 = (SortedNatTree.contains FastRows.row3511) := by
    change Data.row3511.contains = (SortedNatTree.contains FastRows.row3511)
    exact FastRows.row3511_checked.symm
  rw [row]
  decide
theorem root260Step6 : scan edge (search edge 3) 4 [1669] 1669 root260Tail6 winner =
    scan edge (search edge 3) 4 [1669] 1669 root260Tail7 winner := by
  rw [root260Tail6, scan, if_neg (by decide), root260Candidates6_checked]
  change scan edge (search edge 3) 4 [1669] 1669 root260Tail7
    (if 3 ≤ root260Candidates6.length then search edge 3 [1669, 3511] root260Candidates6 5180 winner else winner) = _
  rw [if_neg (by decide)]
def root260Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root260Candidates7_checked : candidates edge 4 1669 4273 root260Tail8 winner.bound = root260Candidates7 := by
  unfold candidates
  have row : edge 4273 = (SortedNatTree.contains FastRows.row4273) := by
    change Data.row4273.contains = (SortedNatTree.contains FastRows.row4273)
    exact FastRows.row4273_checked.symm
  rw [row]
  decide
theorem root260Step7 : scan edge (search edge 3) 4 [1669] 1669 root260Tail7 winner =
    scan edge (search edge 3) 4 [1669] 1669 root260Tail8 winner := by
  rw [root260Tail7, scan, if_neg (by decide), root260Candidates7_checked]
  change scan edge (search edge 3) 4 [1669] 1669 root260Tail8
    (if 3 ≤ root260Candidates7.length then search edge 3 [1669, 4273] root260Candidates7 5942 winner else winner) = _
  rw [if_neg (by decide)]
def root260Candidates8 : List Nat := [5743, 7873]
set_option maxRecDepth 8192 in
theorem root260Candidates8_checked : candidates edge 4 1669 4441 root260Tail9 winner.bound = root260Candidates8 := by
  unfold candidates
  have row : edge 4441 = (SortedNatTree.contains FastRows.row4441) := by
    change Data.row4441.contains = (SortedNatTree.contains FastRows.row4441)
    exact FastRows.row4441_checked.symm
  rw [row]
  decide
theorem root260Step8 : scan edge (search edge 3) 4 [1669] 1669 root260Tail8 winner =
    scan edge (search edge 3) 4 [1669] 1669 root260Tail9 winner := by
  rw [root260Tail8, scan, if_neg (by decide), root260Candidates8_checked]
  change scan edge (search edge 3) 4 [1669] 1669 root260Tail9
    (if 3 ≤ root260Candidates8.length then search edge 3 [1669, 4441] root260Candidates8 6110 winner else winner) = _
  rw [if_neg (by decide)]
def root260Candidates9 : List Nat := [9091]
set_option maxRecDepth 8192 in
theorem root260Candidates9_checked : candidates edge 4 1669 4651 root260Tail10 winner.bound = root260Candidates9 := by
  unfold candidates
  have row : edge 4651 = (SortedNatTree.contains FastRows.row4651) := by
    change Data.row4651.contains = (SortedNatTree.contains FastRows.row4651)
    exact FastRows.row4651_checked.symm
  rw [row]
  decide
theorem root260Step9 : scan edge (search edge 3) 4 [1669] 1669 root260Tail9 winner =
    scan edge (search edge 3) 4 [1669] 1669 root260Tail10 winner := by
  rw [root260Tail9, scan, if_neg (by decide), root260Candidates9_checked]
  change scan edge (search edge 3) 4 [1669] 1669 root260Tail10
    (if 3 ≤ root260Candidates9.length then search edge 3 [1669, 4651] root260Candidates9 6320 winner else winner) = _
  rw [if_neg (by decide)]
def root260Candidates10 : List Nat := [6577, 7867]
set_option maxRecDepth 8192 in
theorem root260Candidates10_checked : candidates edge 4 1669 5659 root260Tail11 winner.bound = root260Candidates10 := by
  unfold candidates
  have row : edge 5659 = (SortedNatTree.contains FastRows.row5659) := by
    change Data.row5659.contains = (SortedNatTree.contains FastRows.row5659)
    exact FastRows.row5659_checked.symm
  rw [row]
  decide
theorem root260Step10 : scan edge (search edge 3) 4 [1669] 1669 root260Tail10 winner =
    scan edge (search edge 3) 4 [1669] 1669 root260Tail11 winner := by
  rw [root260Tail10, scan, if_neg (by decide), root260Candidates10_checked]
  change scan edge (search edge 3) 4 [1669] 1669 root260Tail11
    (if 3 ≤ root260Candidates10.length then search edge 3 [1669, 5659] root260Candidates10 7328 winner else winner) = _
  rw [if_neg (by decide)]
def root260Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root260Candidates11_checked : candidates edge 4 1669 5743 root260Tail12 winner.bound = root260Candidates11 := by
  unfold candidates
  have row : edge 5743 = (SortedNatTree.contains FastRows.row5743) := by
    change Data.row5743.contains = (SortedNatTree.contains FastRows.row5743)
    exact FastRows.row5743_checked.symm
  rw [row]
  decide
theorem root260Step11 : scan edge (search edge 3) 4 [1669] 1669 root260Tail11 winner =
    scan edge (search edge 3) 4 [1669] 1669 root260Tail12 winner := by
  rw [root260Tail11, scan, if_neg (by decide), root260Candidates11_checked]
  change scan edge (search edge 3) 4 [1669] 1669 root260Tail12
    (if 3 ≤ root260Candidates11.length then search edge 3 [1669, 5743] root260Candidates11 7412 winner else winner) = _
  rw [if_neg (by decide)]
theorem root260Scan12 : scan edge (search edge 3) 4 [1669] 1669 root260Tail12 winner = winner := by rw [root260Tail12, scan, if_pos (by decide)]
theorem root260Scan11 : scan edge (search edge 3) 4 [1669] 1669 root260Tail11 winner = winner :=
  (root260Step11).trans root260Scan12
theorem root260Scan10 : scan edge (search edge 3) 4 [1669] 1669 root260Tail10 winner = winner :=
  (root260Step10).trans root260Scan11
theorem root260Scan9 : scan edge (search edge 3) 4 [1669] 1669 root260Tail9 winner = winner :=
  (root260Step9).trans root260Scan10
theorem root260Scan8 : scan edge (search edge 3) 4 [1669] 1669 root260Tail8 winner = winner :=
  (root260Step8).trans root260Scan9
theorem root260Scan7 : scan edge (search edge 3) 4 [1669] 1669 root260Tail7 winner = winner :=
  (root260Step7).trans root260Scan8
theorem root260Scan6 : scan edge (search edge 3) 4 [1669] 1669 root260Tail6 winner = winner :=
  (root260Step6).trans root260Scan7
theorem root260Scan5 : scan edge (search edge 3) 4 [1669] 1669 root260Tail5 winner = winner :=
  (root260Step5).trans root260Scan6
theorem root260Scan4 : scan edge (search edge 3) 4 [1669] 1669 root260Tail4 winner = winner :=
  (root260Step4).trans root260Scan5
theorem root260Scan3 : scan edge (search edge 3) 4 [1669] 1669 root260Tail3 winner = winner :=
  (root260Step3).trans root260Scan4
theorem root260Scan2 : scan edge (search edge 3) 4 [1669] 1669 root260Tail2 winner = winner :=
  (root260Step2).trans root260Scan3
theorem root260Scan1 : scan edge (search edge 3) 4 [1669] 1669 root260Tail1 winner = winner :=
  (root260Step1).trans root260Scan2
theorem root260Scan0 : scan edge (search edge 3) 4 [1669] 1669 root260Tail0 winner = winner :=
  (root260Step0).trans root260Scan1
set_option maxRecDepth 4096 in
theorem root260Branch : search edge 4 [1669] root260Candidates 1669 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1669] 1669 root260Tail0 winner = winner
  exact root260Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
