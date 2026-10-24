import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root294Candidates : List Nat := Data.row1951
set_option maxRecDepth 8192 in
theorem root294Candidates_checked : candidates edge 5 0 1951 tail295 winner.bound = root294Candidates := by
  apply candidates_eq_row edge 5 0 1951 tail295 root294Candidates winner.bound tail295_ordered
  · decide
  · decide
  · funext q
    rfl
def root294Tail46 : List Nat := []
def root294Tail45 : List Nat := 22483 :: root294Tail46
def root294Tail44 : List Nat := 22303 :: root294Tail45
def root294Tail43 : List Nat := 22171 :: root294Tail44
def root294Tail42 : List Nat := 21187 :: root294Tail43
def root294Tail41 : List Nat := 20047 :: root294Tail42
def root294Tail40 : List Nat := 19237 :: root294Tail41
def root294Tail39 : List Nat := 19087 :: root294Tail40
def root294Tail38 : List Nat := 18541 :: root294Tail39
def root294Tail37 : List Nat := 18457 :: root294Tail38
def root294Tail36 : List Nat := 17599 :: root294Tail37
def root294Tail35 : List Nat := 17383 :: root294Tail36
def root294Tail34 : List Nat := 17011 :: root294Tail35
def root294Tail33 : List Nat := 16087 :: root294Tail34
def root294Tail32 : List Nat := 15031 :: root294Tail33
def root294Tail31 : List Nat := 14149 :: root294Tail32
def root294Tail30 : List Nat := 13903 :: root294Tail31
def root294Tail29 : List Nat := 13687 :: root294Tail30
def root294Tail28 : List Nat := 12301 :: root294Tail29
def root294Tail27 : List Nat := 12277 :: root294Tail28
def root294Tail26 : List Nat := 11863 :: root294Tail27
def root294Tail25 : List Nat := 11443 :: root294Tail26
def root294Tail24 : List Nat := 10939 :: root294Tail25
def root294Tail23 : List Nat := 10429 :: root294Tail24
def root294Tail22 : List Nat := 9901 :: root294Tail23
def root294Tail21 : List Nat := 9883 :: root294Tail22
def root294Tail20 : List Nat := 9421 :: root294Tail21
def root294Tail19 : List Nat := 8731 :: root294Tail20
def root294Tail18 : List Nat := 8581 :: root294Tail19
def root294Tail17 : List Nat := 8167 :: root294Tail18
def root294Tail16 : List Nat := 7489 :: root294Tail17
def root294Tail15 : List Nat := 7027 :: root294Tail16
def root294Tail14 : List Nat := 6823 :: root294Tail15
def root294Tail13 : List Nat := 6547 :: root294Tail14
def root294Tail12 : List Nat := 6379 :: root294Tail13
def root294Tail11 : List Nat := 6151 :: root294Tail12
def root294Tail10 : List Nat := 6037 :: root294Tail11
def root294Tail9 : List Nat := 5437 :: root294Tail10
def root294Tail8 : List Nat := 5227 :: root294Tail9
def root294Tail7 : List Nat := 4909 :: root294Tail8
def root294Tail6 : List Nat := 4783 :: root294Tail7
def root294Tail5 : List Nat := 4549 :: root294Tail6
def root294Tail4 : List Nat := 3733 :: root294Tail5
def root294Tail3 : List Nat := 3643 :: root294Tail4
def root294Tail2 : List Nat := 3547 :: root294Tail3
def root294Tail1 : List Nat := 2683 :: root294Tail2
def root294Tail0 : List Nat := 2341 :: root294Tail1
def root294Candidates0 : List Nat := [7489, 21187]
set_option maxRecDepth 8192 in
theorem root294Candidates0_checked : candidates edge 4 1951 2341 root294Tail1 winner.bound = root294Candidates0 := by
  unfold candidates
  have row : edge 2341 = (SortedNatTree.contains FastRows.row2341) := by
    change Data.row2341.contains = (SortedNatTree.contains FastRows.row2341)
    exact FastRows.row2341_checked.symm
  rw [row]
  decide
theorem root294Step0 : scan edge (search edge 3) 4 [1951] 1951 root294Tail0 winner =
    scan edge (search edge 3) 4 [1951] 1951 root294Tail1 winner := by
  rw [root294Tail0, scan, if_neg (by decide), root294Candidates0_checked]
  change scan edge (search edge 3) 4 [1951] 1951 root294Tail1
    (if 3 ≤ root294Candidates0.length then search edge 3 [1951, 2341] root294Candidates0 4292 winner else winner) = _
  rw [if_neg (by decide)]
def root294Candidates1 : List Nat := [8167]
set_option maxRecDepth 8192 in
theorem root294Candidates1_checked : candidates edge 4 1951 2683 root294Tail2 winner.bound = root294Candidates1 := by
  unfold candidates
  have row : edge 2683 = (SortedNatTree.contains FastRows.row2683) := by
    change Data.row2683.contains = (SortedNatTree.contains FastRows.row2683)
    exact FastRows.row2683_checked.symm
  rw [row]
  decide
theorem root294Step1 : scan edge (search edge 3) 4 [1951] 1951 root294Tail1 winner =
    scan edge (search edge 3) 4 [1951] 1951 root294Tail2 winner := by
  rw [root294Tail1, scan, if_neg (by decide), root294Candidates1_checked]
  change scan edge (search edge 3) 4 [1951] 1951 root294Tail2
    (if 3 ≤ root294Candidates1.length then search edge 3 [1951, 2683] root294Candidates1 4634 winner else winner) = _
  rw [if_neg (by decide)]
def root294Candidates2 : List Nat := [3643, 6151, 18457]
set_option maxRecDepth 8192 in
theorem root294Candidates2_checked : candidates edge 4 1951 3547 root294Tail3 winner.bound = root294Candidates2 := by
  unfold candidates
  have row : edge 3547 = (SortedNatTree.contains FastRows.row3547) := by
    change Data.row3547.contains = (SortedNatTree.contains FastRows.row3547)
    exact FastRows.row3547_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root294Child2 : search edge 3 [1951, 3547] root294Candidates2 5498 winner = winner := by decide
theorem root294Step2 : scan edge (search edge 3) 4 [1951] 1951 root294Tail2 winner =
    scan edge (search edge 3) 4 [1951] 1951 root294Tail3 winner := by
  rw [root294Tail2, scan, if_neg (by decide), root294Candidates2_checked]
  change scan edge (search edge 3) 4 [1951] 1951 root294Tail3
    (if 3 ≤ root294Candidates2.length then search edge 3 [1951, 3547] root294Candidates2 5498 winner else winner) = _
  rw [if_pos (by decide), root294Child2]
def root294Candidates3 : List Nat := [8581, 13687]
set_option maxRecDepth 8192 in
theorem root294Candidates3_checked : candidates edge 4 1951 3643 root294Tail4 winner.bound = root294Candidates3 := by
  unfold candidates
  have row : edge 3643 = (SortedNatTree.contains FastRows.row3643) := by
    change Data.row3643.contains = (SortedNatTree.contains FastRows.row3643)
    exact FastRows.row3643_checked.symm
  rw [row]
  decide
theorem root294Step3 : scan edge (search edge 3) 4 [1951] 1951 root294Tail3 winner =
    scan edge (search edge 3) 4 [1951] 1951 root294Tail4 winner := by
  rw [root294Tail3, scan, if_neg (by decide), root294Candidates3_checked]
  change scan edge (search edge 3) 4 [1951] 1951 root294Tail4
    (if 3 ≤ root294Candidates3.length then search edge 3 [1951, 3643] root294Candidates3 5594 winner else winner) = _
  rw [if_neg (by decide)]
def root294Candidates4 : List Nat := [6151]
set_option maxRecDepth 8192 in
theorem root294Candidates4_checked : candidates edge 4 1951 3733 root294Tail5 winner.bound = root294Candidates4 := by
  unfold candidates
  have row : edge 3733 = (SortedNatTree.contains FastRows.row3733) := by
    change Data.row3733.contains = (SortedNatTree.contains FastRows.row3733)
    exact FastRows.row3733_checked.symm
  rw [row]
  decide
theorem root294Step4 : scan edge (search edge 3) 4 [1951] 1951 root294Tail4 winner =
    scan edge (search edge 3) 4 [1951] 1951 root294Tail5 winner := by
  rw [root294Tail4, scan, if_neg (by decide), root294Candidates4_checked]
  change scan edge (search edge 3) 4 [1951] 1951 root294Tail5
    (if 3 ≤ root294Candidates4.length then search edge 3 [1951, 3733] root294Candidates4 5684 winner else winner) = _
  rw [if_neg (by decide)]
def root294Candidates5 : List Nat := [6151]
set_option maxRecDepth 8192 in
theorem root294Candidates5_checked : candidates edge 4 1951 4549 root294Tail6 winner.bound = root294Candidates5 := by
  unfold candidates
  have row : edge 4549 = (SortedNatTree.contains FastRows.row4549) := by
    change Data.row4549.contains = (SortedNatTree.contains FastRows.row4549)
    exact FastRows.row4549_checked.symm
  rw [row]
  decide
theorem root294Step5 : scan edge (search edge 3) 4 [1951] 1951 root294Tail5 winner =
    scan edge (search edge 3) 4 [1951] 1951 root294Tail6 winner := by
  rw [root294Tail5, scan, if_neg (by decide), root294Candidates5_checked]
  change scan edge (search edge 3) 4 [1951] 1951 root294Tail6
    (if 3 ≤ root294Candidates5.length then search edge 3 [1951, 4549] root294Candidates5 6500 winner else winner) = _
  rw [if_neg (by decide)]
def root294Candidates6 : List Nat := [16087]
set_option maxRecDepth 8192 in
theorem root294Candidates6_checked : candidates edge 4 1951 4783 root294Tail7 winner.bound = root294Candidates6 := by
  unfold candidates
  have row : edge 4783 = (SortedNatTree.contains FastRows.row4783) := by
    change Data.row4783.contains = (SortedNatTree.contains FastRows.row4783)
    exact FastRows.row4783_checked.symm
  rw [row]
  decide
theorem root294Step6 : scan edge (search edge 3) 4 [1951] 1951 root294Tail6 winner =
    scan edge (search edge 3) 4 [1951] 1951 root294Tail7 winner := by
  rw [root294Tail6, scan, if_neg (by decide), root294Candidates6_checked]
  change scan edge (search edge 3) 4 [1951] 1951 root294Tail7
    (if 3 ≤ root294Candidates6.length then search edge 3 [1951, 4783] root294Candidates6 6734 winner else winner) = _
  rw [if_neg (by decide)]
def root294Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root294Candidates7_checked : candidates edge 4 1951 4909 root294Tail8 winner.bound = root294Candidates7 := by
  unfold candidates
  have row : edge 4909 = (SortedNatTree.contains FastRows.row4909) := by
    change Data.row4909.contains = (SortedNatTree.contains FastRows.row4909)
    exact FastRows.row4909_checked.symm
  rw [row]
  decide
theorem root294Step7 : scan edge (search edge 3) 4 [1951] 1951 root294Tail7 winner =
    scan edge (search edge 3) 4 [1951] 1951 root294Tail8 winner := by
  rw [root294Tail7, scan, if_neg (by decide), root294Candidates7_checked]
  change scan edge (search edge 3) 4 [1951] 1951 root294Tail8
    (if 3 ≤ root294Candidates7.length then search edge 3 [1951, 4909] root294Candidates7 6860 winner else winner) = _
  rw [if_neg (by decide)]
def root294Candidates8 : List Nat := [6379, 18541]
set_option maxRecDepth 8192 in
theorem root294Candidates8_checked : candidates edge 4 1951 5227 root294Tail9 winner.bound = root294Candidates8 := by
  unfold candidates
  have row : edge 5227 = (SortedNatTree.contains FastRows.row5227) := by
    change Data.row5227.contains = (SortedNatTree.contains FastRows.row5227)
    exact FastRows.row5227_checked.symm
  rw [row]
  decide
theorem root294Step8 : scan edge (search edge 3) 4 [1951] 1951 root294Tail8 winner =
    scan edge (search edge 3) 4 [1951] 1951 root294Tail9 winner := by
  rw [root294Tail8, scan, if_neg (by decide), root294Candidates8_checked]
  change scan edge (search edge 3) 4 [1951] 1951 root294Tail9
    (if 3 ≤ root294Candidates8.length then search edge 3 [1951, 5227] root294Candidates8 7178 winner else winner) = _
  rw [if_neg (by decide)]
def root294Candidates9 : List Nat := [9883, 11863, 18541]
set_option maxRecDepth 8192 in
theorem root294Candidates9_checked : candidates edge 4 1951 5437 root294Tail10 winner.bound = root294Candidates9 := by
  unfold candidates
  have row : edge 5437 = (SortedNatTree.contains FastRows.row5437) := by
    change Data.row5437.contains = (SortedNatTree.contains FastRows.row5437)
    exact FastRows.row5437_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root294Child9 : search edge 3 [1951, 5437] root294Candidates9 7388 winner = winner := by decide
theorem root294Step9 : scan edge (search edge 3) 4 [1951] 1951 root294Tail9 winner =
    scan edge (search edge 3) 4 [1951] 1951 root294Tail10 winner := by
  rw [root294Tail9, scan, if_neg (by decide), root294Candidates9_checked]
  change scan edge (search edge 3) 4 [1951] 1951 root294Tail10
    (if 3 ≤ root294Candidates9.length then search edge 3 [1951, 5437] root294Candidates9 7388 winner else winner) = _
  rw [if_pos (by decide), root294Child9]
theorem root294Scan10 : scan edge (search edge 3) 4 [1951] 1951 root294Tail10 winner = winner := by rw [root294Tail10, scan, if_pos (by decide)]
theorem root294Scan9 : scan edge (search edge 3) 4 [1951] 1951 root294Tail9 winner = winner :=
  (root294Step9).trans root294Scan10
theorem root294Scan8 : scan edge (search edge 3) 4 [1951] 1951 root294Tail8 winner = winner :=
  (root294Step8).trans root294Scan9
theorem root294Scan7 : scan edge (search edge 3) 4 [1951] 1951 root294Tail7 winner = winner :=
  (root294Step7).trans root294Scan8
theorem root294Scan6 : scan edge (search edge 3) 4 [1951] 1951 root294Tail6 winner = winner :=
  (root294Step6).trans root294Scan7
theorem root294Scan5 : scan edge (search edge 3) 4 [1951] 1951 root294Tail5 winner = winner :=
  (root294Step5).trans root294Scan6
theorem root294Scan4 : scan edge (search edge 3) 4 [1951] 1951 root294Tail4 winner = winner :=
  (root294Step4).trans root294Scan5
theorem root294Scan3 : scan edge (search edge 3) 4 [1951] 1951 root294Tail3 winner = winner :=
  (root294Step3).trans root294Scan4
theorem root294Scan2 : scan edge (search edge 3) 4 [1951] 1951 root294Tail2 winner = winner :=
  (root294Step2).trans root294Scan3
theorem root294Scan1 : scan edge (search edge 3) 4 [1951] 1951 root294Tail1 winner = winner :=
  (root294Step1).trans root294Scan2
theorem root294Scan0 : scan edge (search edge 3) 4 [1951] 1951 root294Tail0 winner = winner :=
  (root294Step0).trans root294Scan1
set_option maxRecDepth 4096 in
theorem root294Branch : search edge 4 [1951] root294Candidates 1951 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1951] 1951 root294Tail0 winner = winner
  exact root294Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
