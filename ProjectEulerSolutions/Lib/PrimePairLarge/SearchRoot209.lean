import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root209Candidates : List Nat := Data.row1301
set_option maxRecDepth 8192 in
theorem root209Candidates_checked : candidates edge 5 0 1301 tail210 winner.bound = root209Candidates := by
  apply candidates_eq_row edge 5 0 1301 tail210 root209Candidates winner.bound tail210_ordered
  · decide
  · decide
  · funext q
    rfl
def root209Tail47 : List Nat := []
def root209Tail46 : List Nat := 24419 :: root209Tail47
def root209Tail45 : List Nat := 23369 :: root209Tail46
def root209Tail44 : List Nat := 23099 :: root209Tail45
def root209Tail43 : List Nat := 20327 :: root209Tail44
def root209Tail42 : List Nat := 19541 :: root209Tail43
def root209Tail41 : List Nat := 19463 :: root209Tail42
def root209Tail40 : List Nat := 19373 :: root209Tail41
def root209Tail39 : List Nat := 17921 :: root209Tail40
def root209Tail38 : List Nat := 16421 :: root209Tail39
def root209Tail37 : List Nat := 16073 :: root209Tail38
def root209Tail36 : List Nat := 15329 :: root209Tail37
def root209Tail35 : List Nat := 15131 :: root209Tail36
def root209Tail34 : List Nat := 15077 :: root209Tail35
def root209Tail33 : List Nat := 14669 :: root209Tail34
def root209Tail32 : List Nat := 14591 :: root209Tail33
def root209Tail31 : List Nat := 14207 :: root209Tail32
def root209Tail30 : List Nat := 13259 :: root209Tail31
def root209Tail29 : List Nat := 13187 :: root209Tail30
def root209Tail28 : List Nat := 13103 :: root209Tail29
def root209Tail27 : List Nat := 12743 :: root209Tail28
def root209Tail26 : List Nat := 12503 :: root209Tail27
def root209Tail25 : List Nat := 12401 :: root209Tail26
def root209Tail24 : List Nat := 11243 :: root209Tail25
def root209Tail23 : List Nat := 10301 :: root209Tail24
def root209Tail22 : List Nat := 10223 :: root209Tail23
def root209Tail21 : List Nat := 10067 :: root209Tail22
def root209Tail20 : List Nat := 9857 :: root209Tail21
def root209Tail19 : List Nat := 9803 :: root209Tail20
def root209Tail18 : List Nat := 9551 :: root209Tail19
def root209Tail17 : List Nat := 7757 :: root209Tail18
def root209Tail16 : List Nat := 7643 :: root209Tail17
def root209Tail15 : List Nat := 7481 :: root209Tail16
def root209Tail14 : List Nat := 7283 :: root209Tail15
def root209Tail13 : List Nat := 7127 :: root209Tail14
def root209Tail12 : List Nat := 6011 :: root209Tail13
def root209Tail11 : List Nat := 5669 :: root209Tail12
def root209Tail10 : List Nat := 5531 :: root209Tail11
def root209Tail9 : List Nat := 5279 :: root209Tail10
def root209Tail8 : List Nat := 5003 :: root209Tail9
def root209Tail7 : List Nat := 4973 :: root209Tail8
def root209Tail6 : List Nat := 4583 :: root209Tail7
def root209Tail5 : List Nat := 2999 :: root209Tail6
def root209Tail4 : List Nat := 2969 :: root209Tail5
def root209Tail3 : List Nat := 2927 :: root209Tail4
def root209Tail2 : List Nat := 2579 :: root209Tail3
def root209Tail1 : List Nat := 2393 :: root209Tail2
def root209Tail0 : List Nat := 2141 :: root209Tail1
def root209Candidates0 : List Nat := [9551]
set_option maxRecDepth 8192 in
theorem root209Candidates0_checked : candidates edge 4 1301 2141 root209Tail1 winner.bound = root209Candidates0 := by
  unfold candidates
  have row : edge 2141 = (SortedNatTree.contains FastRows.row2141) := by
    change Data.row2141.contains = (SortedNatTree.contains FastRows.row2141)
    exact FastRows.row2141_checked.symm
  rw [row]
  decide
theorem root209Step0 : scan edge (search edge 3) 4 [1301] 1301 root209Tail0 winner =
    scan edge (search edge 3) 4 [1301] 1301 root209Tail1 winner := by
  rw [root209Tail0, scan, if_neg (by decide), root209Candidates0_checked]
  change scan edge (search edge 3) 4 [1301] 1301 root209Tail1
    (if 3 ≤ root209Candidates0.length then search edge 3 [1301, 2141] root209Candidates0 3442 winner else winner) = _
  rw [if_neg (by decide)]
def root209Candidates1 : List Nat := [2927, 7127, 9551, 16073]
set_option maxRecDepth 8192 in
theorem root209Candidates1_checked : candidates edge 4 1301 2393 root209Tail2 winner.bound = root209Candidates1 := by
  unfold candidates
  have row : edge 2393 = (SortedNatTree.contains FastRows.row2393) := by
    change Data.row2393.contains = (SortedNatTree.contains FastRows.row2393)
    exact FastRows.row2393_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root209Child1 : search edge 3 [1301, 2393] root209Candidates1 3694 winner = winner := by decide
theorem root209Step1 : scan edge (search edge 3) 4 [1301] 1301 root209Tail1 winner =
    scan edge (search edge 3) 4 [1301] 1301 root209Tail2 winner := by
  rw [root209Tail1, scan, if_neg (by decide), root209Candidates1_checked]
  change scan edge (search edge 3) 4 [1301] 1301 root209Tail2
    (if 3 ≤ root209Candidates1.length then search edge 3 [1301, 2393] root209Candidates1 3694 winner else winner) = _
  rw [if_pos (by decide), root209Child1]
def root209Candidates2 : List Nat := [19373, 19463]
set_option maxRecDepth 8192 in
theorem root209Candidates2_checked : candidates edge 4 1301 2579 root209Tail3 winner.bound = root209Candidates2 := by
  unfold candidates
  have row : edge 2579 = (SortedNatTree.contains FastRows.row2579) := by
    change Data.row2579.contains = (SortedNatTree.contains FastRows.row2579)
    exact FastRows.row2579_checked.symm
  rw [row]
  decide
theorem root209Step2 : scan edge (search edge 3) 4 [1301] 1301 root209Tail2 winner =
    scan edge (search edge 3) 4 [1301] 1301 root209Tail3 winner := by
  rw [root209Tail2, scan, if_neg (by decide), root209Candidates2_checked]
  change scan edge (search edge 3) 4 [1301] 1301 root209Tail3
    (if 3 ≤ root209Candidates2.length then search edge 3 [1301, 2579] root209Candidates2 3880 winner else winner) = _
  rw [if_neg (by decide)]
def root209Candidates3 : List Nat := [5531, 13103]
set_option maxRecDepth 8192 in
theorem root209Candidates3_checked : candidates edge 4 1301 2927 root209Tail4 winner.bound = root209Candidates3 := by
  unfold candidates
  have row : edge 2927 = (SortedNatTree.contains FastRows.row2927) := by
    change Data.row2927.contains = (SortedNatTree.contains FastRows.row2927)
    exact FastRows.row2927_checked.symm
  rw [row]
  decide
theorem root209Step3 : scan edge (search edge 3) 4 [1301] 1301 root209Tail3 winner =
    scan edge (search edge 3) 4 [1301] 1301 root209Tail4 winner := by
  rw [root209Tail3, scan, if_neg (by decide), root209Candidates3_checked]
  change scan edge (search edge 3) 4 [1301] 1301 root209Tail4
    (if 3 ≤ root209Candidates3.length then search edge 3 [1301, 2927] root209Candidates3 4228 winner else winner) = _
  rw [if_neg (by decide)]
def root209Candidates4 : List Nat := [10067]
set_option maxRecDepth 8192 in
theorem root209Candidates4_checked : candidates edge 4 1301 2969 root209Tail5 winner.bound = root209Candidates4 := by
  unfold candidates
  have row : edge 2969 = (SortedNatTree.contains FastRows.row2969) := by
    change Data.row2969.contains = (SortedNatTree.contains FastRows.row2969)
    exact FastRows.row2969_checked.symm
  rw [row]
  decide
theorem root209Step4 : scan edge (search edge 3) 4 [1301] 1301 root209Tail4 winner =
    scan edge (search edge 3) 4 [1301] 1301 root209Tail5 winner := by
  rw [root209Tail4, scan, if_neg (by decide), root209Candidates4_checked]
  change scan edge (search edge 3) 4 [1301] 1301 root209Tail5
    (if 3 ≤ root209Candidates4.length then search edge 3 [1301, 2969] root209Candidates4 4270 winner else winner) = _
  rw [if_neg (by decide)]
def root209Candidates5 : List Nat := [5003, 7481, 14669]
set_option maxRecDepth 8192 in
theorem root209Candidates5_checked : candidates edge 4 1301 2999 root209Tail6 winner.bound = root209Candidates5 := by
  unfold candidates
  have row : edge 2999 = (SortedNatTree.contains FastRows.row2999) := by
    change Data.row2999.contains = (SortedNatTree.contains FastRows.row2999)
    exact FastRows.row2999_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root209Child5 : search edge 3 [1301, 2999] root209Candidates5 4300 winner = winner := by decide
theorem root209Step5 : scan edge (search edge 3) 4 [1301] 1301 root209Tail5 winner =
    scan edge (search edge 3) 4 [1301] 1301 root209Tail6 winner := by
  rw [root209Tail5, scan, if_neg (by decide), root209Candidates5_checked]
  change scan edge (search edge 3) 4 [1301] 1301 root209Tail6
    (if 3 ≤ root209Candidates5.length then search edge 3 [1301, 2999] root209Candidates5 4300 winner else winner) = _
  rw [if_pos (by decide), root209Child5]
def root209Candidates6 : List Nat := [5003, 9551]
set_option maxRecDepth 8192 in
theorem root209Candidates6_checked : candidates edge 4 1301 4583 root209Tail7 winner.bound = root209Candidates6 := by
  unfold candidates
  have row : edge 4583 = (SortedNatTree.contains FastRows.row4583) := by
    change Data.row4583.contains = (SortedNatTree.contains FastRows.row4583)
    exact FastRows.row4583_checked.symm
  rw [row]
  decide
theorem root209Step6 : scan edge (search edge 3) 4 [1301] 1301 root209Tail6 winner =
    scan edge (search edge 3) 4 [1301] 1301 root209Tail7 winner := by
  rw [root209Tail6, scan, if_neg (by decide), root209Candidates6_checked]
  change scan edge (search edge 3) 4 [1301] 1301 root209Tail7
    (if 3 ≤ root209Candidates6.length then search edge 3 [1301, 4583] root209Candidates6 5884 winner else winner) = _
  rw [if_neg (by decide)]
def root209Candidates7 : List Nat := [10223]
set_option maxRecDepth 8192 in
theorem root209Candidates7_checked : candidates edge 4 1301 4973 root209Tail8 winner.bound = root209Candidates7 := by
  unfold candidates
  have row : edge 4973 = (SortedNatTree.contains FastRows.row4973) := by
    change Data.row4973.contains = (SortedNatTree.contains FastRows.row4973)
    exact FastRows.row4973_checked.symm
  rw [row]
  decide
theorem root209Step7 : scan edge (search edge 3) 4 [1301] 1301 root209Tail7 winner =
    scan edge (search edge 3) 4 [1301] 1301 root209Tail8 winner := by
  rw [root209Tail7, scan, if_neg (by decide), root209Candidates7_checked]
  change scan edge (search edge 3) 4 [1301] 1301 root209Tail8
    (if 3 ≤ root209Candidates7.length then search edge 3 [1301, 4973] root209Candidates7 6274 winner else winner) = _
  rw [if_neg (by decide)]
def root209Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root209Candidates8_checked : candidates edge 4 1301 5003 root209Tail9 winner.bound = root209Candidates8 := by
  unfold candidates
  have row : edge 5003 = (SortedNatTree.contains FastRows.row5003) := by
    change Data.row5003.contains = (SortedNatTree.contains FastRows.row5003)
    exact FastRows.row5003_checked.symm
  rw [row]
  decide
theorem root209Step8 : scan edge (search edge 3) 4 [1301] 1301 root209Tail8 winner =
    scan edge (search edge 3) 4 [1301] 1301 root209Tail9 winner := by
  rw [root209Tail8, scan, if_neg (by decide), root209Candidates8_checked]
  change scan edge (search edge 3) 4 [1301] 1301 root209Tail9
    (if 3 ≤ root209Candidates8.length then search edge 3 [1301, 5003] root209Candidates8 6304 winner else winner) = _
  rw [if_neg (by decide)]
def root209Candidates9 : List Nat := [12503]
set_option maxRecDepth 8192 in
theorem root209Candidates9_checked : candidates edge 4 1301 5279 root209Tail10 winner.bound = root209Candidates9 := by
  unfold candidates
  have row : edge 5279 = (SortedNatTree.contains FastRows.row5279) := by
    change Data.row5279.contains = (SortedNatTree.contains FastRows.row5279)
    exact FastRows.row5279_checked.symm
  rw [row]
  decide
theorem root209Step9 : scan edge (search edge 3) 4 [1301] 1301 root209Tail9 winner =
    scan edge (search edge 3) 4 [1301] 1301 root209Tail10 winner := by
  rw [root209Tail9, scan, if_neg (by decide), root209Candidates9_checked]
  change scan edge (search edge 3) 4 [1301] 1301 root209Tail10
    (if 3 ≤ root209Candidates9.length then search edge 3 [1301, 5279] root209Candidates9 6580 winner else winner) = _
  rw [if_neg (by decide)]
def root209Candidates10 : List Nat := [15131]
set_option maxRecDepth 8192 in
theorem root209Candidates10_checked : candidates edge 4 1301 5531 root209Tail11 winner.bound = root209Candidates10 := by
  unfold candidates
  have row : edge 5531 = (SortedNatTree.contains FastRows.row5531) := by
    change Data.row5531.contains = (SortedNatTree.contains FastRows.row5531)
    exact FastRows.row5531_checked.symm
  rw [row]
  decide
theorem root209Step10 : scan edge (search edge 3) 4 [1301] 1301 root209Tail10 winner =
    scan edge (search edge 3) 4 [1301] 1301 root209Tail11 winner := by
  rw [root209Tail10, scan, if_neg (by decide), root209Candidates10_checked]
  change scan edge (search edge 3) 4 [1301] 1301 root209Tail11
    (if 3 ≤ root209Candidates10.length then search edge 3 [1301, 5531] root209Candidates10 6832 winner else winner) = _
  rw [if_neg (by decide)]
def root209Candidates11 : List Nat := [6011, 12743, 16421]
set_option maxRecDepth 8192 in
theorem root209Candidates11_checked : candidates edge 4 1301 5669 root209Tail12 winner.bound = root209Candidates11 := by
  unfold candidates
  have row : edge 5669 = (SortedNatTree.contains FastRows.row5669) := by
    change Data.row5669.contains = (SortedNatTree.contains FastRows.row5669)
    exact FastRows.row5669_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root209Child11 : search edge 3 [1301, 5669] root209Candidates11 6970 winner = winner := by decide
theorem root209Step11 : scan edge (search edge 3) 4 [1301] 1301 root209Tail11 winner =
    scan edge (search edge 3) 4 [1301] 1301 root209Tail12 winner := by
  rw [root209Tail11, scan, if_neg (by decide), root209Candidates11_checked]
  change scan edge (search edge 3) 4 [1301] 1301 root209Tail12
    (if 3 ≤ root209Candidates11.length then search edge 3 [1301, 5669] root209Candidates11 6970 winner else winner) = _
  rw [if_pos (by decide), root209Child11]
def root209Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root209Candidates12_checked : candidates edge 4 1301 6011 root209Tail13 winner.bound = root209Candidates12 := by
  unfold candidates
  have row : edge 6011 = (SortedNatTree.contains FastRows.row6011) := by
    change Data.row6011.contains = (SortedNatTree.contains FastRows.row6011)
    exact FastRows.row6011_checked.symm
  rw [row]
  decide
theorem root209Step12 : scan edge (search edge 3) 4 [1301] 1301 root209Tail12 winner =
    scan edge (search edge 3) 4 [1301] 1301 root209Tail13 winner := by
  rw [root209Tail12, scan, if_neg (by decide), root209Candidates12_checked]
  change scan edge (search edge 3) 4 [1301] 1301 root209Tail13
    (if 3 ≤ root209Candidates12.length then search edge 3 [1301, 6011] root209Candidates12 7312 winner else winner) = _
  rw [if_neg (by decide)]
theorem root209Scan13 : scan edge (search edge 3) 4 [1301] 1301 root209Tail13 winner = winner := by rw [root209Tail13, scan, if_pos (by decide)]
theorem root209Scan12 : scan edge (search edge 3) 4 [1301] 1301 root209Tail12 winner = winner :=
  (root209Step12).trans root209Scan13
theorem root209Scan11 : scan edge (search edge 3) 4 [1301] 1301 root209Tail11 winner = winner :=
  (root209Step11).trans root209Scan12
theorem root209Scan10 : scan edge (search edge 3) 4 [1301] 1301 root209Tail10 winner = winner :=
  (root209Step10).trans root209Scan11
theorem root209Scan9 : scan edge (search edge 3) 4 [1301] 1301 root209Tail9 winner = winner :=
  (root209Step9).trans root209Scan10
theorem root209Scan8 : scan edge (search edge 3) 4 [1301] 1301 root209Tail8 winner = winner :=
  (root209Step8).trans root209Scan9
theorem root209Scan7 : scan edge (search edge 3) 4 [1301] 1301 root209Tail7 winner = winner :=
  (root209Step7).trans root209Scan8
theorem root209Scan6 : scan edge (search edge 3) 4 [1301] 1301 root209Tail6 winner = winner :=
  (root209Step6).trans root209Scan7
theorem root209Scan5 : scan edge (search edge 3) 4 [1301] 1301 root209Tail5 winner = winner :=
  (root209Step5).trans root209Scan6
theorem root209Scan4 : scan edge (search edge 3) 4 [1301] 1301 root209Tail4 winner = winner :=
  (root209Step4).trans root209Scan5
theorem root209Scan3 : scan edge (search edge 3) 4 [1301] 1301 root209Tail3 winner = winner :=
  (root209Step3).trans root209Scan4
theorem root209Scan2 : scan edge (search edge 3) 4 [1301] 1301 root209Tail2 winner = winner :=
  (root209Step2).trans root209Scan3
theorem root209Scan1 : scan edge (search edge 3) 4 [1301] 1301 root209Tail1 winner = winner :=
  (root209Step1).trans root209Scan2
theorem root209Scan0 : scan edge (search edge 3) 4 [1301] 1301 root209Tail0 winner = winner :=
  (root209Step0).trans root209Scan1
set_option maxRecDepth 4096 in
theorem root209Branch : search edge 4 [1301] root209Candidates 1301 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1301] 1301 root209Tail0 winner = winner
  exact root209Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
