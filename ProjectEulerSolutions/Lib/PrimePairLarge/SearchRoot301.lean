import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root301Candidates : List Nat := Data.row2003
set_option maxRecDepth 8192 in
theorem root301Candidates_checked : candidates edge 5 0 2003 tail302 winner.bound = root301Candidates := by
  apply candidates_eq_row edge 5 0 2003 tail302 root301Candidates winner.bound tail302_ordered
  · decide
  · decide
  · funext q
    rfl
def root301Tail53 : List Nat := []
def root301Tail52 : List Nat := 22541 :: root301Tail53
def root301Tail51 : List Nat := 22259 :: root301Tail52
def root301Tail50 : List Nat := 21419 :: root301Tail51
def root301Tail49 : List Nat := 20693 :: root301Tail50
def root301Tail48 : List Nat := 19937 :: root301Tail49
def root301Tail47 : List Nat := 19583 :: root301Tail48
def root301Tail46 : List Nat := 18749 :: root301Tail47
def root301Tail45 : List Nat := 17597 :: root301Tail46
def root301Tail44 : List Nat := 17573 :: root301Tail45
def root301Tail43 : List Nat := 16673 :: root301Tail44
def root301Tail42 : List Nat := 16607 :: root301Tail43
def root301Tail41 : List Nat := 15959 :: root301Tail42
def root301Tail40 : List Nat := 15683 :: root301Tail41
def root301Tail39 : List Nat := 15329 :: root301Tail40
def root301Tail38 : List Nat := 14867 :: root301Tail39
def root301Tail37 : List Nat := 14669 :: root301Tail38
def root301Tail36 : List Nat := 13931 :: root301Tail37
def root301Tail35 : List Nat := 13721 :: root301Tail36
def root301Tail34 : List Nat := 13313 :: root301Tail35
def root301Tail33 : List Nat := 12713 :: root301Tail34
def root301Tail32 : List Nat := 11831 :: root301Tail33
def root301Tail31 : List Nat := 11777 :: root301Tail32
def root301Tail30 : List Nat := 10973 :: root301Tail31
def root301Tail29 : List Nat := 10739 :: root301Tail30
def root301Tail28 : List Nat := 10601 :: root301Tail29
def root301Tail27 : List Nat := 10343 :: root301Tail28
def root301Tail26 : List Nat := 10091 :: root301Tail27
def root301Tail25 : List Nat := 8747 :: root301Tail26
def root301Tail24 : List Nat := 8609 :: root301Tail25
def root301Tail23 : List Nat := 8297 :: root301Tail24
def root301Tail22 : List Nat := 8273 :: root301Tail23
def root301Tail21 : List Nat := 8171 :: root301Tail22
def root301Tail20 : List Nat := 7211 :: root301Tail21
def root301Tail19 : List Nat := 7013 :: root301Tail20
def root301Tail18 : List Nat := 6983 :: root301Tail19
def root301Tail17 : List Nat := 6833 :: root301Tail18
def root301Tail16 : List Nat := 6791 :: root301Tail17
def root301Tail15 : List Nat := 6689 :: root301Tail16
def root301Tail14 : List Nat := 6563 :: root301Tail15
def root301Tail13 : List Nat := 5693 :: root301Tail14
def root301Tail12 : List Nat := 5657 :: root301Tail13
def root301Tail11 : List Nat := 5651 :: root301Tail12
def root301Tail10 : List Nat := 5189 :: root301Tail11
def root301Tail9 : List Nat := 5153 :: root301Tail10
def root301Tail8 : List Nat := 5051 :: root301Tail9
def root301Tail7 : List Nat := 5039 :: root301Tail8
def root301Tail6 : List Nat := 4673 :: root301Tail7
def root301Tail5 : List Nat := 4271 :: root301Tail6
def root301Tail4 : List Nat := 3557 :: root301Tail5
def root301Tail3 : List Nat := 3413 :: root301Tail4
def root301Tail2 : List Nat := 2543 :: root301Tail3
def root301Tail1 : List Nat := 2237 :: root301Tail2
def root301Tail0 : List Nat := 2087 :: root301Tail1
def root301Candidates0 : List Nat := [6689, 11831]
set_option maxRecDepth 8192 in
theorem root301Candidates0_checked : candidates edge 4 2003 2087 root301Tail1 winner.bound = root301Candidates0 := by
  unfold candidates
  have row : edge 2087 = (SortedNatTree.contains FastRows.row2087) := by
    change Data.row2087.contains = (SortedNatTree.contains FastRows.row2087)
    exact FastRows.row2087_checked.symm
  rw [row]
  decide
theorem root301Step0 : scan edge (search edge 3) 4 [2003] 2003 root301Tail0 winner =
    scan edge (search edge 3) 4 [2003] 2003 root301Tail1 winner := by
  rw [root301Tail0, scan, if_neg (by decide), root301Candidates0_checked]
  change scan edge (search edge 3) 4 [2003] 2003 root301Tail1
    (if 3 ≤ root301Candidates0.length then search edge 3 [2003, 2087] root301Candidates0 4090 winner else winner) = _
  rw [if_neg (by decide)]
def root301Candidates1 : List Nat := [5693, 6689, 6791]
set_option maxRecDepth 8192 in
theorem root301Candidates1_checked : candidates edge 4 2003 2237 root301Tail2 winner.bound = root301Candidates1 := by
  unfold candidates
  have row : edge 2237 = (SortedNatTree.contains FastRows.row2237) := by
    change Data.row2237.contains = (SortedNatTree.contains FastRows.row2237)
    exact FastRows.row2237_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root301Child1 : search edge 3 [2003, 2237] root301Candidates1 4240 winner = winner := by decide
theorem root301Step1 : scan edge (search edge 3) 4 [2003] 2003 root301Tail1 winner =
    scan edge (search edge 3) 4 [2003] 2003 root301Tail2 winner := by
  rw [root301Tail1, scan, if_neg (by decide), root301Candidates1_checked]
  change scan edge (search edge 3) 4 [2003] 2003 root301Tail2
    (if 3 ≤ root301Candidates1.length then search edge 3 [2003, 2237] root301Candidates1 4240 winner else winner) = _
  rw [if_pos (by decide), root301Child1]
def root301Candidates2 : List Nat := [5051, 5153, 6791, 15329]
set_option maxRecDepth 8192 in
theorem root301Candidates2_checked : candidates edge 4 2003 2543 root301Tail3 winner.bound = root301Candidates2 := by
  unfold candidates
  have row : edge 2543 = (SortedNatTree.contains FastRows.row2543) := by
    change Data.row2543.contains = (SortedNatTree.contains FastRows.row2543)
    exact FastRows.row2543_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root301Child2 : search edge 3 [2003, 2543] root301Candidates2 4546 winner = winner := by decide
theorem root301Step2 : scan edge (search edge 3) 4 [2003] 2003 root301Tail2 winner =
    scan edge (search edge 3) 4 [2003] 2003 root301Tail3 winner := by
  rw [root301Tail2, scan, if_neg (by decide), root301Candidates2_checked]
  change scan edge (search edge 3) 4 [2003] 2003 root301Tail3
    (if 3 ≤ root301Candidates2.length then search edge 3 [2003, 2543] root301Candidates2 4546 winner else winner) = _
  rw [if_pos (by decide), root301Child2]
def root301Candidates3 : List Nat := [6689, 7211]
set_option maxRecDepth 8192 in
theorem root301Candidates3_checked : candidates edge 4 2003 3413 root301Tail4 winner.bound = root301Candidates3 := by
  unfold candidates
  have row : edge 3413 = (SortedNatTree.contains FastRows.row3413) := by
    change Data.row3413.contains = (SortedNatTree.contains FastRows.row3413)
    exact FastRows.row3413_checked.symm
  rw [row]
  decide
theorem root301Step3 : scan edge (search edge 3) 4 [2003] 2003 root301Tail3 winner =
    scan edge (search edge 3) 4 [2003] 2003 root301Tail4 winner := by
  rw [root301Tail3, scan, if_neg (by decide), root301Candidates3_checked]
  change scan edge (search edge 3) 4 [2003] 2003 root301Tail4
    (if 3 ≤ root301Candidates3.length then search edge 3 [2003, 3413] root301Candidates3 5416 winner else winner) = _
  rw [if_neg (by decide)]
def root301Candidates4 : List Nat := [5051]
set_option maxRecDepth 8192 in
theorem root301Candidates4_checked : candidates edge 4 2003 3557 root301Tail5 winner.bound = root301Candidates4 := by
  unfold candidates
  have row : edge 3557 = (SortedNatTree.contains FastRows.row3557) := by
    change Data.row3557.contains = (SortedNatTree.contains FastRows.row3557)
    exact FastRows.row3557_checked.symm
  rw [row]
  decide
theorem root301Step4 : scan edge (search edge 3) 4 [2003] 2003 root301Tail4 winner =
    scan edge (search edge 3) 4 [2003] 2003 root301Tail5 winner := by
  rw [root301Tail4, scan, if_neg (by decide), root301Candidates4_checked]
  change scan edge (search edge 3) 4 [2003] 2003 root301Tail5
    (if 3 ≤ root301Candidates4.length then search edge 3 [2003, 3557] root301Candidates4 5560 winner else winner) = _
  rw [if_neg (by decide)]
def root301Candidates5 : List Nat := [4673, 5693, 11777]
set_option maxRecDepth 8192 in
theorem root301Candidates5_checked : candidates edge 4 2003 4271 root301Tail6 winner.bound = root301Candidates5 := by
  unfold candidates
  have row : edge 4271 = (SortedNatTree.contains FastRows.row4271) := by
    change Data.row4271.contains = (SortedNatTree.contains FastRows.row4271)
    exact FastRows.row4271_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root301Child5 : search edge 3 [2003, 4271] root301Candidates5 6274 winner = winner := by decide
theorem root301Step5 : scan edge (search edge 3) 4 [2003] 2003 root301Tail5 winner =
    scan edge (search edge 3) 4 [2003] 2003 root301Tail6 winner := by
  rw [root301Tail5, scan, if_neg (by decide), root301Candidates5_checked]
  change scan edge (search edge 3) 4 [2003] 2003 root301Tail6
    (if 3 ≤ root301Candidates5.length then search edge 3 [2003, 4271] root301Candidates5 6274 winner else winner) = _
  rw [if_pos (by decide), root301Child5]
def root301Candidates6 : List Nat := [10601]
set_option maxRecDepth 8192 in
theorem root301Candidates6_checked : candidates edge 4 2003 4673 root301Tail7 winner.bound = root301Candidates6 := by
  unfold candidates
  have row : edge 4673 = (SortedNatTree.contains FastRows.row4673) := by
    change Data.row4673.contains = (SortedNatTree.contains FastRows.row4673)
    exact FastRows.row4673_checked.symm
  rw [row]
  decide
theorem root301Step6 : scan edge (search edge 3) 4 [2003] 2003 root301Tail6 winner =
    scan edge (search edge 3) 4 [2003] 2003 root301Tail7 winner := by
  rw [root301Tail6, scan, if_neg (by decide), root301Candidates6_checked]
  change scan edge (search edge 3) 4 [2003] 2003 root301Tail7
    (if 3 ≤ root301Candidates6.length then search edge 3 [2003, 4673] root301Candidates6 6676 winner else winner) = _
  rw [if_neg (by decide)]
def root301Candidates7 : List Nat := [5153]
set_option maxRecDepth 8192 in
theorem root301Candidates7_checked : candidates edge 4 2003 5039 root301Tail8 winner.bound = root301Candidates7 := by
  unfold candidates
  have row : edge 5039 = (SortedNatTree.contains FastRows.row5039) := by
    change Data.row5039.contains = (SortedNatTree.contains FastRows.row5039)
    exact FastRows.row5039_checked.symm
  rw [row]
  decide
theorem root301Step7 : scan edge (search edge 3) 4 [2003] 2003 root301Tail7 winner =
    scan edge (search edge 3) 4 [2003] 2003 root301Tail8 winner := by
  rw [root301Tail7, scan, if_neg (by decide), root301Candidates7_checked]
  change scan edge (search edge 3) 4 [2003] 2003 root301Tail8
    (if 3 ≤ root301Candidates7.length then search edge 3 [2003, 5039] root301Candidates7 7042 winner else winner) = _
  rw [if_neg (by decide)]
def root301Candidates8 : List Nat := [6833]
set_option maxRecDepth 8192 in
theorem root301Candidates8_checked : candidates edge 4 2003 5051 root301Tail9 winner.bound = root301Candidates8 := by
  unfold candidates
  have row : edge 5051 = (SortedNatTree.contains FastRows.row5051) := by
    change Data.row5051.contains = (SortedNatTree.contains FastRows.row5051)
    exact FastRows.row5051_checked.symm
  rw [row]
  decide
theorem root301Step8 : scan edge (search edge 3) 4 [2003] 2003 root301Tail8 winner =
    scan edge (search edge 3) 4 [2003] 2003 root301Tail9 winner := by
  rw [root301Tail8, scan, if_neg (by decide), root301Candidates8_checked]
  change scan edge (search edge 3) 4 [2003] 2003 root301Tail9
    (if 3 ≤ root301Candidates8.length then search edge 3 [2003, 5051] root301Candidates8 7054 winner else winner) = _
  rw [if_neg (by decide)]
def root301Candidates9 : List Nat := [8297, 15683]
set_option maxRecDepth 8192 in
theorem root301Candidates9_checked : candidates edge 4 2003 5153 root301Tail10 winner.bound = root301Candidates9 := by
  unfold candidates
  have row : edge 5153 = (SortedNatTree.contains FastRows.row5153) := by
    change Data.row5153.contains = (SortedNatTree.contains FastRows.row5153)
    exact FastRows.row5153_checked.symm
  rw [row]
  decide
theorem root301Step9 : scan edge (search edge 3) 4 [2003] 2003 root301Tail9 winner =
    scan edge (search edge 3) 4 [2003] 2003 root301Tail10 winner := by
  rw [root301Tail9, scan, if_neg (by decide), root301Candidates9_checked]
  change scan edge (search edge 3) 4 [2003] 2003 root301Tail10
    (if 3 ≤ root301Candidates9.length then search edge 3 [2003, 5153] root301Candidates9 7156 winner else winner) = _
  rw [if_neg (by decide)]
def root301Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root301Candidates10_checked : candidates edge 4 2003 5189 root301Tail11 winner.bound = root301Candidates10 := by
  unfold candidates
  have row : edge 5189 = (SortedNatTree.contains FastRows.row5189) := by
    change Data.row5189.contains = (SortedNatTree.contains FastRows.row5189)
    exact FastRows.row5189_checked.symm
  rw [row]
  decide
theorem root301Step10 : scan edge (search edge 3) 4 [2003] 2003 root301Tail10 winner =
    scan edge (search edge 3) 4 [2003] 2003 root301Tail11 winner := by
  rw [root301Tail10, scan, if_neg (by decide), root301Candidates10_checked]
  change scan edge (search edge 3) 4 [2003] 2003 root301Tail11
    (if 3 ≤ root301Candidates10.length then search edge 3 [2003, 5189] root301Candidates10 7192 winner else winner) = _
  rw [if_neg (by decide)]
def root301Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root301Candidates11_checked : candidates edge 4 2003 5651 root301Tail12 winner.bound = root301Candidates11 := by
  unfold candidates
  have row : edge 5651 = (SortedNatTree.contains FastRows.row5651) := by
    change Data.row5651.contains = (SortedNatTree.contains FastRows.row5651)
    exact FastRows.row5651_checked.symm
  rw [row]
  decide
theorem root301Step11 : scan edge (search edge 3) 4 [2003] 2003 root301Tail11 winner =
    scan edge (search edge 3) 4 [2003] 2003 root301Tail12 winner := by
  rw [root301Tail11, scan, if_neg (by decide), root301Candidates11_checked]
  change scan edge (search edge 3) 4 [2003] 2003 root301Tail12
    (if 3 ≤ root301Candidates11.length then search edge 3 [2003, 5651] root301Candidates11 7654 winner else winner) = _
  rw [if_neg (by decide)]
def root301Candidates12 : List Nat := [6791, 6983]
set_option maxRecDepth 8192 in
theorem root301Candidates12_checked : candidates edge 4 2003 5657 root301Tail13 winner.bound = root301Candidates12 := by
  unfold candidates
  have row : edge 5657 = (SortedNatTree.contains FastRows.row5657) := by
    change Data.row5657.contains = (SortedNatTree.contains FastRows.row5657)
    exact FastRows.row5657_checked.symm
  rw [row]
  decide
theorem root301Step12 : scan edge (search edge 3) 4 [2003] 2003 root301Tail12 winner =
    scan edge (search edge 3) 4 [2003] 2003 root301Tail13 winner := by
  rw [root301Tail12, scan, if_neg (by decide), root301Candidates12_checked]
  change scan edge (search edge 3) 4 [2003] 2003 root301Tail13
    (if 3 ≤ root301Candidates12.length then search edge 3 [2003, 5657] root301Candidates12 7660 winner else winner) = _
  rw [if_neg (by decide)]
def root301Candidates13 : List Nat := [6563, 8171]
set_option maxRecDepth 8192 in
theorem root301Candidates13_checked : candidates edge 4 2003 5693 root301Tail14 winner.bound = root301Candidates13 := by
  unfold candidates
  have row : edge 5693 = (SortedNatTree.contains FastRows.row5693) := by
    change Data.row5693.contains = (SortedNatTree.contains FastRows.row5693)
    exact FastRows.row5693_checked.symm
  rw [row]
  decide
theorem root301Step13 : scan edge (search edge 3) 4 [2003] 2003 root301Tail13 winner =
    scan edge (search edge 3) 4 [2003] 2003 root301Tail14 winner := by
  rw [root301Tail13, scan, if_neg (by decide), root301Candidates13_checked]
  change scan edge (search edge 3) 4 [2003] 2003 root301Tail14
    (if 3 ≤ root301Candidates13.length then search edge 3 [2003, 5693] root301Candidates13 7696 winner else winner) = _
  rw [if_neg (by decide)]
theorem root301Scan14 : scan edge (search edge 3) 4 [2003] 2003 root301Tail14 winner = winner := by rw [root301Tail14, scan, if_pos (by decide)]
theorem root301Scan13 : scan edge (search edge 3) 4 [2003] 2003 root301Tail13 winner = winner :=
  (root301Step13).trans root301Scan14
theorem root301Scan12 : scan edge (search edge 3) 4 [2003] 2003 root301Tail12 winner = winner :=
  (root301Step12).trans root301Scan13
theorem root301Scan11 : scan edge (search edge 3) 4 [2003] 2003 root301Tail11 winner = winner :=
  (root301Step11).trans root301Scan12
theorem root301Scan10 : scan edge (search edge 3) 4 [2003] 2003 root301Tail10 winner = winner :=
  (root301Step10).trans root301Scan11
theorem root301Scan9 : scan edge (search edge 3) 4 [2003] 2003 root301Tail9 winner = winner :=
  (root301Step9).trans root301Scan10
theorem root301Scan8 : scan edge (search edge 3) 4 [2003] 2003 root301Tail8 winner = winner :=
  (root301Step8).trans root301Scan9
theorem root301Scan7 : scan edge (search edge 3) 4 [2003] 2003 root301Tail7 winner = winner :=
  (root301Step7).trans root301Scan8
theorem root301Scan6 : scan edge (search edge 3) 4 [2003] 2003 root301Tail6 winner = winner :=
  (root301Step6).trans root301Scan7
theorem root301Scan5 : scan edge (search edge 3) 4 [2003] 2003 root301Tail5 winner = winner :=
  (root301Step5).trans root301Scan6
theorem root301Scan4 : scan edge (search edge 3) 4 [2003] 2003 root301Tail4 winner = winner :=
  (root301Step4).trans root301Scan5
theorem root301Scan3 : scan edge (search edge 3) 4 [2003] 2003 root301Tail3 winner = winner :=
  (root301Step3).trans root301Scan4
theorem root301Scan2 : scan edge (search edge 3) 4 [2003] 2003 root301Tail2 winner = winner :=
  (root301Step2).trans root301Scan3
theorem root301Scan1 : scan edge (search edge 3) 4 [2003] 2003 root301Tail1 winner = winner :=
  (root301Step1).trans root301Scan2
theorem root301Scan0 : scan edge (search edge 3) 4 [2003] 2003 root301Tail0 winner = winner :=
  (root301Step0).trans root301Scan1
set_option maxRecDepth 4096 in
theorem root301Branch : search edge 4 [2003] root301Candidates 2003 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2003] 2003 root301Tail0 winner = winner
  exact root301Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
