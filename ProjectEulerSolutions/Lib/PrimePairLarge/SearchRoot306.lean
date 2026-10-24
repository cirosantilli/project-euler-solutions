import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root306Candidates : List Nat := Data.row2039
set_option maxRecDepth 8192 in
theorem root306Candidates_checked : candidates edge 5 0 2039 tail307 winner.bound = root306Candidates := by
  apply candidates_eq_row edge 5 0 2039 tail307 root306Candidates winner.bound tail307_ordered
  · decide
  · decide
  · funext q
    rfl
def root306Tail40 : List Nat := []
def root306Tail39 : List Nat := 22409 :: root306Tail40
def root306Tail38 : List Nat := 22091 :: root306Tail39
def root306Tail37 : List Nat := 22031 :: root306Tail38
def root306Tail36 : List Nat := 22013 :: root306Tail37
def root306Tail35 : List Nat := 21587 :: root306Tail36
def root306Tail34 : List Nat := 20393 :: root306Tail35
def root306Tail33 : List Nat := 20021 :: root306Tail34
def root306Tail32 : List Nat := 19181 :: root306Tail33
def root306Tail31 : List Nat := 18593 :: root306Tail32
def root306Tail30 : List Nat := 18503 :: root306Tail31
def root306Tail29 : List Nat := 18269 :: root306Tail30
def root306Tail28 : List Nat := 16829 :: root306Tail29
def root306Tail27 : List Nat := 16553 :: root306Tail28
def root306Tail26 : List Nat := 16007 :: root306Tail27
def root306Tail25 : List Nat := 15749 :: root306Tail26
def root306Tail24 : List Nat := 15461 :: root306Tail25
def root306Tail23 : List Nat := 15329 :: root306Tail24
def root306Tail22 : List Nat := 14891 :: root306Tail23
def root306Tail21 : List Nat := 12941 :: root306Tail22
def root306Tail20 : List Nat := 11069 :: root306Tail21
def root306Tail19 : List Nat := 10463 :: root306Tail20
def root306Tail18 : List Nat := 10331 :: root306Tail19
def root306Tail17 : List Nat := 10253 :: root306Tail18
def root306Tail16 : List Nat := 10079 :: root306Tail17
def root306Tail15 : List Nat := 7673 :: root306Tail16
def root306Tail14 : List Nat := 6689 :: root306Tail15
def root306Tail13 : List Nat := 6203 :: root306Tail14
def root306Tail12 : List Nat := 6053 :: root306Tail13
def root306Tail11 : List Nat := 4937 :: root306Tail12
def root306Tail10 : List Nat := 4799 :: root306Tail11
def root306Tail9 : List Nat := 4523 :: root306Tail10
def root306Tail8 : List Nat := 4079 :: root306Tail9
def root306Tail7 : List Nat := 3917 :: root306Tail8
def root306Tail6 : List Nat := 3767 :: root306Tail7
def root306Tail5 : List Nat := 3491 :: root306Tail6
def root306Tail4 : List Nat := 3323 :: root306Tail5
def root306Tail3 : List Nat := 3299 :: root306Tail4
def root306Tail2 : List Nat := 2591 :: root306Tail3
def root306Tail1 : List Nat := 2417 :: root306Tail2
def root306Tail0 : List Nat := 2357 :: root306Tail1
def root306Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root306Candidates0_checked : candidates edge 4 2039 2357 root306Tail1 winner.bound = root306Candidates0 := by
  unfold candidates
  have row : edge 2357 = (SortedNatTree.contains FastRows.row2357) := by
    change Data.row2357.contains = (SortedNatTree.contains FastRows.row2357)
    exact FastRows.row2357_checked.symm
  rw [row]
  decide
theorem root306Step0 : scan edge (search edge 3) 4 [2039] 2039 root306Tail0 winner =
    scan edge (search edge 3) 4 [2039] 2039 root306Tail1 winner := by
  rw [root306Tail0, scan, if_neg (by decide), root306Candidates0_checked]
  change scan edge (search edge 3) 4 [2039] 2039 root306Tail1
    (if 3 ≤ root306Candidates0.length then search edge 3 [2039, 2357] root306Candidates0 4396 winner else winner) = _
  rw [if_neg (by decide)]
def root306Candidates1 : List Nat := [10463]
set_option maxRecDepth 8192 in
theorem root306Candidates1_checked : candidates edge 4 2039 2417 root306Tail2 winner.bound = root306Candidates1 := by
  unfold candidates
  have row : edge 2417 = (SortedNatTree.contains FastRows.row2417) := by
    change Data.row2417.contains = (SortedNatTree.contains FastRows.row2417)
    exact FastRows.row2417_checked.symm
  rw [row]
  decide
theorem root306Step1 : scan edge (search edge 3) 4 [2039] 2039 root306Tail1 winner =
    scan edge (search edge 3) 4 [2039] 2039 root306Tail2 winner := by
  rw [root306Tail1, scan, if_neg (by decide), root306Candidates1_checked]
  change scan edge (search edge 3) 4 [2039] 2039 root306Tail2
    (if 3 ≤ root306Candidates1.length then search edge 3 [2039, 2417] root306Candidates1 4456 winner else winner) = _
  rw [if_neg (by decide)]
def root306Candidates2 : List Nat := [4937, 16829]
set_option maxRecDepth 8192 in
theorem root306Candidates2_checked : candidates edge 4 2039 2591 root306Tail3 winner.bound = root306Candidates2 := by
  unfold candidates
  have row : edge 2591 = (SortedNatTree.contains FastRows.row2591) := by
    change Data.row2591.contains = (SortedNatTree.contains FastRows.row2591)
    exact FastRows.row2591_checked.symm
  rw [row]
  decide
theorem root306Step2 : scan edge (search edge 3) 4 [2039] 2039 root306Tail2 winner =
    scan edge (search edge 3) 4 [2039] 2039 root306Tail3 winner := by
  rw [root306Tail2, scan, if_neg (by decide), root306Candidates2_checked]
  change scan edge (search edge 3) 4 [2039] 2039 root306Tail3
    (if 3 ≤ root306Candidates2.length then search edge 3 [2039, 2591] root306Candidates2 4630 winner else winner) = _
  rw [if_neg (by decide)]
def root306Candidates3 : List Nat := [3767, 4523, 6203]
set_option maxRecDepth 8192 in
theorem root306Candidates3_checked : candidates edge 4 2039 3299 root306Tail4 winner.bound = root306Candidates3 := by
  unfold candidates
  have row : edge 3299 = (SortedNatTree.contains FastRows.row3299) := by
    change Data.row3299.contains = (SortedNatTree.contains FastRows.row3299)
    exact FastRows.row3299_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root306Child3 : search edge 3 [2039, 3299] root306Candidates3 5338 winner = winner := by decide
theorem root306Step3 : scan edge (search edge 3) 4 [2039] 2039 root306Tail3 winner =
    scan edge (search edge 3) 4 [2039] 2039 root306Tail4 winner := by
  rw [root306Tail3, scan, if_neg (by decide), root306Candidates3_checked]
  change scan edge (search edge 3) 4 [2039] 2039 root306Tail4
    (if 3 ≤ root306Candidates3.length then search edge 3 [2039, 3299] root306Candidates3 5338 winner else winner) = _
  rw [if_pos (by decide), root306Child3]
def root306Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root306Candidates4_checked : candidates edge 4 2039 3323 root306Tail5 winner.bound = root306Candidates4 := by
  unfold candidates
  have row : edge 3323 = (SortedNatTree.contains FastRows.row3323) := by
    change Data.row3323.contains = (SortedNatTree.contains FastRows.row3323)
    exact FastRows.row3323_checked.symm
  rw [row]
  decide
theorem root306Step4 : scan edge (search edge 3) 4 [2039] 2039 root306Tail4 winner =
    scan edge (search edge 3) 4 [2039] 2039 root306Tail5 winner := by
  rw [root306Tail4, scan, if_neg (by decide), root306Candidates4_checked]
  change scan edge (search edge 3) 4 [2039] 2039 root306Tail5
    (if 3 ≤ root306Candidates4.length then search edge 3 [2039, 3323] root306Candidates4 5362 winner else winner) = _
  rw [if_neg (by decide)]
def root306Candidates5 : List Nat := [11069, 18593]
set_option maxRecDepth 8192 in
theorem root306Candidates5_checked : candidates edge 4 2039 3491 root306Tail6 winner.bound = root306Candidates5 := by
  unfold candidates
  have row : edge 3491 = (SortedNatTree.contains FastRows.row3491) := by
    change Data.row3491.contains = (SortedNatTree.contains FastRows.row3491)
    exact FastRows.row3491_checked.symm
  rw [row]
  decide
theorem root306Step5 : scan edge (search edge 3) 4 [2039] 2039 root306Tail5 winner =
    scan edge (search edge 3) 4 [2039] 2039 root306Tail6 winner := by
  rw [root306Tail5, scan, if_neg (by decide), root306Candidates5_checked]
  change scan edge (search edge 3) 4 [2039] 2039 root306Tail6
    (if 3 ≤ root306Candidates5.length then search edge 3 [2039, 3491] root306Candidates5 5530 winner else winner) = _
  rw [if_neg (by decide)]
def root306Candidates6 : List Nat := [4523, 4937]
set_option maxRecDepth 8192 in
theorem root306Candidates6_checked : candidates edge 4 2039 3767 root306Tail7 winner.bound = root306Candidates6 := by
  unfold candidates
  have row : edge 3767 = (SortedNatTree.contains FastRows.row3767) := by
    change Data.row3767.contains = (SortedNatTree.contains FastRows.row3767)
    exact FastRows.row3767_checked.symm
  rw [row]
  decide
theorem root306Step6 : scan edge (search edge 3) 4 [2039] 2039 root306Tail6 winner =
    scan edge (search edge 3) 4 [2039] 2039 root306Tail7 winner := by
  rw [root306Tail6, scan, if_neg (by decide), root306Candidates6_checked]
  change scan edge (search edge 3) 4 [2039] 2039 root306Tail7
    (if 3 ≤ root306Candidates6.length then search edge 3 [2039, 3767] root306Candidates6 5806 winner else winner) = _
  rw [if_neg (by decide)]
def root306Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root306Candidates7_checked : candidates edge 4 2039 3917 root306Tail8 winner.bound = root306Candidates7 := by
  unfold candidates
  have row : edge 3917 = (SortedNatTree.contains FastRows.row3917) := by
    change Data.row3917.contains = (SortedNatTree.contains FastRows.row3917)
    exact FastRows.row3917_checked.symm
  rw [row]
  decide
theorem root306Step7 : scan edge (search edge 3) 4 [2039] 2039 root306Tail7 winner =
    scan edge (search edge 3) 4 [2039] 2039 root306Tail8 winner := by
  rw [root306Tail7, scan, if_neg (by decide), root306Candidates7_checked]
  change scan edge (search edge 3) 4 [2039] 2039 root306Tail8
    (if 3 ≤ root306Candidates7.length then search edge 3 [2039, 3917] root306Candidates7 5956 winner else winner) = _
  rw [if_neg (by decide)]
def root306Candidates8 : List Nat := [6053, 11069]
set_option maxRecDepth 8192 in
theorem root306Candidates8_checked : candidates edge 4 2039 4079 root306Tail9 winner.bound = root306Candidates8 := by
  unfold candidates
  have row : edge 4079 = (SortedNatTree.contains FastRows.row4079) := by
    change Data.row4079.contains = (SortedNatTree.contains FastRows.row4079)
    exact FastRows.row4079_checked.symm
  rw [row]
  decide
theorem root306Step8 : scan edge (search edge 3) 4 [2039] 2039 root306Tail8 winner =
    scan edge (search edge 3) 4 [2039] 2039 root306Tail9 winner := by
  rw [root306Tail8, scan, if_neg (by decide), root306Candidates8_checked]
  change scan edge (search edge 3) 4 [2039] 2039 root306Tail9
    (if 3 ≤ root306Candidates8.length then search edge 3 [2039, 4079] root306Candidates8 6118 winner else winner) = _
  rw [if_neg (by decide)]
def root306Candidates9 : List Nat := [18593]
set_option maxRecDepth 8192 in
theorem root306Candidates9_checked : candidates edge 4 2039 4523 root306Tail10 winner.bound = root306Candidates9 := by
  unfold candidates
  have row : edge 4523 = (SortedNatTree.contains FastRows.row4523) := by
    change Data.row4523.contains = (SortedNatTree.contains FastRows.row4523)
    exact FastRows.row4523_checked.symm
  rw [row]
  decide
theorem root306Step9 : scan edge (search edge 3) 4 [2039] 2039 root306Tail9 winner =
    scan edge (search edge 3) 4 [2039] 2039 root306Tail10 winner := by
  rw [root306Tail9, scan, if_neg (by decide), root306Candidates9_checked]
  change scan edge (search edge 3) 4 [2039] 2039 root306Tail10
    (if 3 ≤ root306Candidates9.length then search edge 3 [2039, 4523] root306Candidates9 6562 winner else winner) = _
  rw [if_neg (by decide)]
def root306Candidates10 : List Nat := [10253]
set_option maxRecDepth 8192 in
theorem root306Candidates10_checked : candidates edge 4 2039 4799 root306Tail11 winner.bound = root306Candidates10 := by
  unfold candidates
  have row : edge 4799 = (SortedNatTree.contains FastRows.row4799) := by
    change Data.row4799.contains = (SortedNatTree.contains FastRows.row4799)
    exact FastRows.row4799_checked.symm
  rw [row]
  decide
theorem root306Step10 : scan edge (search edge 3) 4 [2039] 2039 root306Tail10 winner =
    scan edge (search edge 3) 4 [2039] 2039 root306Tail11 winner := by
  rw [root306Tail10, scan, if_neg (by decide), root306Candidates10_checked]
  change scan edge (search edge 3) 4 [2039] 2039 root306Tail11
    (if 3 ≤ root306Candidates10.length then search edge 3 [2039, 4799] root306Candidates10 6838 winner else winner) = _
  rw [if_neg (by decide)]
def root306Candidates11 : List Nat := [6053]
set_option maxRecDepth 8192 in
theorem root306Candidates11_checked : candidates edge 4 2039 4937 root306Tail12 winner.bound = root306Candidates11 := by
  unfold candidates
  have row : edge 4937 = (SortedNatTree.contains FastRows.row4937) := by
    change Data.row4937.contains = (SortedNatTree.contains FastRows.row4937)
    exact FastRows.row4937_checked.symm
  rw [row]
  decide
theorem root306Step11 : scan edge (search edge 3) 4 [2039] 2039 root306Tail11 winner =
    scan edge (search edge 3) 4 [2039] 2039 root306Tail12 winner := by
  rw [root306Tail11, scan, if_neg (by decide), root306Candidates11_checked]
  change scan edge (search edge 3) 4 [2039] 2039 root306Tail12
    (if 3 ≤ root306Candidates11.length then search edge 3 [2039, 4937] root306Candidates11 6976 winner else winner) = _
  rw [if_neg (by decide)]
theorem root306Scan12 : scan edge (search edge 3) 4 [2039] 2039 root306Tail12 winner = winner := by rw [root306Tail12, scan, if_pos (by decide)]
theorem root306Scan11 : scan edge (search edge 3) 4 [2039] 2039 root306Tail11 winner = winner :=
  (root306Step11).trans root306Scan12
theorem root306Scan10 : scan edge (search edge 3) 4 [2039] 2039 root306Tail10 winner = winner :=
  (root306Step10).trans root306Scan11
theorem root306Scan9 : scan edge (search edge 3) 4 [2039] 2039 root306Tail9 winner = winner :=
  (root306Step9).trans root306Scan10
theorem root306Scan8 : scan edge (search edge 3) 4 [2039] 2039 root306Tail8 winner = winner :=
  (root306Step8).trans root306Scan9
theorem root306Scan7 : scan edge (search edge 3) 4 [2039] 2039 root306Tail7 winner = winner :=
  (root306Step7).trans root306Scan8
theorem root306Scan6 : scan edge (search edge 3) 4 [2039] 2039 root306Tail6 winner = winner :=
  (root306Step6).trans root306Scan7
theorem root306Scan5 : scan edge (search edge 3) 4 [2039] 2039 root306Tail5 winner = winner :=
  (root306Step5).trans root306Scan6
theorem root306Scan4 : scan edge (search edge 3) 4 [2039] 2039 root306Tail4 winner = winner :=
  (root306Step4).trans root306Scan5
theorem root306Scan3 : scan edge (search edge 3) 4 [2039] 2039 root306Tail3 winner = winner :=
  (root306Step3).trans root306Scan4
theorem root306Scan2 : scan edge (search edge 3) 4 [2039] 2039 root306Tail2 winner = winner :=
  (root306Step2).trans root306Scan3
theorem root306Scan1 : scan edge (search edge 3) 4 [2039] 2039 root306Tail1 winner = winner :=
  (root306Step1).trans root306Scan2
theorem root306Scan0 : scan edge (search edge 3) 4 [2039] 2039 root306Tail0 winner = winner :=
  (root306Step0).trans root306Scan1
set_option maxRecDepth 4096 in
theorem root306Branch : search edge 4 [2039] root306Candidates 2039 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2039] 2039 root306Tail0 winner = winner
  exact root306Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
