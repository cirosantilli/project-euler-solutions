import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root355Candidates : List Nat := Data.row2411
set_option maxRecDepth 8192 in
theorem root355Candidates_checked : candidates edge 5 0 2411 tail356 winner.bound = root355Candidates := by
  apply candidates_eq_row edge 5 0 2411 tail356 root355Candidates winner.bound tail356_ordered
  · decide
  · decide
  · funext q
    rfl
def root355Tail36 : List Nat := []
def root355Tail35 : List Nat := 21821 :: root355Tail36
def root355Tail34 : List Nat := 20921 :: root355Tail35
def root355Tail33 : List Nat := 20543 :: root355Tail34
def root355Tail32 : List Nat := 19661 :: root355Tail33
def root355Tail31 : List Nat := 19553 :: root355Tail32
def root355Tail30 : List Nat := 18869 :: root355Tail31
def root355Tail29 : List Nat := 18749 :: root355Tail30
def root355Tail28 : List Nat := 18287 :: root355Tail29
def root355Tail27 : List Nat := 17099 :: root355Tail28
def root355Tail26 : List Nat := 16691 :: root355Tail27
def root355Tail25 : List Nat := 16217 :: root355Tail26
def root355Tail24 : List Nat := 15269 :: root355Tail25
def root355Tail23 : List Nat := 13043 :: root355Tail24
def root355Tail22 : List Nat := 12899 :: root355Tail23
def root355Tail21 : List Nat := 12107 :: root355Tail22
def root355Tail20 : List Nat := 11777 :: root355Tail21
def root355Tail19 : List Nat := 11069 :: root355Tail20
def root355Tail18 : List Nat := 10253 :: root355Tail19
def root355Tail17 : List Nat := 9803 :: root355Tail18
def root355Tail16 : List Nat := 9533 :: root355Tail17
def root355Tail15 : List Nat := 9497 :: root355Tail16
def root355Tail14 : List Nat := 9281 :: root355Tail15
def root355Tail13 : List Nat := 9257 :: root355Tail14
def root355Tail12 : List Nat := 8447 :: root355Tail13
def root355Tail11 : List Nat := 8273 :: root355Tail12
def root355Tail10 : List Nat := 7481 :: root355Tail11
def root355Tail9 : List Nat := 7433 :: root355Tail10
def root355Tail8 : List Nat := 7019 :: root355Tail9
def root355Tail7 : List Nat := 5879 :: root355Tail8
def root355Tail6 : List Nat := 5519 :: root355Tail7
def root355Tail5 : List Nat := 5171 :: root355Tail6
def root355Tail4 : List Nat := 4919 :: root355Tail5
def root355Tail3 : List Nat := 4793 :: root355Tail4
def root355Tail2 : List Nat := 4253 :: root355Tail3
def root355Tail1 : List Nat := 4241 :: root355Tail2
def root355Tail0 : List Nat := 3917 :: root355Tail1
def root355Candidates0 : List Nat := [12107, 18869]
set_option maxRecDepth 8192 in
theorem root355Candidates0_checked : candidates edge 4 2411 3917 root355Tail1 winner.bound = root355Candidates0 := by
  unfold candidates
  have row : edge 3917 = (SortedNatTree.contains FastRows.row3917) := by
    change Data.row3917.contains = (SortedNatTree.contains FastRows.row3917)
    exact FastRows.row3917_checked.symm
  rw [row]
  decide
theorem root355Step0 : scan edge (search edge 3) 4 [2411] 2411 root355Tail0 winner =
    scan edge (search edge 3) 4 [2411] 2411 root355Tail1 winner := by
  rw [root355Tail0, scan, if_neg (by decide), root355Candidates0_checked]
  change scan edge (search edge 3) 4 [2411] 2411 root355Tail1
    (if 3 ≤ root355Candidates0.length then search edge 3 [2411, 3917] root355Candidates0 6328 winner else winner) = _
  rw [if_neg (by decide)]
def root355Candidates1 : List Nat := [4793]
set_option maxRecDepth 8192 in
theorem root355Candidates1_checked : candidates edge 4 2411 4241 root355Tail2 winner.bound = root355Candidates1 := by
  unfold candidates
  have row : edge 4241 = (SortedNatTree.contains FastRows.row4241) := by
    change Data.row4241.contains = (SortedNatTree.contains FastRows.row4241)
    exact FastRows.row4241_checked.symm
  rw [row]
  decide
theorem root355Step1 : scan edge (search edge 3) 4 [2411] 2411 root355Tail1 winner =
    scan edge (search edge 3) 4 [2411] 2411 root355Tail2 winner := by
  rw [root355Tail1, scan, if_neg (by decide), root355Candidates1_checked]
  change scan edge (search edge 3) 4 [2411] 2411 root355Tail2
    (if 3 ≤ root355Candidates1.length then search edge 3 [2411, 4241] root355Candidates1 6652 winner else winner) = _
  rw [if_neg (by decide)]
def root355Candidates2 : List Nat := [17099]
set_option maxRecDepth 8192 in
theorem root355Candidates2_checked : candidates edge 4 2411 4253 root355Tail3 winner.bound = root355Candidates2 := by
  unfold candidates
  have row : edge 4253 = (SortedNatTree.contains FastRows.row4253) := by
    change Data.row4253.contains = (SortedNatTree.contains FastRows.row4253)
    exact FastRows.row4253_checked.symm
  rw [row]
  decide
theorem root355Step2 : scan edge (search edge 3) 4 [2411] 2411 root355Tail2 winner =
    scan edge (search edge 3) 4 [2411] 2411 root355Tail3 winner := by
  rw [root355Tail2, scan, if_neg (by decide), root355Candidates2_checked]
  change scan edge (search edge 3) 4 [2411] 2411 root355Tail3
    (if 3 ≤ root355Candidates2.length then search edge 3 [2411, 4253] root355Candidates2 6664 winner else winner) = _
  rw [if_neg (by decide)]
def root355Candidates3 : List Nat := [7481]
set_option maxRecDepth 8192 in
theorem root355Candidates3_checked : candidates edge 4 2411 4793 root355Tail4 winner.bound = root355Candidates3 := by
  unfold candidates
  have row : edge 4793 = (SortedNatTree.contains FastRows.row4793) := by
    change Data.row4793.contains = (SortedNatTree.contains FastRows.row4793)
    exact FastRows.row4793_checked.symm
  rw [row]
  decide
theorem root355Step3 : scan edge (search edge 3) 4 [2411] 2411 root355Tail3 winner =
    scan edge (search edge 3) 4 [2411] 2411 root355Tail4 winner := by
  rw [root355Tail3, scan, if_neg (by decide), root355Candidates3_checked]
  change scan edge (search edge 3) 4 [2411] 2411 root355Tail4
    (if 3 ≤ root355Candidates3.length then search edge 3 [2411, 4793] root355Candidates3 7204 winner else winner) = _
  rw [if_neg (by decide)]
def root355Candidates4 : List Nat := [7481]
set_option maxRecDepth 8192 in
theorem root355Candidates4_checked : candidates edge 4 2411 4919 root355Tail5 winner.bound = root355Candidates4 := by
  unfold candidates
  have row : edge 4919 = (SortedNatTree.contains FastRows.row4919) := by
    change Data.row4919.contains = (SortedNatTree.contains FastRows.row4919)
    exact FastRows.row4919_checked.symm
  rw [row]
  decide
theorem root355Step4 : scan edge (search edge 3) 4 [2411] 2411 root355Tail4 winner =
    scan edge (search edge 3) 4 [2411] 2411 root355Tail5 winner := by
  rw [root355Tail4, scan, if_neg (by decide), root355Candidates4_checked]
  change scan edge (search edge 3) 4 [2411] 2411 root355Tail5
    (if 3 ≤ root355Candidates4.length then search edge 3 [2411, 4919] root355Candidates4 7330 winner else winner) = _
  rw [if_neg (by decide)]
def root355Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root355Candidates5_checked : candidates edge 4 2411 5171 root355Tail6 winner.bound = root355Candidates5 := by
  unfold candidates
  have row : edge 5171 = (SortedNatTree.contains FastRows.row5171) := by
    change Data.row5171.contains = (SortedNatTree.contains FastRows.row5171)
    exact FastRows.row5171_checked.symm
  rw [row]
  decide
theorem root355Step5 : scan edge (search edge 3) 4 [2411] 2411 root355Tail5 winner =
    scan edge (search edge 3) 4 [2411] 2411 root355Tail6 winner := by
  rw [root355Tail5, scan, if_neg (by decide), root355Candidates5_checked]
  change scan edge (search edge 3) 4 [2411] 2411 root355Tail6
    (if 3 ≤ root355Candidates5.length then search edge 3 [2411, 5171] root355Candidates5 7582 winner else winner) = _
  rw [if_neg (by decide)]
def root355Candidates6 : List Nat := [9533, 16691]
set_option maxRecDepth 8192 in
theorem root355Candidates6_checked : candidates edge 4 2411 5519 root355Tail7 winner.bound = root355Candidates6 := by
  unfold candidates
  have row : edge 5519 = (SortedNatTree.contains FastRows.row5519) := by
    change Data.row5519.contains = (SortedNatTree.contains FastRows.row5519)
    exact FastRows.row5519_checked.symm
  rw [row]
  decide
theorem root355Step6 : scan edge (search edge 3) 4 [2411] 2411 root355Tail6 winner =
    scan edge (search edge 3) 4 [2411] 2411 root355Tail7 winner := by
  rw [root355Tail6, scan, if_neg (by decide), root355Candidates6_checked]
  change scan edge (search edge 3) 4 [2411] 2411 root355Tail7
    (if 3 ≤ root355Candidates6.length then search edge 3 [2411, 5519] root355Candidates6 7930 winner else winner) = _
  rw [if_neg (by decide)]
def root355Candidates7 : List Nat := [8273]
set_option maxRecDepth 8192 in
theorem root355Candidates7_checked : candidates edge 4 2411 5879 root355Tail8 winner.bound = root355Candidates7 := by
  unfold candidates
  have row : edge 5879 = (SortedNatTree.contains FastRows.row5879) := by
    change Data.row5879.contains = (SortedNatTree.contains FastRows.row5879)
    exact FastRows.row5879_checked.symm
  rw [row]
  decide
theorem root355Step7 : scan edge (search edge 3) 4 [2411] 2411 root355Tail7 winner =
    scan edge (search edge 3) 4 [2411] 2411 root355Tail8 winner := by
  rw [root355Tail7, scan, if_neg (by decide), root355Candidates7_checked]
  change scan edge (search edge 3) 4 [2411] 2411 root355Tail8
    (if 3 ≤ root355Candidates7.length then search edge 3 [2411, 5879] root355Candidates7 8290 winner else winner) = _
  rw [if_neg (by decide)]
theorem root355Scan8 : scan edge (search edge 3) 4 [2411] 2411 root355Tail8 winner = winner := by rw [root355Tail8, scan, if_pos (by decide)]
theorem root355Scan7 : scan edge (search edge 3) 4 [2411] 2411 root355Tail7 winner = winner :=
  (root355Step7).trans root355Scan8
theorem root355Scan6 : scan edge (search edge 3) 4 [2411] 2411 root355Tail6 winner = winner :=
  (root355Step6).trans root355Scan7
theorem root355Scan5 : scan edge (search edge 3) 4 [2411] 2411 root355Tail5 winner = winner :=
  (root355Step5).trans root355Scan6
theorem root355Scan4 : scan edge (search edge 3) 4 [2411] 2411 root355Tail4 winner = winner :=
  (root355Step4).trans root355Scan5
theorem root355Scan3 : scan edge (search edge 3) 4 [2411] 2411 root355Tail3 winner = winner :=
  (root355Step3).trans root355Scan4
theorem root355Scan2 : scan edge (search edge 3) 4 [2411] 2411 root355Tail2 winner = winner :=
  (root355Step2).trans root355Scan3
theorem root355Scan1 : scan edge (search edge 3) 4 [2411] 2411 root355Tail1 winner = winner :=
  (root355Step1).trans root355Scan2
theorem root355Scan0 : scan edge (search edge 3) 4 [2411] 2411 root355Tail0 winner = winner :=
  (root355Step0).trans root355Scan1
set_option maxRecDepth 4096 in
theorem root355Branch : search edge 4 [2411] root355Candidates 2411 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2411] 2411 root355Tail0 winner = winner
  exact root355Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
