import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root438Candidates : List Nat := Data.row3083
set_option maxRecDepth 8192 in
theorem root438Candidates_checked : candidates edge 5 0 3083 tail439 winner.bound = root438Candidates := by
  apply candidates_eq_row edge 5 0 3083 tail439 root438Candidates winner.bound tail439_ordered
  · decide
  · decide
  · funext q
    rfl
def root438Tail35 : List Nat := []
def root438Tail34 : List Nat := 22691 :: root438Tail35
def root438Tail33 : List Nat := 22031 :: root438Tail34
def root438Tail32 : List Nat := 21803 :: root438Tail33
def root438Tail31 : List Nat := 21347 :: root438Tail32
def root438Tail30 : List Nat := 21149 :: root438Tail31
def root438Tail29 : List Nat := 20759 :: root438Tail30
def root438Tail28 : List Nat := 18797 :: root438Tail29
def root438Tail27 : List Nat := 18251 :: root438Tail28
def root438Tail26 : List Nat := 17939 :: root438Tail27
def root438Tail25 : List Nat := 17903 :: root438Tail26
def root438Tail24 : List Nat := 17021 :: root438Tail25
def root438Tail23 : List Nat := 15467 :: root438Tail24
def root438Tail22 : List Nat := 15053 :: root438Tail23
def root438Tail21 : List Nat := 14411 :: root438Tail22
def root438Tail20 : List Nat := 12197 :: root438Tail21
def root438Tail19 : List Nat := 11699 :: root438Tail20
def root438Tail18 : List Nat := 11519 :: root438Tail19
def root438Tail17 : List Nat := 10133 :: root438Tail18
def root438Tail16 : List Nat := 9929 :: root438Tail17
def root438Tail15 : List Nat := 9461 :: root438Tail16
def root438Tail14 : List Nat := 8963 :: root438Tail15
def root438Tail13 : List Nat := 8747 :: root438Tail14
def root438Tail12 : List Nat := 7907 :: root438Tail13
def root438Tail11 : List Nat := 7307 :: root438Tail12
def root438Tail10 : List Nat := 7187 :: root438Tail11
def root438Tail9 : List Nat := 6737 :: root438Tail10
def root438Tail8 : List Nat := 6257 :: root438Tail9
def root438Tail7 : List Nat := 5333 :: root438Tail8
def root438Tail6 : List Nat := 4931 :: root438Tail7
def root438Tail5 : List Nat := 4871 :: root438Tail6
def root438Tail4 : List Nat := 4703 :: root438Tail5
def root438Tail3 : List Nat := 4373 :: root438Tail4
def root438Tail2 : List Nat := 3797 :: root438Tail3
def root438Tail1 : List Nat := 3449 :: root438Tail2
def root438Tail0 : List Nat := 3329 :: root438Tail1
def root438Candidates0 : List Nat := [11519]
set_option maxRecDepth 8192 in
theorem root438Candidates0_checked : candidates edge 4 3083 3329 root438Tail1 winner.bound = root438Candidates0 := by
  unfold candidates
  have row : edge 3329 = (SortedNatTree.contains FastRows.row3329) := by
    change Data.row3329.contains = (SortedNatTree.contains FastRows.row3329)
    exact FastRows.row3329_checked.symm
  rw [row]
  decide
theorem root438Step0 : scan edge (search edge 3) 4 [3083] 3083 root438Tail0 winner =
    scan edge (search edge 3) 4 [3083] 3083 root438Tail1 winner := by
  rw [root438Tail0, scan, if_neg (by decide), root438Candidates0_checked]
  change scan edge (search edge 3) 4 [3083] 3083 root438Tail1
    (if 3 ≤ root438Candidates0.length then search edge 3 [3083, 3329] root438Candidates0 6412 winner else winner) = _
  rw [if_neg (by decide)]
def root438Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root438Candidates1_checked : candidates edge 4 3083 3449 root438Tail2 winner.bound = root438Candidates1 := by
  unfold candidates
  have row : edge 3449 = (SortedNatTree.contains FastRows.row3449) := by
    change Data.row3449.contains = (SortedNatTree.contains FastRows.row3449)
    exact FastRows.row3449_checked.symm
  rw [row]
  decide
theorem root438Step1 : scan edge (search edge 3) 4 [3083] 3083 root438Tail1 winner =
    scan edge (search edge 3) 4 [3083] 3083 root438Tail2 winner := by
  rw [root438Tail1, scan, if_neg (by decide), root438Candidates1_checked]
  change scan edge (search edge 3) 4 [3083] 3083 root438Tail2
    (if 3 ≤ root438Candidates1.length then search edge 3 [3083, 3449] root438Candidates1 6532 winner else winner) = _
  rw [if_neg (by decide)]
def root438Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root438Candidates2_checked : candidates edge 4 3083 3797 root438Tail3 winner.bound = root438Candidates2 := by
  unfold candidates
  have row : edge 3797 = (SortedNatTree.contains FastRows.row3797) := by
    change Data.row3797.contains = (SortedNatTree.contains FastRows.row3797)
    exact FastRows.row3797_checked.symm
  rw [row]
  decide
theorem root438Step2 : scan edge (search edge 3) 4 [3083] 3083 root438Tail2 winner =
    scan edge (search edge 3) 4 [3083] 3083 root438Tail3 winner := by
  rw [root438Tail2, scan, if_neg (by decide), root438Candidates2_checked]
  change scan edge (search edge 3) 4 [3083] 3083 root438Tail3
    (if 3 ≤ root438Candidates2.length then search edge 3 [3083, 3797] root438Candidates2 6880 winner else winner) = _
  rw [if_neg (by decide)]
def root438Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root438Candidates3_checked : candidates edge 4 3083 4373 root438Tail4 winner.bound = root438Candidates3 := by
  unfold candidates
  have row : edge 4373 = (SortedNatTree.contains FastRows.row4373) := by
    change Data.row4373.contains = (SortedNatTree.contains FastRows.row4373)
    exact FastRows.row4373_checked.symm
  rw [row]
  decide
theorem root438Step3 : scan edge (search edge 3) 4 [3083] 3083 root438Tail3 winner =
    scan edge (search edge 3) 4 [3083] 3083 root438Tail4 winner := by
  rw [root438Tail3, scan, if_neg (by decide), root438Candidates3_checked]
  change scan edge (search edge 3) 4 [3083] 3083 root438Tail4
    (if 3 ≤ root438Candidates3.length then search edge 3 [3083, 4373] root438Candidates3 7456 winner else winner) = _
  rw [if_neg (by decide)]
def root438Candidates4 : List Nat := [6257]
set_option maxRecDepth 8192 in
theorem root438Candidates4_checked : candidates edge 4 3083 4703 root438Tail5 winner.bound = root438Candidates4 := by
  unfold candidates
  have row : edge 4703 = (SortedNatTree.contains FastRows.row4703) := by
    change Data.row4703.contains = (SortedNatTree.contains FastRows.row4703)
    exact FastRows.row4703_checked.symm
  rw [row]
  decide
theorem root438Step4 : scan edge (search edge 3) 4 [3083] 3083 root438Tail4 winner =
    scan edge (search edge 3) 4 [3083] 3083 root438Tail5 winner := by
  rw [root438Tail4, scan, if_neg (by decide), root438Candidates4_checked]
  change scan edge (search edge 3) 4 [3083] 3083 root438Tail5
    (if 3 ≤ root438Candidates4.length then search edge 3 [3083, 4703] root438Candidates4 7786 winner else winner) = _
  rw [if_neg (by decide)]
def root438Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root438Candidates5_checked : candidates edge 4 3083 4871 root438Tail6 winner.bound = root438Candidates5 := by
  unfold candidates
  have row : edge 4871 = (SortedNatTree.contains FastRows.row4871) := by
    change Data.row4871.contains = (SortedNatTree.contains FastRows.row4871)
    exact FastRows.row4871_checked.symm
  rw [row]
  decide
theorem root438Step5 : scan edge (search edge 3) 4 [3083] 3083 root438Tail5 winner =
    scan edge (search edge 3) 4 [3083] 3083 root438Tail6 winner := by
  rw [root438Tail5, scan, if_neg (by decide), root438Candidates5_checked]
  change scan edge (search edge 3) 4 [3083] 3083 root438Tail6
    (if 3 ≤ root438Candidates5.length then search edge 3 [3083, 4871] root438Candidates5 7954 winner else winner) = _
  rw [if_neg (by decide)]
def root438Candidates6 : List Nat := [17903]
set_option maxRecDepth 8192 in
theorem root438Candidates6_checked : candidates edge 4 3083 4931 root438Tail7 winner.bound = root438Candidates6 := by
  unfold candidates
  have row : edge 4931 = (SortedNatTree.contains FastRows.row4931) := by
    change Data.row4931.contains = (SortedNatTree.contains FastRows.row4931)
    exact FastRows.row4931_checked.symm
  rw [row]
  decide
theorem root438Step6 : scan edge (search edge 3) 4 [3083] 3083 root438Tail6 winner =
    scan edge (search edge 3) 4 [3083] 3083 root438Tail7 winner := by
  rw [root438Tail6, scan, if_neg (by decide), root438Candidates6_checked]
  change scan edge (search edge 3) 4 [3083] 3083 root438Tail7
    (if 3 ≤ root438Candidates6.length then search edge 3 [3083, 4931] root438Candidates6 8014 winner else winner) = _
  rw [if_neg (by decide)]
def root438Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root438Candidates7_checked : candidates edge 4 3083 5333 root438Tail8 winner.bound = root438Candidates7 := by
  unfold candidates
  have row : edge 5333 = (SortedNatTree.contains FastRows.row5333) := by
    change Data.row5333.contains = (SortedNatTree.contains FastRows.row5333)
    exact FastRows.row5333_checked.symm
  rw [row]
  decide
theorem root438Step7 : scan edge (search edge 3) 4 [3083] 3083 root438Tail7 winner =
    scan edge (search edge 3) 4 [3083] 3083 root438Tail8 winner := by
  rw [root438Tail7, scan, if_neg (by decide), root438Candidates7_checked]
  change scan edge (search edge 3) 4 [3083] 3083 root438Tail8
    (if 3 ≤ root438Candidates7.length then search edge 3 [3083, 5333] root438Candidates7 8416 winner else winner) = _
  rw [if_neg (by decide)]
theorem root438Scan8 : scan edge (search edge 3) 4 [3083] 3083 root438Tail8 winner = winner := by rw [root438Tail8, scan, if_pos (by decide)]
theorem root438Scan7 : scan edge (search edge 3) 4 [3083] 3083 root438Tail7 winner = winner :=
  (root438Step7).trans root438Scan8
theorem root438Scan6 : scan edge (search edge 3) 4 [3083] 3083 root438Tail6 winner = winner :=
  (root438Step6).trans root438Scan7
theorem root438Scan5 : scan edge (search edge 3) 4 [3083] 3083 root438Tail5 winner = winner :=
  (root438Step5).trans root438Scan6
theorem root438Scan4 : scan edge (search edge 3) 4 [3083] 3083 root438Tail4 winner = winner :=
  (root438Step4).trans root438Scan5
theorem root438Scan3 : scan edge (search edge 3) 4 [3083] 3083 root438Tail3 winner = winner :=
  (root438Step3).trans root438Scan4
theorem root438Scan2 : scan edge (search edge 3) 4 [3083] 3083 root438Tail2 winner = winner :=
  (root438Step2).trans root438Scan3
theorem root438Scan1 : scan edge (search edge 3) 4 [3083] 3083 root438Tail1 winner = winner :=
  (root438Step1).trans root438Scan2
theorem root438Scan0 : scan edge (search edge 3) 4 [3083] 3083 root438Tail0 winner = winner :=
  (root438Step0).trans root438Scan1
set_option maxRecDepth 4096 in
theorem root438Branch : search edge 4 [3083] root438Candidates 3083 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3083] 3083 root438Tail0 winner = winner
  exact root438Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
