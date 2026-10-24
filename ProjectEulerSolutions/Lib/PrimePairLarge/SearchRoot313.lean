import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root313Candidates : List Nat := Data.row2089
set_option maxRecDepth 8192 in
theorem root313Candidates_checked : candidates edge 5 0 2089 tail314 winner.bound = root313Candidates := by
  apply candidates_eq_row edge 5 0 2089 tail314 root313Candidates winner.bound tail314_ordered
  · decide
  · decide
  · funext q
    rfl
def root313Tail40 : List Nat := []
def root313Tail39 : List Nat := 23041 :: root313Tail40
def root313Tail38 : List Nat := 22453 :: root313Tail39
def root313Tail37 : List Nat := 21997 :: root313Tail38
def root313Tail36 : List Nat := 20533 :: root313Tail37
def root313Tail35 : List Nat := 19507 :: root313Tail36
def root313Tail34 : List Nat := 18181 :: root313Tail35
def root313Tail33 : List Nat := 18121 :: root313Tail34
def root313Tail32 : List Nat := 17041 :: root313Tail33
def root313Tail31 : List Nat := 16693 :: root313Tail32
def root313Tail30 : List Nat := 15859 :: root313Tail31
def root313Tail29 : List Nat := 15823 :: root313Tail30
def root313Tail28 : List Nat := 15277 :: root313Tail29
def root313Tail27 : List Nat := 13921 :: root313Tail28
def root313Tail26 : List Nat := 13633 :: root313Tail27
def root313Tail25 : List Nat := 13249 :: root313Tail26
def root313Tail24 : List Nat := 12757 :: root313Tail25
def root313Tail23 : List Nat := 12037 :: root313Tail24
def root313Tail22 : List Nat := 10687 :: root313Tail23
def root313Tail21 : List Nat := 10369 :: root313Tail22
def root313Tail20 : List Nat := 9679 :: root313Tail21
def root313Tail19 : List Nat := 9091 :: root313Tail20
def root313Tail18 : List Nat := 9049 :: root313Tail19
def root313Tail17 : List Nat := 8263 :: root313Tail18
def root313Tail16 : List Nat := 8209 :: root313Tail17
def root313Tail15 : List Nat := 7873 :: root313Tail16
def root313Tail14 : List Nat := 7411 :: root313Tail15
def root313Tail13 : List Nat := 7297 :: root313Tail14
def root313Tail12 : List Nat := 7237 :: root313Tail13
def root313Tail11 : List Nat := 7039 :: root313Tail12
def root313Tail10 : List Nat := 6277 :: root313Tail11
def root313Tail9 : List Nat := 5827 :: root313Tail10
def root313Tail8 : List Nat := 5443 :: root313Tail9
def root313Tail7 : List Nat := 4597 :: root313Tail8
def root313Tail6 : List Nat := 4219 :: root313Tail7
def root313Tail5 : List Nat := 4099 :: root313Tail6
def root313Tail4 : List Nat := 3931 :: root313Tail5
def root313Tail3 : List Nat := 3643 :: root313Tail4
def root313Tail2 : List Nat := 3181 :: root313Tail3
def root313Tail1 : List Nat := 2953 :: root313Tail2
def root313Tail0 : List Nat := 2341 :: root313Tail1
def root313Candidates0 : List Nat := [9679]
set_option maxRecDepth 8192 in
theorem root313Candidates0_checked : candidates edge 4 2089 2341 root313Tail1 winner.bound = root313Candidates0 := by
  unfold candidates
  have row : edge 2341 = (SortedNatTree.contains FastRows.row2341) := by
    change Data.row2341.contains = (SortedNatTree.contains FastRows.row2341)
    exact FastRows.row2341_checked.symm
  rw [row]
  decide
theorem root313Step0 : scan edge (search edge 3) 4 [2089] 2089 root313Tail0 winner =
    scan edge (search edge 3) 4 [2089] 2089 root313Tail1 winner := by
  rw [root313Tail0, scan, if_neg (by decide), root313Candidates0_checked]
  change scan edge (search edge 3) 4 [2089] 2089 root313Tail1
    (if 3 ≤ root313Candidates0.length then search edge 3 [2089, 2341] root313Candidates0 4430 winner else winner) = _
  rw [if_neg (by decide)]
def root313Candidates1 : List Nat := [3181, 3643, 13633]
set_option maxRecDepth 8192 in
theorem root313Candidates1_checked : candidates edge 4 2089 2953 root313Tail2 winner.bound = root313Candidates1 := by
  unfold candidates
  have row : edge 2953 = (SortedNatTree.contains FastRows.row2953) := by
    change Data.row2953.contains = (SortedNatTree.contains FastRows.row2953)
    exact FastRows.row2953_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root313Child1 : search edge 3 [2089, 2953] root313Candidates1 5042 winner = winner := by decide
theorem root313Step1 : scan edge (search edge 3) 4 [2089] 2089 root313Tail1 winner =
    scan edge (search edge 3) 4 [2089] 2089 root313Tail2 winner := by
  rw [root313Tail1, scan, if_neg (by decide), root313Candidates1_checked]
  change scan edge (search edge 3) 4 [2089] 2089 root313Tail2
    (if 3 ≤ root313Candidates1.length then search edge 3 [2089, 2953] root313Candidates1 5042 winner else winner) = _
  rw [if_pos (by decide), root313Child1]
def root313Candidates2 : List Nat := [4099, 4219]
set_option maxRecDepth 8192 in
theorem root313Candidates2_checked : candidates edge 4 2089 3181 root313Tail3 winner.bound = root313Candidates2 := by
  unfold candidates
  have row : edge 3181 = (SortedNatTree.contains FastRows.row3181) := by
    change Data.row3181.contains = (SortedNatTree.contains FastRows.row3181)
    exact FastRows.row3181_checked.symm
  rw [row]
  decide
theorem root313Step2 : scan edge (search edge 3) 4 [2089] 2089 root313Tail2 winner =
    scan edge (search edge 3) 4 [2089] 2089 root313Tail3 winner := by
  rw [root313Tail2, scan, if_neg (by decide), root313Candidates2_checked]
  change scan edge (search edge 3) 4 [2089] 2089 root313Tail3
    (if 3 ≤ root313Candidates2.length then search edge 3 [2089, 3181] root313Candidates2 5270 winner else winner) = _
  rw [if_neg (by decide)]
def root313Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root313Candidates3_checked : candidates edge 4 2089 3643 root313Tail4 winner.bound = root313Candidates3 := by
  unfold candidates
  have row : edge 3643 = (SortedNatTree.contains FastRows.row3643) := by
    change Data.row3643.contains = (SortedNatTree.contains FastRows.row3643)
    exact FastRows.row3643_checked.symm
  rw [row]
  decide
theorem root313Step3 : scan edge (search edge 3) 4 [2089] 2089 root313Tail3 winner =
    scan edge (search edge 3) 4 [2089] 2089 root313Tail4 winner := by
  rw [root313Tail3, scan, if_neg (by decide), root313Candidates3_checked]
  change scan edge (search edge 3) 4 [2089] 2089 root313Tail4
    (if 3 ≤ root313Candidates3.length then search edge 3 [2089, 3643] root313Candidates3 5732 winner else winner) = _
  rw [if_neg (by decide)]
def root313Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root313Candidates4_checked : candidates edge 4 2089 3931 root313Tail5 winner.bound = root313Candidates4 := by
  unfold candidates
  have row : edge 3931 = (SortedNatTree.contains FastRows.row3931) := by
    change Data.row3931.contains = (SortedNatTree.contains FastRows.row3931)
    exact FastRows.row3931_checked.symm
  rw [row]
  decide
theorem root313Step4 : scan edge (search edge 3) 4 [2089] 2089 root313Tail4 winner =
    scan edge (search edge 3) 4 [2089] 2089 root313Tail5 winner := by
  rw [root313Tail4, scan, if_neg (by decide), root313Candidates4_checked]
  change scan edge (search edge 3) 4 [2089] 2089 root313Tail5
    (if 3 ≤ root313Candidates4.length then search edge 3 [2089, 3931] root313Candidates4 6020 winner else winner) = _
  rw [if_neg (by decide)]
def root313Candidates5 : List Nat := [12757, 16693]
set_option maxRecDepth 8192 in
theorem root313Candidates5_checked : candidates edge 4 2089 4099 root313Tail6 winner.bound = root313Candidates5 := by
  unfold candidates
  have row : edge 4099 = (SortedNatTree.contains FastRows.row4099) := by
    change Data.row4099.contains = (SortedNatTree.contains FastRows.row4099)
    exact FastRows.row4099_checked.symm
  rw [row]
  decide
theorem root313Step5 : scan edge (search edge 3) 4 [2089] 2089 root313Tail5 winner =
    scan edge (search edge 3) 4 [2089] 2089 root313Tail6 winner := by
  rw [root313Tail5, scan, if_neg (by decide), root313Candidates5_checked]
  change scan edge (search edge 3) 4 [2089] 2089 root313Tail6
    (if 3 ≤ root313Candidates5.length then search edge 3 [2089, 4099] root313Candidates5 6188 winner else winner) = _
  rw [if_neg (by decide)]
def root313Candidates6 : List Nat := [4597, 8209, 9679]
set_option maxRecDepth 8192 in
theorem root313Candidates6_checked : candidates edge 4 2089 4219 root313Tail7 winner.bound = root313Candidates6 := by
  unfold candidates
  have row : edge 4219 = (SortedNatTree.contains FastRows.row4219) := by
    change Data.row4219.contains = (SortedNatTree.contains FastRows.row4219)
    exact FastRows.row4219_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root313Child6 : search edge 3 [2089, 4219] root313Candidates6 6308 winner = winner := by decide
theorem root313Step6 : scan edge (search edge 3) 4 [2089] 2089 root313Tail6 winner =
    scan edge (search edge 3) 4 [2089] 2089 root313Tail7 winner := by
  rw [root313Tail6, scan, if_neg (by decide), root313Candidates6_checked]
  change scan edge (search edge 3) 4 [2089] 2089 root313Tail7
    (if 3 ≤ root313Candidates6.length then search edge 3 [2089, 4219] root313Candidates6 6308 winner else winner) = _
  rw [if_pos (by decide), root313Child6]
def root313Candidates7 : List Nat := [8263]
set_option maxRecDepth 8192 in
theorem root313Candidates7_checked : candidates edge 4 2089 4597 root313Tail8 winner.bound = root313Candidates7 := by
  unfold candidates
  have row : edge 4597 = (SortedNatTree.contains FastRows.row4597) := by
    change Data.row4597.contains = (SortedNatTree.contains FastRows.row4597)
    exact FastRows.row4597_checked.symm
  rw [row]
  decide
theorem root313Step7 : scan edge (search edge 3) 4 [2089] 2089 root313Tail7 winner =
    scan edge (search edge 3) 4 [2089] 2089 root313Tail8 winner := by
  rw [root313Tail7, scan, if_neg (by decide), root313Candidates7_checked]
  change scan edge (search edge 3) 4 [2089] 2089 root313Tail8
    (if 3 ≤ root313Candidates7.length then search edge 3 [2089, 4597] root313Candidates7 6686 winner else winner) = _
  rw [if_neg (by decide)]
def root313Candidates8 : List Nat := [7039, 10687, 12037, 15859]
set_option maxRecDepth 8192 in
theorem root313Candidates8_checked : candidates edge 4 2089 5443 root313Tail9 winner.bound = root313Candidates8 := by
  unfold candidates
  have row : edge 5443 = (SortedNatTree.contains FastRows.row5443) := by
    change Data.row5443.contains = (SortedNatTree.contains FastRows.row5443)
    exact FastRows.row5443_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root313Child8 : search edge 3 [2089, 5443] root313Candidates8 7532 winner = winner := by decide
theorem root313Step8 : scan edge (search edge 3) 4 [2089] 2089 root313Tail8 winner =
    scan edge (search edge 3) 4 [2089] 2089 root313Tail9 winner := by
  rw [root313Tail8, scan, if_neg (by decide), root313Candidates8_checked]
  change scan edge (search edge 3) 4 [2089] 2089 root313Tail9
    (if 3 ≤ root313Candidates8.length then search edge 3 [2089, 5443] root313Candidates8 7532 winner else winner) = _
  rw [if_pos (by decide), root313Child8]
def root313Candidates9 : List Nat := [9679]
set_option maxRecDepth 8192 in
theorem root313Candidates9_checked : candidates edge 4 2089 5827 root313Tail10 winner.bound = root313Candidates9 := by
  unfold candidates
  have row : edge 5827 = (SortedNatTree.contains FastRows.row5827) := by
    change Data.row5827.contains = (SortedNatTree.contains FastRows.row5827)
    exact FastRows.row5827_checked.symm
  rw [row]
  decide
theorem root313Step9 : scan edge (search edge 3) 4 [2089] 2089 root313Tail9 winner =
    scan edge (search edge 3) 4 [2089] 2089 root313Tail10 winner := by
  rw [root313Tail9, scan, if_neg (by decide), root313Candidates9_checked]
  change scan edge (search edge 3) 4 [2089] 2089 root313Tail10
    (if 3 ≤ root313Candidates9.length then search edge 3 [2089, 5827] root313Candidates9 7916 winner else winner) = _
  rw [if_neg (by decide)]
theorem root313Scan10 : scan edge (search edge 3) 4 [2089] 2089 root313Tail10 winner = winner := by rw [root313Tail10, scan, if_pos (by decide)]
theorem root313Scan9 : scan edge (search edge 3) 4 [2089] 2089 root313Tail9 winner = winner :=
  (root313Step9).trans root313Scan10
theorem root313Scan8 : scan edge (search edge 3) 4 [2089] 2089 root313Tail8 winner = winner :=
  (root313Step8).trans root313Scan9
theorem root313Scan7 : scan edge (search edge 3) 4 [2089] 2089 root313Tail7 winner = winner :=
  (root313Step7).trans root313Scan8
theorem root313Scan6 : scan edge (search edge 3) 4 [2089] 2089 root313Tail6 winner = winner :=
  (root313Step6).trans root313Scan7
theorem root313Scan5 : scan edge (search edge 3) 4 [2089] 2089 root313Tail5 winner = winner :=
  (root313Step5).trans root313Scan6
theorem root313Scan4 : scan edge (search edge 3) 4 [2089] 2089 root313Tail4 winner = winner :=
  (root313Step4).trans root313Scan5
theorem root313Scan3 : scan edge (search edge 3) 4 [2089] 2089 root313Tail3 winner = winner :=
  (root313Step3).trans root313Scan4
theorem root313Scan2 : scan edge (search edge 3) 4 [2089] 2089 root313Tail2 winner = winner :=
  (root313Step2).trans root313Scan3
theorem root313Scan1 : scan edge (search edge 3) 4 [2089] 2089 root313Tail1 winner = winner :=
  (root313Step1).trans root313Scan2
theorem root313Scan0 : scan edge (search edge 3) 4 [2089] 2089 root313Tail0 winner = winner :=
  (root313Step0).trans root313Scan1
set_option maxRecDepth 4096 in
theorem root313Branch : search edge 4 [2089] root313Candidates 2089 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2089] 2089 root313Tail0 winner = winner
  exact root313Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
