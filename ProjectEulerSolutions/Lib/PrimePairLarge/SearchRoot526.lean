import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root526Candidates : List Nat := Data.row3803
set_option maxRecDepth 8192 in
theorem root526Candidates_checked : candidates edge 5 0 3803 tail527 winner.bound = root526Candidates := by
  apply candidates_eq_row edge 5 0 3803 tail527 root526Candidates winner.bound tail527_ordered
  · decide
  · decide
  · funext q
    rfl
def root526Tail42 : List Nat := []
def root526Tail41 : List Nat := 21419 :: root526Tail42
def root526Tail40 : List Nat := 20747 :: root526Tail41
def root526Tail39 : List Nat := 20147 :: root526Tail40
def root526Tail38 : List Nat := 19991 :: root526Tail39
def root526Tail37 : List Nat := 18773 :: root526Tail38
def root526Tail36 : List Nat := 18503 :: root526Tail37
def root526Tail35 : List Nat := 18311 :: root526Tail36
def root526Tail34 : List Nat := 17939 :: root526Tail35
def root526Tail33 : List Nat := 17291 :: root526Tail34
def root526Tail32 : List Nat := 17159 :: root526Tail33
def root526Tail31 : List Nat := 16253 :: root526Tail32
def root526Tail30 : List Nat := 16127 :: root526Tail31
def root526Tail29 : List Nat := 16073 :: root526Tail30
def root526Tail28 : List Nat := 15443 :: root526Tail29
def root526Tail27 : List Nat := 15083 :: root526Tail28
def root526Tail26 : List Nat := 13931 :: root526Tail27
def root526Tail25 : List Nat := 13523 :: root526Tail26
def root526Tail24 : List Nat := 13421 :: root526Tail25
def root526Tail23 : List Nat := 13259 :: root526Tail24
def root526Tail22 : List Nat := 13151 :: root526Tail23
def root526Tail21 : List Nat := 13127 :: root526Tail22
def root526Tail20 : List Nat := 12671 :: root526Tail21
def root526Tail19 : List Nat := 12539 :: root526Tail20
def root526Tail18 : List Nat := 11801 :: root526Tail19
def root526Tail17 : List Nat := 11213 :: root526Tail18
def root526Tail16 : List Nat := 11171 :: root526Tail17
def root526Tail15 : List Nat := 10949 :: root526Tail16
def root526Tail14 : List Nat := 9941 :: root526Tail15
def root526Tail13 : List Nat := 9587 :: root526Tail14
def root526Tail12 : List Nat := 8999 :: root526Tail13
def root526Tail11 : List Nat := 8069 :: root526Tail12
def root526Tail10 : List Nat := 7253 :: root526Tail11
def root526Tail9 : List Nat := 7121 :: root526Tail10
def root526Tail8 : List Nat := 6521 :: root526Tail9
def root526Tail7 : List Nat := 5507 :: root526Tail8
def root526Tail6 : List Nat := 5087 :: root526Tail7
def root526Tail5 : List Nat := 5051 :: root526Tail6
def root526Tail4 : List Nat := 4943 :: root526Tail5
def root526Tail3 : List Nat := 4547 :: root526Tail4
def root526Tail2 : List Nat := 4289 :: root526Tail3
def root526Tail1 : List Nat := 4001 :: root526Tail2
def root526Tail0 : List Nat := 3911 :: root526Tail1
def root526Candidates0 : List Nat := [13127, 13523]
set_option maxRecDepth 8192 in
theorem root526Candidates0_checked : candidates edge 4 3803 3911 root526Tail1 winner.bound = root526Candidates0 := by
  unfold candidates
  have row : edge 3911 = (SortedNatTree.contains FastRows.row3911) := by
    change Data.row3911.contains = (SortedNatTree.contains FastRows.row3911)
    exact FastRows.row3911_checked.symm
  rw [row]
  decide
theorem root526Step0 : scan edge (search edge 3) 4 [3803] 3803 root526Tail0 winner =
    scan edge (search edge 3) 4 [3803] 3803 root526Tail1 winner := by
  rw [root526Tail0, scan, if_neg (by decide), root526Candidates0_checked]
  change scan edge (search edge 3) 4 [3803] 3803 root526Tail1
    (if 3 ≤ root526Candidates0.length then search edge 3 [3803, 3911] root526Candidates0 7714 winner else winner) = _
  rw [if_neg (by decide)]
def root526Candidates1 : List Nat := [7121]
set_option maxRecDepth 8192 in
theorem root526Candidates1_checked : candidates edge 4 3803 4001 root526Tail2 winner.bound = root526Candidates1 := by
  unfold candidates
  have row : edge 4001 = (SortedNatTree.contains FastRows.row4001) := by
    change Data.row4001.contains = (SortedNatTree.contains FastRows.row4001)
    exact FastRows.row4001_checked.symm
  rw [row]
  decide
theorem root526Step1 : scan edge (search edge 3) 4 [3803] 3803 root526Tail1 winner =
    scan edge (search edge 3) 4 [3803] 3803 root526Tail2 winner := by
  rw [root526Tail1, scan, if_neg (by decide), root526Candidates1_checked]
  change scan edge (search edge 3) 4 [3803] 3803 root526Tail2
    (if 3 ≤ root526Candidates1.length then search edge 3 [3803, 4001] root526Candidates1 7804 winner else winner) = _
  rw [if_neg (by decide)]
def root526Candidates2 : List Nat := [5507, 13151, 16073]
set_option maxRecDepth 8192 in
theorem root526Candidates2_checked : candidates edge 4 3803 4289 root526Tail3 winner.bound = root526Candidates2 := by
  unfold candidates
  have row : edge 4289 = (SortedNatTree.contains FastRows.row4289) := by
    change Data.row4289.contains = (SortedNatTree.contains FastRows.row4289)
    exact FastRows.row4289_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root526Child2 : search edge 3 [3803, 4289] root526Candidates2 8092 winner = winner := by decide
theorem root526Step2 : scan edge (search edge 3) 4 [3803] 3803 root526Tail2 winner =
    scan edge (search edge 3) 4 [3803] 3803 root526Tail3 winner := by
  rw [root526Tail2, scan, if_neg (by decide), root526Candidates2_checked]
  change scan edge (search edge 3) 4 [3803] 3803 root526Tail3
    (if 3 ≤ root526Candidates2.length then search edge 3 [3803, 4289] root526Candidates2 8092 winner else winner) = _
  rw [if_pos (by decide), root526Child2]
def root526Candidates3 : List Nat := [4943, 6521]
set_option maxRecDepth 8192 in
theorem root526Candidates3_checked : candidates edge 4 3803 4547 root526Tail4 winner.bound = root526Candidates3 := by
  unfold candidates
  have row : edge 4547 = (SortedNatTree.contains FastRows.row4547) := by
    change Data.row4547.contains = (SortedNatTree.contains FastRows.row4547)
    exact FastRows.row4547_checked.symm
  rw [row]
  decide
theorem root526Step3 : scan edge (search edge 3) 4 [3803] 3803 root526Tail3 winner =
    scan edge (search edge 3) 4 [3803] 3803 root526Tail4 winner := by
  rw [root526Tail3, scan, if_neg (by decide), root526Candidates3_checked]
  change scan edge (search edge 3) 4 [3803] 3803 root526Tail4
    (if 3 ≤ root526Candidates3.length then search edge 3 [3803, 4547] root526Candidates3 8350 winner else winner) = _
  rw [if_neg (by decide)]
def root526Candidates4 : List Nat := [11801]
set_option maxRecDepth 8192 in
theorem root526Candidates4_checked : candidates edge 4 3803 4943 root526Tail5 winner.bound = root526Candidates4 := by
  unfold candidates
  have row : edge 4943 = (SortedNatTree.contains FastRows.row4943) := by
    change Data.row4943.contains = (SortedNatTree.contains FastRows.row4943)
    exact FastRows.row4943_checked.symm
  rw [row]
  decide
theorem root526Step4 : scan edge (search edge 3) 4 [3803] 3803 root526Tail4 winner =
    scan edge (search edge 3) 4 [3803] 3803 root526Tail5 winner := by
  rw [root526Tail4, scan, if_neg (by decide), root526Candidates4_checked]
  change scan edge (search edge 3) 4 [3803] 3803 root526Tail5
    (if 3 ≤ root526Candidates4.length then search edge 3 [3803, 4943] root526Candidates4 8746 winner else winner) = _
  rw [if_neg (by decide)]
def root526Candidates5 : List Nat := [7253, 12539]
set_option maxRecDepth 8192 in
theorem root526Candidates5_checked : candidates edge 4 3803 5051 root526Tail6 winner.bound = root526Candidates5 := by
  unfold candidates
  have row : edge 5051 = (SortedNatTree.contains FastRows.row5051) := by
    change Data.row5051.contains = (SortedNatTree.contains FastRows.row5051)
    exact FastRows.row5051_checked.symm
  rw [row]
  decide
theorem root526Step5 : scan edge (search edge 3) 4 [3803] 3803 root526Tail5 winner =
    scan edge (search edge 3) 4 [3803] 3803 root526Tail6 winner := by
  rw [root526Tail5, scan, if_neg (by decide), root526Candidates5_checked]
  change scan edge (search edge 3) 4 [3803] 3803 root526Tail6
    (if 3 ≤ root526Candidates5.length then search edge 3 [3803, 5051] root526Candidates5 8854 winner else winner) = _
  rw [if_neg (by decide)]
def root526Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root526Candidates6_checked : candidates edge 4 3803 5087 root526Tail7 winner.bound = root526Candidates6 := by
  unfold candidates
  have row : edge 5087 = (SortedNatTree.contains FastRows.row5087) := by
    change Data.row5087.contains = (SortedNatTree.contains FastRows.row5087)
    exact FastRows.row5087_checked.symm
  rw [row]
  decide
theorem root526Step6 : scan edge (search edge 3) 4 [3803] 3803 root526Tail6 winner =
    scan edge (search edge 3) 4 [3803] 3803 root526Tail7 winner := by
  rw [root526Tail6, scan, if_neg (by decide), root526Candidates6_checked]
  change scan edge (search edge 3) 4 [3803] 3803 root526Tail7
    (if 3 ≤ root526Candidates6.length then search edge 3 [3803, 5087] root526Candidates6 8890 winner else winner) = _
  rw [if_neg (by decide)]
def root526Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root526Candidates7_checked : candidates edge 4 3803 5507 root526Tail8 winner.bound = root526Candidates7 := by
  unfold candidates
  have row : edge 5507 = (SortedNatTree.contains FastRows.row5507) := by
    change Data.row5507.contains = (SortedNatTree.contains FastRows.row5507)
    exact FastRows.row5507_checked.symm
  rw [row]
  decide
theorem root526Step7 : scan edge (search edge 3) 4 [3803] 3803 root526Tail7 winner =
    scan edge (search edge 3) 4 [3803] 3803 root526Tail8 winner := by
  rw [root526Tail7, scan, if_neg (by decide), root526Candidates7_checked]
  change scan edge (search edge 3) 4 [3803] 3803 root526Tail8
    (if 3 ≤ root526Candidates7.length then search edge 3 [3803, 5507] root526Candidates7 9310 winner else winner) = _
  rw [if_neg (by decide)]
theorem root526Scan8 : scan edge (search edge 3) 4 [3803] 3803 root526Tail8 winner = winner := by rw [root526Tail8, scan, if_pos (by decide)]
theorem root526Scan7 : scan edge (search edge 3) 4 [3803] 3803 root526Tail7 winner = winner :=
  (root526Step7).trans root526Scan8
theorem root526Scan6 : scan edge (search edge 3) 4 [3803] 3803 root526Tail6 winner = winner :=
  (root526Step6).trans root526Scan7
theorem root526Scan5 : scan edge (search edge 3) 4 [3803] 3803 root526Tail5 winner = winner :=
  (root526Step5).trans root526Scan6
theorem root526Scan4 : scan edge (search edge 3) 4 [3803] 3803 root526Tail4 winner = winner :=
  (root526Step4).trans root526Scan5
theorem root526Scan3 : scan edge (search edge 3) 4 [3803] 3803 root526Tail3 winner = winner :=
  (root526Step3).trans root526Scan4
theorem root526Scan2 : scan edge (search edge 3) 4 [3803] 3803 root526Tail2 winner = winner :=
  (root526Step2).trans root526Scan3
theorem root526Scan1 : scan edge (search edge 3) 4 [3803] 3803 root526Tail1 winner = winner :=
  (root526Step1).trans root526Scan2
theorem root526Scan0 : scan edge (search edge 3) 4 [3803] 3803 root526Tail0 winner = winner :=
  (root526Step0).trans root526Scan1
set_option maxRecDepth 4096 in
theorem root526Branch : search edge 4 [3803] root526Candidates 3803 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3803] 3803 root526Tail0 winner = winner
  exact root526Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
