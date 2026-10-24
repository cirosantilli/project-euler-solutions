import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root158Candidates : List Nat := Data.row947
set_option maxRecDepth 8192 in
theorem root158Candidates_checked : candidates edge 5 0 947 tail159 winner.bound = root158Candidates := by
  apply candidates_eq_row edge 5 0 947 tail159 root158Candidates winner.bound tail159_ordered
  · decide
  · decide
  · funext q
    rfl
def root158Tail61 : List Nat := []
def root158Tail60 : List Nat := 24683 :: root158Tail61
def root158Tail59 : List Nat := 24083 :: root158Tail60
def root158Tail58 : List Nat := 23747 :: root158Tail59
def root158Tail57 : List Nat := 23633 :: root158Tail58
def root158Tail56 : List Nat := 22973 :: root158Tail57
def root158Tail55 : List Nat := 22739 :: root158Tail56
def root158Tail54 : List Nat := 22721 :: root158Tail55
def root158Tail53 : List Nat := 22637 :: root158Tail54
def root158Tail52 : List Nat := 21563 :: root158Tail53
def root158Tail51 : List Nat := 21503 :: root158Tail52
def root158Tail50 : List Nat := 21269 :: root158Tail51
def root158Tail49 : List Nat := 20849 :: root158Tail50
def root158Tail48 : List Nat := 20807 :: root158Tail49
def root158Tail47 : List Nat := 20543 :: root158Tail48
def root158Tail46 : List Nat := 20261 :: root158Tail47
def root158Tail45 : List Nat := 20117 :: root158Tail46
def root158Tail44 : List Nat := 19853 :: root158Tail45
def root158Tail43 : List Nat := 18233 :: root158Tail44
def root158Tail42 : List Nat := 17981 :: root158Tail43
def root158Tail41 : List Nat := 17783 :: root158Tail42
def root158Tail40 : List Nat := 17729 :: root158Tail41
def root158Tail39 : List Nat := 17477 :: root158Tail40
def root158Tail38 : List Nat := 17291 :: root158Tail39
def root158Tail37 : List Nat := 16931 :: root158Tail38
def root158Tail36 : List Nat := 16901 :: root158Tail37
def root158Tail35 : List Nat := 16187 :: root158Tail36
def root158Tail34 : List Nat := 15641 :: root158Tail35
def root158Tail33 : List Nat := 14591 :: root158Tail34
def root158Tail32 : List Nat := 14549 :: root158Tail33
def root158Tail31 : List Nat := 14159 :: root158Tail32
def root158Tail30 : List Nat := 13901 :: root158Tail31
def root158Tail29 : List Nat := 13763 :: root158Tail30
def root158Tail28 : List Nat := 13691 :: root158Tail29
def root158Tail27 : List Nat := 12953 :: root158Tail28
def root158Tail26 : List Nat := 12923 :: root158Tail27
def root158Tail25 : List Nat := 12377 :: root158Tail26
def root158Tail24 : List Nat := 12197 :: root158Tail25
def root158Tail23 : List Nat := 10559 :: root158Tail24
def root158Tail22 : List Nat := 10337 :: root158Tail23
def root158Tail21 : List Nat := 9419 :: root158Tail22
def root158Tail20 : List Nat := 9377 :: root158Tail21
def root158Tail19 : List Nat := 9203 :: root158Tail20
def root158Tail18 : List Nat := 8627 :: root158Tail19
def root158Tail17 : List Nat := 8597 :: root158Tail18
def root158Tail16 : List Nat := 8387 :: root158Tail17
def root158Tail15 : List Nat := 8111 :: root158Tail16
def root158Tail14 : List Nat := 6947 :: root158Tail15
def root158Tail13 : List Nat := 5981 :: root158Tail14
def root158Tail12 : List Nat := 5051 :: root158Tail13
def root158Tail11 : List Nat := 5009 :: root158Tail12
def root158Tail10 : List Nat := 4931 :: root158Tail11
def root158Tail9 : List Nat := 4637 :: root158Tail10
def root158Tail8 : List Nat := 4523 :: root158Tail9
def root158Tail7 : List Nat := 3623 :: root158Tail8
def root158Tail6 : List Nat := 3491 :: root158Tail7
def root158Tail5 : List Nat := 3041 :: root158Tail6
def root158Tail4 : List Nat := 2657 :: root158Tail5
def root158Tail3 : List Nat := 1949 :: root158Tail4
def root158Tail2 : List Nat := 1901 :: root158Tail3
def root158Tail1 : List Nat := 1439 :: root158Tail2
def root158Tail0 : List Nat := 1097 :: root158Tail1
def root158Candidates0 : List Nat := [4637, 12377]
set_option maxRecDepth 8192 in
theorem root158Candidates0_checked : candidates edge 4 947 1097 root158Tail1 winner.bound = root158Candidates0 := by
  unfold candidates
  have row : edge 1097 = (SortedNatTree.contains FastRows.row1097) := by
    change Data.row1097.contains = (SortedNatTree.contains FastRows.row1097)
    exact FastRows.row1097_checked.symm
  rw [row]
  decide
theorem root158Step0 : scan edge (search edge 3) 4 [947] 947 root158Tail0 winner =
    scan edge (search edge 3) 4 [947] 947 root158Tail1 winner := by
  rw [root158Tail0, scan, if_neg (by decide), root158Candidates0_checked]
  change scan edge (search edge 3) 4 [947] 947 root158Tail1
    (if 3 ≤ root158Candidates0.length then search edge 3 [947, 1097] root158Candidates0 2044 winner else winner) = _
  rw [if_neg (by decide)]
def root158Candidates1 : List Nat := [4931, 20807]
set_option maxRecDepth 8192 in
theorem root158Candidates1_checked : candidates edge 4 947 1439 root158Tail2 winner.bound = root158Candidates1 := by
  unfold candidates
  have row : edge 1439 = (SortedNatTree.contains FastRows.row1439) := by
    change Data.row1439.contains = (SortedNatTree.contains FastRows.row1439)
    exact FastRows.row1439_checked.symm
  rw [row]
  decide
theorem root158Step1 : scan edge (search edge 3) 4 [947] 947 root158Tail1 winner =
    scan edge (search edge 3) 4 [947] 947 root158Tail2 winner := by
  rw [root158Tail1, scan, if_neg (by decide), root158Candidates1_checked]
  change scan edge (search edge 3) 4 [947] 947 root158Tail2
    (if 3 ≤ root158Candidates1.length then search edge 3 [947, 1439] root158Candidates1 2386 winner else winner) = _
  rw [if_neg (by decide)]
def root158Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root158Candidates2_checked : candidates edge 4 947 1901 root158Tail3 winner.bound = root158Candidates2 := by
  unfold candidates
  have row : edge 1901 = (SortedNatTree.contains FastRows.row1901) := by
    change Data.row1901.contains = (SortedNatTree.contains FastRows.row1901)
    exact FastRows.row1901_checked.symm
  rw [row]
  decide
theorem root158Step2 : scan edge (search edge 3) 4 [947] 947 root158Tail2 winner =
    scan edge (search edge 3) 4 [947] 947 root158Tail3 winner := by
  rw [root158Tail2, scan, if_neg (by decide), root158Candidates2_checked]
  change scan edge (search edge 3) 4 [947] 947 root158Tail3
    (if 3 ≤ root158Candidates2.length then search edge 3 [947, 1901] root158Candidates2 2848 winner else winner) = _
  rw [if_neg (by decide)]
def root158Candidates3 : List Nat := [2657, 18233, 20261]
set_option maxRecDepth 8192 in
theorem root158Candidates3_checked : candidates edge 4 947 1949 root158Tail4 winner.bound = root158Candidates3 := by
  unfold candidates
  have row : edge 1949 = (SortedNatTree.contains FastRows.row1949) := by
    change Data.row1949.contains = (SortedNatTree.contains FastRows.row1949)
    exact FastRows.row1949_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root158Child3 : search edge 3 [947, 1949] root158Candidates3 2896 winner = winner := by decide
theorem root158Step3 : scan edge (search edge 3) 4 [947] 947 root158Tail3 winner =
    scan edge (search edge 3) 4 [947] 947 root158Tail4 winner := by
  rw [root158Tail3, scan, if_neg (by decide), root158Candidates3_checked]
  change scan edge (search edge 3) 4 [947] 947 root158Tail4
    (if 3 ≤ root158Candidates3.length then search edge 3 [947, 1949] root158Candidates3 2896 winner else winner) = _
  rw [if_pos (by decide), root158Child3]
def root158Candidates4 : List Nat := [17477]
set_option maxRecDepth 8192 in
theorem root158Candidates4_checked : candidates edge 4 947 2657 root158Tail5 winner.bound = root158Candidates4 := by
  unfold candidates
  have row : edge 2657 = (SortedNatTree.contains FastRows.row2657) := by
    change Data.row2657.contains = (SortedNatTree.contains FastRows.row2657)
    exact FastRows.row2657_checked.symm
  rw [row]
  decide
theorem root158Step4 : scan edge (search edge 3) 4 [947] 947 root158Tail4 winner =
    scan edge (search edge 3) 4 [947] 947 root158Tail5 winner := by
  rw [root158Tail4, scan, if_neg (by decide), root158Candidates4_checked]
  change scan edge (search edge 3) 4 [947] 947 root158Tail5
    (if 3 ≤ root158Candidates4.length then search edge 3 [947, 2657] root158Candidates4 3604 winner else winner) = _
  rw [if_neg (by decide)]
def root158Candidates5 : List Nat := [3491, 17729]
set_option maxRecDepth 8192 in
theorem root158Candidates5_checked : candidates edge 4 947 3041 root158Tail6 winner.bound = root158Candidates5 := by
  unfold candidates
  have row : edge 3041 = (SortedNatTree.contains FastRows.row3041) := by
    change Data.row3041.contains = (SortedNatTree.contains FastRows.row3041)
    exact FastRows.row3041_checked.symm
  rw [row]
  decide
theorem root158Step5 : scan edge (search edge 3) 4 [947] 947 root158Tail5 winner =
    scan edge (search edge 3) 4 [947] 947 root158Tail6 winner := by
  rw [root158Tail5, scan, if_neg (by decide), root158Candidates5_checked]
  change scan edge (search edge 3) 4 [947] 947 root158Tail6
    (if 3 ≤ root158Candidates5.length then search edge 3 [947, 3041] root158Candidates5 3988 winner else winner) = _
  rw [if_neg (by decide)]
def root158Candidates6 : List Nat := [4931, 8111]
set_option maxRecDepth 8192 in
theorem root158Candidates6_checked : candidates edge 4 947 3491 root158Tail7 winner.bound = root158Candidates6 := by
  unfold candidates
  have row : edge 3491 = (SortedNatTree.contains FastRows.row3491) := by
    change Data.row3491.contains = (SortedNatTree.contains FastRows.row3491)
    exact FastRows.row3491_checked.symm
  rw [row]
  decide
theorem root158Step6 : scan edge (search edge 3) 4 [947] 947 root158Tail6 winner =
    scan edge (search edge 3) 4 [947] 947 root158Tail7 winner := by
  rw [root158Tail6, scan, if_neg (by decide), root158Candidates6_checked]
  change scan edge (search edge 3) 4 [947] 947 root158Tail7
    (if 3 ≤ root158Candidates6.length then search edge 3 [947, 3491] root158Candidates6 4438 winner else winner) = _
  rw [if_neg (by decide)]
def root158Candidates7 : List Nat := [5981, 16901]
set_option maxRecDepth 8192 in
theorem root158Candidates7_checked : candidates edge 4 947 3623 root158Tail8 winner.bound = root158Candidates7 := by
  unfold candidates
  have row : edge 3623 = (SortedNatTree.contains FastRows.row3623) := by
    change Data.row3623.contains = (SortedNatTree.contains FastRows.row3623)
    exact FastRows.row3623_checked.symm
  rw [row]
  decide
theorem root158Step7 : scan edge (search edge 3) 4 [947] 947 root158Tail7 winner =
    scan edge (search edge 3) 4 [947] 947 root158Tail8 winner := by
  rw [root158Tail7, scan, if_neg (by decide), root158Candidates7_checked]
  change scan edge (search edge 3) 4 [947] 947 root158Tail8
    (if 3 ≤ root158Candidates7.length then search edge 3 [947, 3623] root158Candidates7 4570 winner else winner) = _
  rw [if_neg (by decide)]
def root158Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root158Candidates8_checked : candidates edge 4 947 4523 root158Tail9 winner.bound = root158Candidates8 := by
  unfold candidates
  have row : edge 4523 = (SortedNatTree.contains FastRows.row4523) := by
    change Data.row4523.contains = (SortedNatTree.contains FastRows.row4523)
    exact FastRows.row4523_checked.symm
  rw [row]
  decide
theorem root158Step8 : scan edge (search edge 3) 4 [947] 947 root158Tail8 winner =
    scan edge (search edge 3) 4 [947] 947 root158Tail9 winner := by
  rw [root158Tail8, scan, if_neg (by decide), root158Candidates8_checked]
  change scan edge (search edge 3) 4 [947] 947 root158Tail9
    (if 3 ≤ root158Candidates8.length then search edge 3 [947, 4523] root158Candidates8 5470 winner else winner) = _
  rw [if_neg (by decide)]
def root158Candidates9 : List Nat := [5009, 20261]
set_option maxRecDepth 8192 in
theorem root158Candidates9_checked : candidates edge 4 947 4637 root158Tail10 winner.bound = root158Candidates9 := by
  unfold candidates
  have row : edge 4637 = (SortedNatTree.contains FastRows.row4637) := by
    change Data.row4637.contains = (SortedNatTree.contains FastRows.row4637)
    exact FastRows.row4637_checked.symm
  rw [row]
  decide
theorem root158Step9 : scan edge (search edge 3) 4 [947] 947 root158Tail9 winner =
    scan edge (search edge 3) 4 [947] 947 root158Tail10 winner := by
  rw [root158Tail9, scan, if_neg (by decide), root158Candidates9_checked]
  change scan edge (search edge 3) 4 [947] 947 root158Tail10
    (if 3 ≤ root158Candidates9.length then search edge 3 [947, 4637] root158Candidates9 5584 winner else winner) = _
  rw [if_neg (by decide)]
def root158Candidates10 : List Nat := [5009, 12953]
set_option maxRecDepth 8192 in
theorem root158Candidates10_checked : candidates edge 4 947 4931 root158Tail11 winner.bound = root158Candidates10 := by
  unfold candidates
  have row : edge 4931 = (SortedNatTree.contains FastRows.row4931) := by
    change Data.row4931.contains = (SortedNatTree.contains FastRows.row4931)
    exact FastRows.row4931_checked.symm
  rw [row]
  decide
theorem root158Step10 : scan edge (search edge 3) 4 [947] 947 root158Tail10 winner =
    scan edge (search edge 3) 4 [947] 947 root158Tail11 winner := by
  rw [root158Tail10, scan, if_neg (by decide), root158Candidates10_checked]
  change scan edge (search edge 3) 4 [947] 947 root158Tail11
    (if 3 ≤ root158Candidates10.length then search edge 3 [947, 4931] root158Candidates10 5878 winner else winner) = _
  rw [if_neg (by decide)]
def root158Candidates11 : List Nat := [6947, 18233]
set_option maxRecDepth 8192 in
theorem root158Candidates11_checked : candidates edge 4 947 5009 root158Tail12 winner.bound = root158Candidates11 := by
  unfold candidates
  have row : edge 5009 = (SortedNatTree.contains FastRows.row5009) := by
    change Data.row5009.contains = (SortedNatTree.contains FastRows.row5009)
    exact FastRows.row5009_checked.symm
  rw [row]
  decide
theorem root158Step11 : scan edge (search edge 3) 4 [947] 947 root158Tail11 winner =
    scan edge (search edge 3) 4 [947] 947 root158Tail12 winner := by
  rw [root158Tail11, scan, if_neg (by decide), root158Candidates11_checked]
  change scan edge (search edge 3) 4 [947] 947 root158Tail12
    (if 3 ≤ root158Candidates11.length then search edge 3 [947, 5009] root158Candidates11 5956 winner else winner) = _
  rw [if_neg (by decide)]
def root158Candidates12 : List Nat := [17729, 17783, 18233]
set_option maxRecDepth 8192 in
theorem root158Candidates12_checked : candidates edge 4 947 5051 root158Tail13 winner.bound = root158Candidates12 := by
  unfold candidates
  have row : edge 5051 = (SortedNatTree.contains FastRows.row5051) := by
    change Data.row5051.contains = (SortedNatTree.contains FastRows.row5051)
    exact FastRows.row5051_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root158Child12 : search edge 3 [947, 5051] root158Candidates12 5998 winner = winner := by decide
theorem root158Step12 : scan edge (search edge 3) 4 [947] 947 root158Tail12 winner =
    scan edge (search edge 3) 4 [947] 947 root158Tail13 winner := by
  rw [root158Tail12, scan, if_neg (by decide), root158Candidates12_checked]
  change scan edge (search edge 3) 4 [947] 947 root158Tail13
    (if 3 ≤ root158Candidates12.length then search edge 3 [947, 5051] root158Candidates12 5998 winner else winner) = _
  rw [if_pos (by decide), root158Child12]
def root158Candidates13 : List Nat := [14549]
set_option maxRecDepth 8192 in
theorem root158Candidates13_checked : candidates edge 4 947 5981 root158Tail14 winner.bound = root158Candidates13 := by
  unfold candidates
  have row : edge 5981 = (SortedNatTree.contains FastRows.row5981) := by
    change Data.row5981.contains = (SortedNatTree.contains FastRows.row5981)
    exact FastRows.row5981_checked.symm
  rw [row]
  decide
theorem root158Step13 : scan edge (search edge 3) 4 [947] 947 root158Tail13 winner =
    scan edge (search edge 3) 4 [947] 947 root158Tail14 winner := by
  rw [root158Tail13, scan, if_neg (by decide), root158Candidates13_checked]
  change scan edge (search edge 3) 4 [947] 947 root158Tail14
    (if 3 ≤ root158Candidates13.length then search edge 3 [947, 5981] root158Candidates13 6928 winner else winner) = _
  rw [if_neg (by decide)]
theorem root158Scan14 : scan edge (search edge 3) 4 [947] 947 root158Tail14 winner = winner := by rw [root158Tail14, scan, if_pos (by decide)]
theorem root158Scan13 : scan edge (search edge 3) 4 [947] 947 root158Tail13 winner = winner :=
  (root158Step13).trans root158Scan14
theorem root158Scan12 : scan edge (search edge 3) 4 [947] 947 root158Tail12 winner = winner :=
  (root158Step12).trans root158Scan13
theorem root158Scan11 : scan edge (search edge 3) 4 [947] 947 root158Tail11 winner = winner :=
  (root158Step11).trans root158Scan12
theorem root158Scan10 : scan edge (search edge 3) 4 [947] 947 root158Tail10 winner = winner :=
  (root158Step10).trans root158Scan11
theorem root158Scan9 : scan edge (search edge 3) 4 [947] 947 root158Tail9 winner = winner :=
  (root158Step9).trans root158Scan10
theorem root158Scan8 : scan edge (search edge 3) 4 [947] 947 root158Tail8 winner = winner :=
  (root158Step8).trans root158Scan9
theorem root158Scan7 : scan edge (search edge 3) 4 [947] 947 root158Tail7 winner = winner :=
  (root158Step7).trans root158Scan8
theorem root158Scan6 : scan edge (search edge 3) 4 [947] 947 root158Tail6 winner = winner :=
  (root158Step6).trans root158Scan7
theorem root158Scan5 : scan edge (search edge 3) 4 [947] 947 root158Tail5 winner = winner :=
  (root158Step5).trans root158Scan6
theorem root158Scan4 : scan edge (search edge 3) 4 [947] 947 root158Tail4 winner = winner :=
  (root158Step4).trans root158Scan5
theorem root158Scan3 : scan edge (search edge 3) 4 [947] 947 root158Tail3 winner = winner :=
  (root158Step3).trans root158Scan4
theorem root158Scan2 : scan edge (search edge 3) 4 [947] 947 root158Tail2 winner = winner :=
  (root158Step2).trans root158Scan3
theorem root158Scan1 : scan edge (search edge 3) 4 [947] 947 root158Tail1 winner = winner :=
  (root158Step1).trans root158Scan2
theorem root158Scan0 : scan edge (search edge 3) 4 [947] 947 root158Tail0 winner = winner :=
  (root158Step0).trans root158Scan1
set_option maxRecDepth 4096 in
theorem root158Branch : search edge 4 [947] root158Candidates 947 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [947] 947 root158Tail0 winner = winner
  exact root158Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
